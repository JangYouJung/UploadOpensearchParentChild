#!/usr/bin/env python
# coding: utf-8

"""
Preprocessing for complex PDF

This script processes complex PDF documents, extracts text, tables, and images,
and loads them into OpenSearch for retrieval.

실행 전 config.ini 파일을 확인해 오픈서치와 베드락 리전 정보를 체크해야 한다.
오픈서치 엔드포인트, 오픈서치 관리자 정보, 리전(Bedrock 모델 활성 여부 확인 필요) 등 

"""

# Standard library imports
import sys
import os
import json
import time
import math
import shutil
import base64
from glob import glob
from io import BytesIO
from pprint import pprint

# Third-party imports
import boto3
import fitz
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import botocore
from termcolor import colored
from pdf2image import convert_from_path
from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace
from flask_restful import Resource
from flask import request
from itertools import chain

# LangChain imports
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch

# Now import local modules after path setup
from src.utils import bedrock, print_ww
from src.utils.bedrock import bedrock_info
from src.utils.common_utils import to_pickle, load_pickle, print_html, to_markdown, retry
from src.utils.pymupdf import to_markdown_pymupdf
from src.utils.opensearch import opensearch_utils
from src.utils.chunk import parant_documents
from src.utils.rag import qa_chain, prompt_repo, show_context_used, retriever_utils, OpenSearchHybridSearchRetriever
import configparser

# 오픈서치 환경변수 
config = configparser.ConfigParser()
config.read('config.ini')
opensearch_domain_endpoint = config['OpenSearch']['ENDPOINT']
opensearch_user_id = config['OpenSearch']['NAME']
opensearch_user_password = config['OpenSearch']['PWD']
aws_region = config['OpenSearch']['REGION']

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def add_python_path(module_path):
    """Add a path to the Python system path if it doesn't already exist."""
    if os.path.abspath(module_path) not in sys.path:
        sys.path.append(os.path.abspath(module_path))
        print(f"python path: {os.path.abspath(module_path)} is added")
    else:
        print(f"python path: {os.path.abspath(module_path)} already exists")
    print("sys.path: ", sys.path)

# Setup paths
module_path = "../../.."
add_python_path(module_path)


class UploadToOpenSearch(Resource): 
    def post(self):
        """Main function to orchestrate the workflow."""

        # Setup paths
        file_path = request.form.get("file_path")
        index_name = request.form.get("index")
        image_path = "./fig"
        
        # Create Bedrock client
        boto3_bedrock = create_bedrock_client()
        
        # Load models
        llm_text, llm_emb, dimension = load_models(boto3_bedrock)
        
        # Prepare data
        print("Preparing data...")
        texts, tables_preprocessed, images_preprocessed = prepare_data(
            file_path=file_path,
            image_path=image_path,
            llm_text=llm_text,
            table_by_pymupdf=False,
            table_as_image=True
        )
        
        # Set index
        print(colored("Set Opensearch index...", "green"))
        index_name, os_client, opensearch_domain_endpoint, http_auth = set_index(dimension, index_name)
        
        # Create vector store
        print(colored("Creating vector store...", "green"))
        vector_db = create_vector_store(index_name, opensearch_domain_endpoint, http_auth, llm_emb)
        
        # Process documents for indexing
        print(colored("Processing documents for indexing...", "green"))
        process_documents_for_indexing(texts, tables_preprocessed, images_preprocessed, vector_db, os_client, index_name)
        
        print(colored("Processing complete!", "green"))

        return {'message': 'Processed'}, 201
    

def image_to_base64(image_path):
    """Convert an image file to base64 encoding."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')


@retry(total_try_cnt=5, sleep_in_sec=10, retryable_exceptions=(botocore.exceptions.EventStreamError))
def summary_img(summarize_chain, img_base64):
    """Generate a summary of an image using the provided chain."""
    img = Image.open(BytesIO(base64.b64decode(img_base64)))
    plt.imshow(img)
    plt.show()

    summary = summarize_chain.invoke(
        {
            "image_base64": img_base64
        }
    )
    return summary


def show_opensearch_doc_info(response):
    """Display information about an OpenSearch document."""
    print("opensearch document id:" , response["_id"])
    print("family_tree:" , response["_source"]["metadata"]["family_tree"])
    print("parent document id:" , response["_source"]["metadata"]["parent_id"])
    print("parent document text: \n" , response["_source"]["text"])


def create_bedrock_client():
    """Create and configure the Bedrock client."""
    boto3_bedrock = bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=aws_region,
    )
    
    return boto3_bedrock


def load_models(boto3_bedrock):
    """Load the LLM and embedding models."""
    # Load LLM (Claude-v3-sonnet)
    llm_text = ChatBedrock(
        model_id=bedrock_info.get_model_id(model_name="Claude-V3-Sonnet"),
        client=boto3_bedrock,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs={
            "max_tokens": 2048,
            "stop_sequences": ["\n\nHuman"],
            # "temperature": 0,
            # "top_k": 350,
            # "top_p": 0.999
        }
    )
    
    # Load embedding model
    llm_emb = BedrockEmbeddings(
        client=boto3_bedrock,
        model_id=bedrock_info.get_model_id(model_name="Titan-Embeddings-G1")
    )
    dimension = 1536
    print("Bedrock Embeddings Model Loaded")
    return llm_text, llm_emb, dimension


def prepare_data(file_path, image_path, llm_text, table_by_pymupdf=False, table_as_image=True):
    """
    Prepare data by extracting text, tables, and images from documents.
    
    Args:
        file_path: Path to the PDF file
        image_path: Path to save extracted images
        llm_text: LLM model for summarization
        table_by_pymupdf: Whether to use PyMuPDF for table parsing
        table_as_image: Whether to process tables as images
        
    Returns:
        Tuple of (texts, tables_preprocessed, images_preprocessed)
    """
    # Clean and create image directory
    if os.path.isdir(image_path): 
        shutil.rmtree(image_path)
    os.mkdir(image_path)
    
    # Load and parse document
    loader = UnstructuredFileLoader(
        file_path=file_path,
        chunking_strategy="by_title",
        mode="elements",
        strategy="hi_res",
        hi_res_model_name="yolox",
        # extract_images_in_pdf=True,
        pdf_extract_images=True,    
        pdf_infer_table_structure=True,
        extract_image_block_output_dir=image_path,
        extract_image_block_to_payload=False,
        max_characters=4096,
        new_after_n_chars=4000,
        combine_text_under_n_chars=2000,
        languages=["kor+eng"],
        post_processors=[clean_bullets, clean_extra_whitespace]
    )
    
    print(colored("Loading and parsing document...", "green"))
    docs = loader.load()
    
    # Save parsing data
    to_pickle(docs, "./data/pickle/parsed_unstructured.pkl")
    docs = load_pickle("./data/pickle/parsed_unstructured.pkl")
    
    # Separate documents by type
    tables, texts = [], []
    images = glob(os.path.join(image_path, "*"))
    
    for doc in docs:
        category = doc.metadata["category"]
        if category == "Table": 
            tables.append(doc)
        elif category == "Image": 
            images.append(doc)
        else: 
            texts.append(doc)
    
    images = glob(os.path.join(image_path, "*"))
    print(f' # texts: {len(texts)} \n # tables: {len(tables)} \n # images: {len(images)}')
    
    # Optional: PyMuPDF table parsing
    if table_by_pymupdf:
        pymupdf_res = fitz.open(file_path)
        md_text = to_markdown_pymupdf(pymupdf_res)
        markdown_path = "./data/pickle/parsed_pymupdf.md"
        to_markdown(md_text, markdown_path)
        
        loader = UnstructuredMarkdownLoader(
            file_path=markdown_path,
            mode="elements",
            chunking_strategy="by_title",
            max_characters=4096,
            new_after_n_chars=4000,
            combine_text_under_n_chars=2000,
        )
        docs_pymupdf = loader.load()
        docs_table_pymupdf = [doc for doc in docs_pymupdf if doc.metadata["category"] == "Table"]
    
    # Process tables as images if enabled
    if table_as_image:
        image_tmp_path = os.path.join(image_path, "tmp")
        if os.path.isdir(image_tmp_path): 
            shutil.rmtree(image_tmp_path)
        os.mkdir(image_tmp_path)
        
        # Convert PDF pages to images
        pages = convert_from_path(file_path)
        for i, page in enumerate(pages):
            print(f'pdf page {i}, size: {page.size}')    
            page.save(f'{image_tmp_path}/{str(i+1)}.jpg', "JPEG")
        
        print("========")
        
        # Extract table regions as images
        for idx, table in enumerate(tables):
            points = table.metadata["coordinates"]["points"]
            page_number = table.metadata["page_number"]
            layout_width = table.metadata["coordinates"]["layout_width"]
            layout_height = table.metadata["coordinates"]["layout_height"]
            
            img = cv2.imread(f'{image_tmp_path}/{page_number}.jpg')
            crop_img = img[math.ceil(points[0][1]):math.ceil(points[1][1]), 
                          math.ceil(points[0][0]):math.ceil(points[3][0])]
            table_image_path = f'{image_path}/table-{idx}.jpg'
            cv2.imwrite(table_image_path, crop_img)
            
            print(f'unstructured width: {layout_width}, height: {layout_height}')
            print(f'page_number: {page_number}')
            print("==")
            
            width, height, _ = crop_img.shape
            image_token = width*height/750
            print(f'image: {table_image_path}, shape: {img.shape}, image_token_for_claude3: {image_token}')
            
            # Resize large images
            if image_token > 1500:
                resize_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                print("   - resize_img.shape = {0}".format(resize_img.shape))
                table_image_resize_path = table_image_path.replace(".jpg", "-resize.jpg")
                cv2.imwrite(table_image_resize_path, resize_img)
                os.remove(table_image_path)
                table_image_path = table_image_resize_path
            
            img_base64 = image_to_base64(table_image_path)
            table.metadata["image_base64"] = img_base64
        
        if os.path.isdir(image_tmp_path): 
            shutil.rmtree(image_tmp_path)
        images = glob(os.path.join(image_path, "*"))
        print(f'images: {images}')
    
    # Process and summarize images
    system_prompt = "You are an assistant tasked with describing table and image."
    system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
    
    # For images
    human_prompt = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + "{image_base64}",
            },
        },
        {
            "type": "text",
            "text": '''
                     Given image, give a concise summary.
                     Don't insert any XML tag such as <text> and </text> when answering.
                     Write in Korean.
            '''
        },
    ]
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
    
    prompt = ChatPromptTemplate.from_messages([
        system_message_template,
        human_message_template
    ])
    
    summarize_chain = prompt | llm_text | StrOutputParser()
    
    # Get image base64 encodings
    img_info = [image_to_base64(img_path) for img_path in images if os.path.basename(img_path).startswith("figure")]
    
    # Summarize images
    image_summaries = []
    for idx, img_base64 in enumerate(img_info):
        summary = summary_img(summarize_chain, img_base64)
        image_summaries.append(summary)
        print("\n==")
        print(idx)
    
    # Create image documents with summaries
    images_preprocessed = []
    for img_path, image_base64, summary in zip(images, img_info, image_summaries):
        metadata = {}
        metadata["img_path"] = img_path
        metadata["category"] = "Image"
        metadata["image_base64"] = image_base64
        
        doc = Document(
            page_content=summary,
            metadata=metadata
        )
        images_preprocessed.append(doc)
    
    # For tables
    human_prompt = [
        {
            "type": "text",
            "text": '''
                     Here is the table: <table>{table}</table>
                     Given table, give a concise summary.
                     Don't insert any XML tag such as <table> and </table> when answering.
                     Write in Korean.
            '''
        },
    ]
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
    
    prompt = ChatPromptTemplate.from_messages([
        system_message_template,
        human_message_template
    ])
    
    summarize_chain = {"table": lambda x:x} | prompt | llm_text | StrOutputParser()
    
    # Add PyMuPDF tables if enabled
    if table_by_pymupdf:
        tables = tables + docs_table_pymupdf
    
    # Summarize tables
    table_info = [(t.page_content, t.metadata["text_as_html"]) for t in tables]
    table_summaries = summarize_chain.batch(table_info, config={"max_concurrency": 1})
    
    if table_as_image: 
        table_info = [(t.page_content, t.metadata["text_as_html"], t.metadata["image_base64"]) 
                      if "image_base64" in t.metadata 
                      else (t.page_content, t.metadata["text_as_html"], None) 
                      for t in tables]
    
    # Create table documents with summaries
    tables_preprocessed = []
    for origin, summary in zip(tables, table_summaries):
        metadata = origin.metadata
        metadata["origin_table"] = origin.page_content
        doc = Document(
            page_content=summary,
            metadata=metadata
        )
        tables_preprocessed.append(doc)
    
    return texts, tables_preprocessed, images_preprocessed

def set_index(dimension, index_name):
    """
    Get OpenSearch index info or create an OpenSearch index for document storage.
    
    Args:
        dimension: Dimension of the embedding vectors
        
    Returns:
        Tuple of (index_name, os_client, opensearch_domain_endpoint, http_auth)
    """
    
    # Define index schema
    index_body = {
        'settings': {
            'analysis': {
                'analyzer': {
                    'my_analyzer': {
                        'char_filter':['html_strip'],
                        'tokenizer': 'nori',
                        'filter': [
                            'my_nori_part_of_speech'
                        ],
                        'type': 'custom'
                    }
                },
                'tokenizer': {
                    'nori': {
                        'decompound_mode': 'mixed',
                        'discard_punctuation': 'true',
                        'type': 'nori_tokenizer'
                    }
                },
                "filter": {
                    "my_nori_part_of_speech": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "J", "XSV", "E", "IC","MAJ","NNB",
                            "SP", "SSC", "SSO",
                            "SC","SE","XSN","XSV",
                            "UNA","NA","VCP","VSV",
                            "VX"
                        ]
                    }
                }
            },
            'index': {
                'knn': True,
                'knn.space_type': 'cosinesimil'
            }
        },
        'mappings': {
            'properties': {
                'metadata': {
                    'properties': {
                        'source': {'type': 'keyword'},
                        'page_number': {'type':'long'},
                        'category': {'type':'text'},
                        'file_directory': {'type':'text'},
                        'last_modified': {'type': 'text'},
                        'type': {'type': 'keyword'},
                        'image_base64': {'type':'text'},
                        'origin_image': {'type':'text'},
                        'origin_table': {'type':'text'},
                    }
                },
                'text': {
                    'analyzer': 'my_analyzer',
                    'search_analyzer': 'my_analyzer',
                    'type': 'text'
                },
                'vector_field': {
                    'type': 'knn_vector',
                    'dimension': f"{dimension}"
                }
            }
        }
    }
    
    http_auth = (opensearch_user_id, opensearch_user_password)
    
    # Create OpenSearch client
    os_client = opensearch_utils.create_aws_opensearch_client(
        aws_region,
        opensearch_domain_endpoint,
        http_auth
    )

    # Create or recreate index
    index_exists = opensearch_utils.check_if_index_exists(
        os_client,
        index_name
    )
    
    # 인덱스 없으면 생성하기
    if not index_exists:
        opensearch_utils.create_index(os_client, index_name, index_body)
        print("Index is created")
    
    index_info = os_client.indices.get(index=index_name)
    pprint(index_info)
    
    return index_name, os_client, opensearch_domain_endpoint, http_auth

def create_vector_store(index_name, opensearch_domain_endpoint, http_auth, llm_emb):
    """
    Create a LangChain OpenSearch vector store.
    
    Args:
        index_name: Name of the OpenSearch index
        opensearch_domain_endpoint: OpenSearch endpoint URL
        http_auth: Authentication credentials
        llm_emb: Embedding model
        
    Returns:
        OpenSearchVectorSearch object
    """
    vector_db = OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=f"https://{opensearch_domain_endpoint}",
        embedding_function=llm_emb,
        http_auth=http_auth,
        is_aoss=False,
        engine="faiss",
        space_type="l2",
        bulk_size=100000,
        timeout=60
    )
    
    return vector_db

def process_documents_for_indexing(texts, tables_preprocessed, images_preprocessed, vector_db, os_client, index_name):
    """
    Process documents for indexing in OpenSearch.
    
    Args:
        texts: Text documents
        tables_preprocessed: Processed table documents
        images_preprocessed: Processed image documents
        vector_db: OpenSearch vector store
        os_client: OpenSearch client
        index_name: Name of the index
        
    Returns:
        None
    """
    # Define chunking parameters
    parent_chunk_size = 4096
    parent_chunk_overlap = 0
    
    child_chunk_size = 1024
    child_chunk_overlap = 256
    
    opensearch_parent_key_name = "parent_id"
    opensearch_family_tree_key_name = "family_tree"
    
    # Create parent chunks
    parent_chunk_docs = parant_documents.create_parent_chunk(
        docs=texts,
        parent_id_key=opensearch_parent_key_name,
        family_tree_id_key=opensearch_family_tree_key_name,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap
    )
    print(f'Number of parent_chunk_docs= {len(parent_chunk_docs)}')
    
    # Add documents to OpenSearch
    parent_ids = vector_db.add_documents(
        documents=parent_chunk_docs,
        vector_field="vector_field",
        bulk_size=1000000
    )
    
    # Check document count
    total_count_docs = opensearch_utils.get_count(os_client, index_name)
    print("total count docs: ", total_count_docs)
    
    # Display first document info
    response = opensearch_utils.get_document(os_client, doc_id=parent_ids[0], index_name=index_name)
    show_opensearch_doc_info(response)

    # child_chunk_docs = create_child_chunk(parent_chunk_docs[0:1], parent_ids)
    child_chunk_docs = parant_documents.create_child_chunk(
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        docs=parent_chunk_docs,
        parent_ids_value=parent_ids,
        parent_id_key=opensearch_parent_key_name,
        family_tree_id_key=opensearch_family_tree_key_name
    )

    print(f"Number of child_chunk_docs= {len(child_chunk_docs)}")

    parent_id = child_chunk_docs[0].metadata["parent_id"]
    print("child's parent_id: ", parent_id)
    
    print(colored("\n###### Search parent in OpenSearch ######", "green"))

    response = opensearch_utils.get_document(os_client, doc_id = parent_id, index_name = index_name)
    show_opensearch_doc_info(response)    
    

    # Process tables and images
    # Add tables to parent documents
    for table in tables_preprocessed:
        table.metadata["family_tree"], table.metadata["parent_id"] = "parent_table", "NA"
    
    # Add images to parent documents
    for image in images_preprocessed:
        image.metadata["family_tree"], image.metadata["parent_id"] = "parent_image", "NA"
    
    # Merge all documents
    docs_preprocessed = list(chain(child_chunk_docs, tables_preprocessed, images_preprocessed))
    
    # Add all documents to OpenSearch
    child_ids = vector_db.add_documents(
        documents=docs_preprocessed, 
        vector_field = "vector_field",
        bulk_size=1000000
    )

    print("length of child_ids: ", len(child_ids))
    
    # Check final document count
    total_count_docs = opensearch_utils.get_count(os_client, index_name)
    print("Final total count docs: ", total_count_docs)



