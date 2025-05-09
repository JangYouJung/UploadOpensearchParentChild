
# OpenSearch Document Processing and Retrieval 

## 🌏 Langauge
* [English](#English)
* [한국어](#한국어)


# English
A comprehensive document processing system that extracts text, tables, and images from complex PDF documents, processes them using AWS Bedrock for AI capabilities, and indexes them in OpenSearch for efficient retrieval and querying.

This project combines advanced document parsing with AI-powered processing to create a searchable knowledge base from PDF documents. It leverages AWS services including Bedrock for AI/ML capabilities and OpenSearch for scalable document storage and retrieval. The system supports hybrid search combining semantic and lexical search approaches, making it ideal for enterprise document management and knowledge retrieval applications.

### Note  
This is the extracted source code from the file `20_applications/02_qa_chatbot/01_preprocess_docs/05_0_load_default_complex_pdf_kr_opensearch.ipynb` in the [aws-samples/multi-modal-chatbot-with-advanced-rag](https://github.com/aws-samples/multi-modal-chatbot-with-advanced-rag) repository.  
It has been modified to run locally without uploading it to SageMaker.


## 📁 Repository Structure
```
.
├── config/                      # Configuration management
│   ├── config.py               # Environment-specific configurations
│   └── create_app.py           # Flask application factory
├── models/                     # Data models
│   └── chat_model.py           # SQLAlchemy chat model definition
├── src/                        # Core source code
│   ├── uploadToOpenSearch.py   # Main document processing and indexing logic
│   └── utils/                  # Utility modules
│       ├── bedrock.py         # AWS Bedrock integration
│       ├── chat.py            # Chat functionality implementation
│       ├── chunk.py           # Document chunking utilities
│       ├── opensearch.py      # OpenSearch operations
│       ├── proc_docs.py       # Document processing utilities
│       ├── rag.py            # Retrieval Augmented Generation
│       ├── s3.py             # AWS S3 operations
│       └── text_to_report.py # Text to chart conversion
├── requirements.txt           # Project dependencies
└── server.py                 # Flask server entry point
```

## 🚀 Usage Instructions
### 📌 Prerequisites
- Python 3.8+
- AWS Account with access to:
  - Bedrock
  - OpenSearch
  - S3
- OpenSearch domain endpoint
- AWS credentials configured

**Required Python packages**
```
boto3>=1.34
flask>=3.1
langchain>=0.2.5
opensearch-py>=2.6.0
PyMuPDF>=1.25.5
```

### ⚙️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. (Option)Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure OpenSearch:
Create config.ini with:
```ini
[OpenSearch]
ENDPOINT=<your-opensearch-endpoint>
NAME=<admin-username>
PWD=<admin-password>
REGION=<aws-region>
```

---

### 🏁  Quick Start

1. Start the server:
```bash
python server.py
```

2. Upload a document:
```python
import requests

data = {
    'file_path': 'YOUR_LOCAL_FILE_PATH.pdf',  # Local path to the PDF file
    'index': 'YOUR_OPENSEARCH_INDEX_NAME'     # Index name (will be created if it doesn't exist)
}

response = requests.post(
    'http://localhost:5000/api/opensearch/upload',
    data=data  # Send as form-data
)

print("Status Code:", response.status_code)
print("Response:", response.text)
```
⚠️ Note: This process may take up to 10 minutes depending on the size and complexity of the document.

---

### 💡 More Detailed Examples

### Process and index a document with custom chunking:
```python
from src.utils.chunk import create_chunk
from src.utils.proc_docs import insert_chunk_opensearch

# Create chunks
chunks = create_chunk(documents, chunk_size=1000, chunk_overlap=100)

# Index chunks
insert_chunk_opensearch(chunks, embeddings_model, os_client, index_name)
```

---


### 🛠️ Troubleshooting

1. OpenSearch Connection Issues
```bash
# Check OpenSearch endpoint
curl -X GET https://<opensearch-endpoint>/_cluster/health
```

- Verify security group settings
- Check credentials in config.ini
- Ensure AWS region matches OpenSearch domain

2. Document Processing Errors
- Check PDF permissions
- Verify sufficient memory for large documents
- Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🔄 Data Flow
The system processes documents through a pipeline that extracts content, enriches it with AI, and makes it searchable.

```ascii
[PDF Document] -> [Content Extraction] -> [AI Processing] -> [Chunking] -> [Embedding] -> [OpenSearch]
     |                    |                     |              |             |              |
     v                    v                     v              v             v              v
  Raw File         Text/Tables/Images     AI Enrichment    Segments    Vector Data    Searchable Index
```

### Key component interactions:
1. Document loader extracts structured content from PDFs
2. AWS Bedrock generates embeddings and processes content
3. Chunking system splits documents into optimal segments
4. OpenSearch stores both vector and text data
5. Hybrid retriever combines semantic and lexical search
6. REST API provides document upload and search interface
7. S3 handles document storage and management

---

# 한국어
# PDF 문서 벡터화 및 OpenSearch 업로드 시스템

복잡한 PDF 문서에서 **텍스트, 표, 이미지**를 추출하고, AWS Bedrock을 통해 AI 기반 처리 후 OpenSearch에 벡터화하여 효율적인 검색과 질의가 가능한 문서 처리 시스템입니다.

이 프로젝트는 고급 문서 파싱 기능과 AI 처리 기능을 결합하여 PDF 문서로부터 검색 가능한 지식 기반을 구축합니다. AWS Bedrock(AI/ML 처리)과 OpenSearch(확장 가능한 문서 저장 및 검색)을 활용하며, **의미론적(semantic) + 문자 기반(lexical) 검색**을 결합한 하이브리드 검색을 지원합니다. 이는 기업 문서 관리 및 지식 검색 응용에 적합합니다.

### 참고 
[aws-samples/multi-modal-chatbot-with-advanced-rag](https://github.com/aws-samples/multi-modal-chatbot-with-advanced-rag) 레포지토리의 `20_applications\02_qa_chatbot\01_preprocess_docs\05_0_load_default_complex_pdf_kr_opensearch.ipynb` 파일을 Sagemaker에 올리지 않아도 로컬에서 실행할 수 있도록 추출한 소스파일입니다. 

---

## 📁 저장소 구조

```
.
├── config/                      # 설정 관리
│   ├── config.py               # 환경별 설정
│   └── create_app.py           # Flask 애플리케이션 팩토리
├── models/                     # 데이터 모델
│   └── chat_model.py           # SQLAlchemy 채팅 모델 정의
├── src/                        # 핵심 소스 코드
│   ├── uploadToOpenSearch.py   # 문서 처리 및 색인 로직
│   └── utils/                  # 유틸리티 모듈
│       ├── bedrock.py         # AWS Bedrock 통합
│       ├── chat.py            # 챗 기능 구현
│       ├── chunk.py           # 문서 청크 처리 유틸
│       ├── opensearch.py      # OpenSearch 연동
│       ├── proc_docs.py       # 문서 처리 유틸
│       ├── rag.py            # RAG 처리 로직
│       ├── s3.py             # S3 연동 기능
│       └── text_to_report.py # 텍스트 → 차트 변환
├── requirements.txt           # 프로젝트 의존성
└── server.py                 # Flask 서버 진입점

```

---

## 🚀 사용법

### 📌 사전 준비

- Python 3.8 이상
- AWS 계정 및 서비스 권한
  - Bedrock
  - OpenSearch
  - S3
- OpenSearch 도메인 엔드포인트
- AWS 자격 증명 설정

**필수 패키지**
```
boto3>=1.34  
flask>=3.1  
langchain>=0.2.5  
opensearch-py>=2.6.0  
PyMuPDF>=1.25.5  
```

---

### ⚙️ 설치

1. 저장소 클론
```bash
git clone <repository-url>
cd <repository-name>
```

2. (선택)가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. OpenSearch 설정 (`config.ini`)
```ini
[OpenSearch]
ENDPOINT=<OpenSearch 엔드포인트>
NAME=<관리자 계정>
PWD=<비밀번호>
REGION=<AWS 리전>
```

---

### 🏁 빠른 시작

1. 서버 실행
```bash
python server.py
```

2. 문서 업로드 API 예제
```python
import requests

data = {
    'file_path': 'YOUR_LOCAL_FILE_PATH.pdf',  # 로컬 파일 경로
    'index': 'YOUR_OPENSEARCH_INDEX_NAME' # 인덱스 이름 ( 없으면 생성됨 )
}

response = requests.post(
    'http://localhost:5000/api/opensearch/upload',
    data=data  # form-data 형식으로 전송
)

print("Status Code:", response.status_code)
print("Response:", response.text)

```
⚠️ 주의: 문서의 크기와 복잡도에 따라 최대 10분까지 소요될 수 있습니다.

---

## 💡 상세 예제

### 문서 청크 분할 및 색인

```python
from src.utils.chunk import create_chunk
from src.utils.proc_docs import insert_chunk_opensearch

chunks = create_chunk(documents, chunk_size=1000, chunk_overlap=100)
insert_chunk_opensearch(chunks, embeddings_model, os_client, index_name)
```

---

## 🛠️ 문제 해결

### OpenSearch 연결 오류
```bash
curl -X GET https://<opensearch-endpoint>/_cluster/health
```

- 보안 그룹 및 포트 확인
- `config.ini` 자격 증명 확인
- AWS 리전 일치 확인

### 문서 처리 오류

- PDF 권한 확인
- 메모리 부족 여부 확인
- 디버그 로그 활성화
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🔄 데이터 흐름

```ascii
 [PDF 문서] -> [콘텐츠 추출]  ->  [AI 처리]  ->  [청크 분할] -> [임베딩] -> [OpenSearch 색인]
      |               |             |              |           |              |
      v               v             v              v           v              v
   원본 파일    텍스트/표/이미지   AI 강화 콘텐츠   문서 조각    벡터 데이터     검색 가능한 인덱스
```

### 주요 처리 흐름
1. PDF 로더가 구조화된 콘텐츠 추출
2. AWS Bedrock이 임베딩 생성 및 AI 처리
3. 청크 시스템이 문서를 세그먼트로 분할
4. OpenSearch에 벡터 및 텍스트 색인
5. 하이브리드 검색기: 의미 기반 + 키워드 기반 결합
6. REST API: 업로드 및 검색 인터페이스 제공
7. S3: 문서 저장소

---
