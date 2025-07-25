import socket
from config.create_app import create_app
from flask_restful import Api
from flask_cors import CORS
from src.uploadToOpenSearch import UploadToOpenSearch
from src.uploadToOpenSearchS3 import UploadToOpenSearchS3

env = socket.gethostbyname(socket.gethostname())
if 'ai.chatbot.com' in env :
    app = create_app('prd')
    print('운영')
else:
    app = create_app('dev')
    print('개발')

api = Api(app)
api.add_resource(UploadToOpenSearch, '/api/opensearch/upload')
api.add_resource(UploadToOpenSearchS3, '/api/opensearch/uploadS3')

if __name__ == "__main__":
    CORS(app)
    app.run(debug=True, port=5001, host='0.0.0.0')