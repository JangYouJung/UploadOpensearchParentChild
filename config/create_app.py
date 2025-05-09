from . import config
from flask import Flask

def create_app(environment):
    # Config파일에서 불러오기
     config_map = {
          'dev': config.Development(),
          'prd': config.Production()
     }
     config_obj = config_map[environment.lower()]

     app = Flask(__name__)
     app.config.from_object(config_obj)

     return app