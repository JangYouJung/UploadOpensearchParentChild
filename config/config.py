import os
dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

class Development(object):
    DEBUG = True
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    FLASK_APP = 'UploadOpenSearch_Dev'
    ENY = 'dev'

    ## JSON에서 한글 표현을 위해서 반영
    JSON_AS_ASCII = False


class Production(object):
    DEBUG = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    FLASK_APP = 'UploadOpenSearch_Prd'
    ENY = 'prd'

    ## JSON에서 한글 표현을 위해서 반영
    JSON_AS_ASCII = False


app_config = {
    'dev': Development(),
    'prd': Production(),
}