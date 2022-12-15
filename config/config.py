import os

from TMTChatbot.Common.common_keys import *
from TMTChatbot.Common.config.config import Config as BaseConfig

from common.common_keys import *


class Config(BaseConfig):
    def __init__(self):

        super(Config, self).__init__()
        self.api_port = int(os.getenv(API_PORT, 35515))
        self.request_url_timeout = int(os.getenv(REQUEST_URL_TIMEOUT, 6))
        self.max_retries = int(os.getenv(MAX_RETRIES, 2))

        self.mongo_host = os.getenv(MONGO_HOST, "172.29.13.23")
        self.mongo_port = int(os.getenv(MONGO_PORT, 20253))
        self.mongo_username = os.getenv(MONGO_USERNAME, "admin")
        self.mongo_password = os.getenv(MONGO_PASSWORD, "admin")
        # self.mongo_database = os.getenv(MONGO_DATABASE, "FaceDatabase")
        # self.mongo_collection = os.getenv(MONGO_COLLECTION, "FaceInformation")
        self.mongodb_name = os.getenv(MONGO_DATABASE, "test")
        self.mongo_collection_name = os.getenv(MONGO_COLLECTION, "test")
        self.mongo_collection_path = f"mongodb://{self.mongo_username}:{self.mongo_password}@{self.mongo_host}:{self.mongo_port}"

        self.model_face_detection_path = os.getenv(MODEL_FACE_DETECTION_PATH, 'libs/weights/face_detection/mnet.25')
        self.model_face_recognition_path = os.getenv(MODEL_FACE_RECOGNITION_PATH,
                                                     'libs/weights/face_recognition/glint360k_r100FC_1.0/model')
        self.register_logger_name = os.getenv(REGISTER_LOGGER_NAME, 'register_logging')

        self.qdrant_host = os.getenv(QDRANT_HOST, "172.29.13.23")
        self.qdrant_port = int(os.getenv(QDRANT_PORT, 6333))
        self.qdrant_collection_name = os.getenv(COLLECTION_NAME, "face_search")
        self.upsert_maxsize = int(os.getenv(UPSERT_MAXSIZE, 5000))
        self.qdrant_timeout = int(os.getenv(QDRANT_TIMEOUT, 20))