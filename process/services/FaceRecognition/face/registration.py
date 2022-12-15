import asyncio
import time
import logging
import hashlib
from io import BytesIO

import cv2
import numpy as np
import aiohttp
from qdrant_client import QdrantClient
from qdrant_client.http import models

from process.services.FaceRecognition.utils.model_utils import FaceDetectionModel, FaceRecognitionModel
# , FaceMeshModel
from process.mongo_client import MongoDb
from support_class import BaseServiceSingleton
from config.config import Config
from common.common_keys import *
# from process.services.FaceRecognition.face.modules.sort import *


class Registration(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(Registration, self).__init__(config=config)
        self.face_detection_model = FaceDetectionModel(config=config)
        self.face_recognition_model = FaceRecognitionModel(config=config)
        # self.face_mesh_model = FaceMeshModel(config=config)
        self.mongo = MongoDb()
        self.mongo.get_database()
        self.client = QdrantClient(host=self.config.qdrant_host, port=self.config.qdrant_port)
        self.create_collection()
        self.face_database = None
        self.logger = logging.getLogger(self.config.register_logger_name)

    @staticmethod
    def hash(s: str) -> int:
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

    def create_collection(self):
        try:
            self.client.http.collections_api.get_collection(self.config.qdrant_collection_name).dict()
            self.logger.info(f"Collection {self.config.qdrant_collection_name} already exist in Qdrant")
        except Exception as e:
            self.logger.info(f"Collection {self.config.qdrant_collection_name} does not exist in Qdrant: {e}")
            self.client.recreate_collection(
                collection_name=self.config.qdrant_collection_name,
                vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
                timeout=self.config.qdrant_timeout
            )

    async def get_decode_image_from_url(self, session, url):
        async with session.get(url) as response:
            start = time.time()
            buffer = BytesIO(await response.read())
            arr = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            self.logger.info(f"Time get & decode image:{time.time() - start}")
            return img

    async def get_decode_image_from_list_url(self, sites):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in sites:
                task = asyncio.ensure_future(self.get_decode_image_from_url(session, url))
                tasks.append(task)
            img_from_list_url = await asyncio.gather(*tasks, return_exceptions=True)
            return img_from_list_url

    def add_insert_remove_new_image_from_app(self, add_insert_remove, list_image_url, personal_information):
        embedded_face = None
        if add_insert_remove == 'add':
            embedded_face, data = [], []
            st = time.time()
            img_ls = asyncio.new_event_loop().run_until_complete(self.get_decode_image_from_list_url(list_image_url))
            self.logger.info(f"get and decode image with async:{time.time() - st}")
            self.logger.info(len(img_ls))
            for i in range(len(list_image_url)):
                image = img_ls[i]
                st = time.time()
                crop_faces, _, _, transformed_landmark, org_crop = self.face_detection_model.get_face_area(
                    image,
                    threshold=0.7,
                    scales=[360, 640])
                self.logger.info(f"Time detect face: {time.time() - st}")
                st = time.time()
                face_embedded = self.face_recognition_model.get_face_embeded(crop_faces[0])
                # face_embedded = np.random.random_sample((512,))
                self.logger.info(f"Time embedding face: {time.time() - st}")
                st = time.time()
                embedded = face_embedded.tolist()
                if isinstance(embedded, list):
                    embedded_face.append([embedded])
                self.logger.info(f"Time add emb {0}: {time.time() - st}")

                data.append(models.PointStruct(id=self.hash(personal_information[PERSONAL_ID] +
                                                            personal_information[PERSONAL_NAME] + list_image_url[i]),
                                               payload={PERSONAL_ID: personal_information[PERSONAL_ID],
                                                        PERSONAL_NAME: personal_information[PERSONAL_NAME],
                                                        IMAGE_URL: list_image_url[i]},
                                               vector=embedded))
            self.face_database = self.mongo.add_db(
                embedded=embedded_face,
                list_img_url=list_image_url,
                personal_information=personal_information)
            self.client.upsert(collection_name=self.config.qdrant_collection_name, points=data, wait=True)
            self.logger.info("Time add face to qdrant:{time.time() - st}")

        else:
            if add_insert_remove == 'remove':
                self.face_database = self.mongo.remove_db(personal_information=personal_information)
                self.client.delete(
                    collection_name=self.config.qdrant_collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[models.FieldCondition(key=PERSONAL_ID,
                                                        match=models.MatchValue(
                                                            value=personal_information[PERSONAL_ID])),
                                  models.FieldCondition(key=PERSONAL_NAME,
                                                        match=models.MatchValue(
                                                            value=personal_information[PERSONAL_NAME]))]
                        )
                    )
                )
                embedded_face = []
        # self.logger.info("Finished add_db")
        return add_insert_remove, embedded_face
