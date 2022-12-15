from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct


import sys
from process.services.FaceRecognition.utils.model_utils import FaceDetectionModel, FaceRecognitionModel
import mxnet as mx
from sklearn import preprocessing
# from configs.constant import *
import cv2
from pytz import timezone 
import numpy as np
import unidecode
from threading import Thread, Lock
from datetime import datetime
import random
# EMBEDDING_DIMENSION = 512
class Evaluation:
    def __init__(self):
        self.face_detection_model = FaceDetectionModel()
        self.face_recognition_model = FaceRecognitionModel()
        self.client = QdrantClient(host="172.29.13.23", port=6333)
        self.threshold = 0.5
        
    def identification(self, image):
        # list_len_embedding, list_person_name, index_faiss = self.get_index_faiss.list_len, self.get_index_faiss.list_id, self.get_index_faiss.index
        result = []
        scales = [640, 1200]
        crop_faces, bounding_boxes, landmarks, transformed_landmark,org = self.face_detection_model.get_face_area(
            image,
            threshold=0.5,
            scales=scales)
        print("so khuon mat: ",len(crop_faces))
        for i in range(len(crop_faces)):
            bounding_box = list(map(int,bounding_boxes[i][0:4]))
            try:
                face_embeded = self.face_recognition_model.get_face_embeded(crop_faces[i])
            except:
                continue
            face_embeded = face_embeded.tolist()
            # print(len(face_embeded))
            search_result = self.client.search(
                                collection_name="face_search",

                                query_vector=face_embeded,
                                limit=1
                            )
            if search_result:
                if search_result[0].score > self.threshold:
                    result = search_result[0].payload['name']
                else:
                    result = 'stranger'
                print("Score:", search_result[0].score, result)
            else:
                result = 'stranger'
            
            [cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)]
            [cv2.putText(image, "{}".format(unidecode.unidecode(result)), (bounding_box[0]+30, bounding_box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)]

            # print("thoi gian luu anh:",str(datetime.now(tz=timezone('Asia/Saigon')).strftime("%d_%m_%Y_%H_%M_%S")))

        cv2.imwrite("/AIHCM/ComputerVision/hungtd/FaceRecognitionAPI/test/{}_{}.jpg".format(
            str(datetime.now(tz=timezone('Asia/Saigon')).strftime("%d_%m_%Y_%H_%M_%S")),
            random.randint(1,10000)),
            image)
            # except:
            #     cv2.imwrite('error_face.jpg', org[i])
        return result, bounding_boxes
    
if __name__ == "__main__":
    evaluate = Evaluation()
    from glob import glob
    image_path = glob("C:/Users/thanglq/Documents/Cong_ty/FaceRecognition/face/data/image/*.png")
    # for i in image_path:
    # # i = "C:/Users/thanglq/Documents/Cong_ty/FaceRecognition/face/data/ttt.png"
    #     image = cv2.imread(i)
    #     evaluate.identification(image)

    image = cv2.imread("/AIHCM/ComputerVision/hungtd/FaceRecognitionAPI/test/66282282418176.jpg")
    result, bounding_boxes = evaluate.identification(image)
    # print(result)


