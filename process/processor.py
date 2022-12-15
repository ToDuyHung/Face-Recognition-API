from time import perf_counter
from support_class import (
    BaseDataModel,
    BaseServiceSingleton
)

from data_models.data_models import InsertUpdateRequest, Response, DeleteRequest

from config.config import Config
from process.mongo_client import MongoDb
from process.face_pipeline import FacePipeline
from common.common_keys import *


class Processor(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(Processor, self).__init__(config=config)
        self.mongo = MongoDb(config=config)
        self.mongo.get_database()
        self.pipeline = FacePipeline(config=config)

    def process(self, input_data: BaseDataModel):
        pass

    def handler_information(self, data: dict):
        msg, embedded_face = self.pipeline.add_insert_remove(data)
        return msg, embedded_face

    async def process_add(self, input_data: InsertUpdateRequest):
        response = Response()
        request = {LIST_URL_IMG: input_data.list_url_img, PERSONAL_INFORMATION: input_data.personal_information.dict()}
        my_query = {"_id": request['personal_information']['id']}
        check_exist = self.mongo.mycol.count_documents(my_query)
        if check_exist:
            response.result = {
                'metadata': {
                    'status': 402,
                    'processing_time': "0.00 s"
                },
                'message': "Người dùng đã tồn tại"
            }
            return response
        request.update({"method": "add"})
        print(request)
        start = perf_counter()
        msg, embedded_face = await self.wait(self.handler_information, request)
        print(type(embedded_face), len(embedded_face))
        end = perf_counter()
        processing_time = "%.2f s" % (end - start)
        self.logger.info(msg)
        response.result = {
            'metadata': {
                'status': 200,
                'processing_time': processing_time
            },
            'message': msg
        }
        return response

    async def process_update(self, input_data: InsertUpdateRequest):
        response = Response()
        request = {PERSONAL_INFORMATION: input_data.personal_information.dict()}
        request.update({"method": "remove"})
        start = perf_counter()
        _, _ = await self.wait(self.handler_information, request)
        request = {LIST_URL_IMG: input_data.list_url_img, PERSONAL_INFORMATION: input_data.personal_information.dict()}
        request.update({"method": "add"})
        msg, embedded_face = await self.wait(self.handler_information, request)
        end = perf_counter()
        processing_time = "%.2f s" % (end - start)
        self.logger.info(msg)
        response.result = {
            'metadata': {
                'status': 200,
                'processing_time': processing_time
            },
            'message': msg
        }
        return response

    async def process_delete(self, input_data: DeleteRequest):
        response = Response()
        request = {PERSONAL_INFORMATION: input_data.personal_information.dict()}
        print(request)
        request.update({"method": "remove"})
        print(request)
        start = perf_counter()
        msg, embedded_face = await self.wait(self.handler_information, request)
        end = perf_counter()
        processing_time = "%.2f s" % (end - start)
        self.logger.info(msg)
        response.result = {
            'metadata': {
                'status': 200,
                'processing_time': processing_time
            },
            'message': msg
        }
        return response
