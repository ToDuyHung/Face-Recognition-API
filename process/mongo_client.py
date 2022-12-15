# import pymongo
# from support_class import BaseServiceSingleton
# from config.config import Config
# from common.common_keys import *


# class MongoDb(BaseServiceSingleton):
#     def __init__(self, config: Config = None):
#         super(MongoDb, self).__init__(config=config)
#         self.mongo_connector = pymongo.MongoClient(host=config.mongo_host,
#                                     port=config.mongo_port,
#                                     username=config.mongo_username,
#                                     password=config.mongo_password)
#         self.mycol = self.mongo_connector[config.mongo_database][config.mongo_collection]


from pymongo import MongoClient
from support_class import BaseServiceSingleton
from config.config import Config
from common.common_keys import *
from config.constant import *
import time

class MongoDb(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(MongoDb, self).__init__(config=config)
        self.db = None
        self.mydb = {}
        self.mycol = {}

    def get_database(self):
        self.db = []
        client = MongoClient(self.config.mongo_collection_path)
        self.mydb = client[self.config.mongodb_name]
        self.mycol = self.mydb[self.config.mongo_collection_name]

        for x in self.mycol.find():
            x = {ID: x[ID], NAME: x[NAME]}
            self.db.append(x)
    
                
    def get_features(self, _id):
        feature = self.mycol.find_one({ID: _id})[FEATURES]
        return feature


    def add_db(self,embedded,personal_information = None, list_img_url=None):
        st = time.time()
        x = {ID: personal_information['id'], NAME: personal_information['name']}
        self.db.append(x)
        print("Time get database:", time.time() - st)
        st = time.time()
        self.mycol.insert_one({"_id":personal_information['id'],
        "name":personal_information['name'], 
        "code":personal_information['code'],
        "branchID":personal_information['branchID'],
        "departmentId":personal_information['departmentId'],
        "img_url": list_img_url, 
        "features":embedded})   

        print("Time insert database:", time.time() - st)

    
    def insert_db(self, embedded, list_img_url, personal_information):

        self.get_database() 
        self.mycol.update_one({"_id":personal_information['id']},{"$set":{"features":embedded}})
        self.mycol.update_one({"_id":personal_information['id']},{"$set":{"img_url":list_img_url}})
        
    def remove_db(self, personal_information = None):
        myquery = {"_id":personal_information['id']}
        self.mycol.delete_one(myquery)
        print("đã xóa")