from support_class import BaseServiceSingleton
from process.services.FaceRecognition.face.registration import Registration
from config.config import Config
from common.common_keys import *


class FacePipeline(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(FacePipeline, self).__init__(config=config)
        self.registration = Registration(config=config)
        self.GlobalRegister = dict()
        self.GlobalUrl = dict()
    
    def add_insert_remove(self, data) -> dict:
        # time.sleep(0.001)
        # data = queue.get()
        msg = ''
        signal_status = data['method']
        if signal_status == 'add':
            self.GlobalRegister[data['personal_information']['id']] = 'processing'
            add_insert_remove, embedded_face = self.registration.add_insert_remove_new_image_from_app(data['method'],
                                                                   list_image_url=data['list_url_img'],
                                                                   personal_information=data['personal_information'])
            for url in self.GlobalUrl:
                self.GlobalUrl[url]['modify_event'].set()
            self.GlobalRegister[data['personal_information']['id']] = 'done'
            print(self.GlobalRegister)
            msg = "Đã thêm thành công"
        if signal_status == 'remove':
            self.GlobalRegister[data['personal_information']['id']] = 'processing'
            add_insert_remove, embedded_face = self.registration.add_insert_remove_new_image_from_app(data['method'],
                                                                   list_image_url=[],
                                                                   personal_information=data['personal_information'])
            print(data['personal_information']['name'])
            for url in self.GlobalUrl:
                self.GlobalUrl[url]['modify_event'].set()
                print(f"{url}hihi")
            self.GlobalRegister[data['personal_information']['id']] = 'done'
            msg = "Đã xóa thành công", 
        return msg, embedded_face