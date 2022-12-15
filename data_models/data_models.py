from typing import List, Any, Optional
from pydantic import BaseModel


class PersonalInformation(BaseModel):
    id: str
    name: str
    code: Optional[str]
    branchID: Optional[str]
    departmentId: Optional[str]


class InsertUpdateRequest(BaseModel):
    list_url_img: List[str]
    personal_information: PersonalInformation


class DeleteRequest(BaseModel):
    personal_information: PersonalInformation


class Response(BaseModel):
    result: Any

    def __init__(self, result: Any = None):
        super(Response, self).__init__(result=result)
        self.result = result
