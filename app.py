from fastapi import FastAPI
import uvicorn

from config.config import Config
from data_models.data_models import InsertUpdateRequest, DeleteRequest
from process.processor import Processor

app = FastAPI()
_config = Config()
processor = Processor(config=_config)


@app.post("/add")
async def process_add(input_data: InsertUpdateRequest):
    response = await processor.process_add(input_data)
    return response


@app.post("/update")
async def process_update(input_data: InsertUpdateRequest):
    response = await processor.process_update(input_data)
    return response


@app.post("/delete")
async def process_add(input_data: DeleteRequest):
    response = await processor.process_delete(input_data)
    return response


if __name__ == "__main__":
    # pipeline.start()
    # pipeline.join()
    uvicorn.run("app:app", host='localhost', port=35515, reload=True, debug=True, workers=1)
