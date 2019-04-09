from fastapi import FastAPI, File, UploadFile
from .utils import ImagePretrainedModel
import json
import os
from pydantic import BaseModel
from typing import List
from starlette.requests import Request

app = FastAPI()

class Embedding(BaseModel):
    filename: str
    embedding: List[float]

class EmbeddingList(BaseModel):
    embeddings: List[Embedding]

global MODEL_ZOO, MODEL_TYPE, MODEL_INSTANCE

with open("model/model_zoo.json", 'rb') as json_file:
    MODEL_ZOO = json.load(json_file)
MODEL_TYPE = os.getenv("KERAS_PRETRAINED_MODEL", "MobileNetV2")
MODEL_INSTANCE = ImagePretrainedModel(MODEL_ZOO, MODEL_TYPE)


@app.post("/get-embedding/", response_model=Embedding)
async def get_embedding(myfile: UploadFile = File(...)):
    # TODO ADD normalize
    #emb = MODEL_INSTANCE.get_embbeding_from_bytes(myfile)
    emb = MODEL_INSTANCE.get_embbeding_from_bytes_cv2(myfile)
    return {"filename": myfile.filename, "embedding": emb}

@app.post("/get-embedding-from-list/")
async def get_embedding_from_list(request: Request):
    # TODO ADD normalize
    form = await request.form()
    myfilelist = form.getlist("files")
    #embs = MODEL_INSTANCE.get_embbeding_from_bytes_list(myfilelist)
    embs = MODEL_INSTANCE.get_embbeding_from_bytes_list_cv2(myfilelist)
    return {"filenames": [myfile.filename for myfile in myfilelist], "embeddings": embs}
