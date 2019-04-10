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
    embedding_list: List[Embedding]

# Load model
with open("model/model_zoo.json", 'rb') as json_file:
    MODEL_ZOO = json.load(json_file)
MODEL_TYPE = os.getenv("KERAS_PRETRAINED_MODEL", "MobileNetV2")
MODEL_INSTANCE = ImagePretrainedModel(MODEL_ZOO, MODEL_TYPE)

# Routes
@app.post("/get-embedding/", response_model=Embedding)
async def get_embedding(file: UploadFile = File(...)):
    # TODO ADD normalize
    emb = MODEL_INSTANCE.get_embbeding_from_bytes_cv2(file)
    return {"filename": file.filename, "embedding": emb}

@app.post("/get-embedding-from-list/", response_model=EmbeddingList)
async def get_embedding_from_list(request: Request):
    # TODO ADD normalize
    form = await request.form()
    myfilelist = form.getlist("files")
    embs = MODEL_INSTANCE.get_embbeding_from_bytes_list_cv2(myfilelist)
    reslist = [{"filename": file.filename, "embedding": embs[i]} for i, file in enumerate(myfilelist)]
    return {"embedding_list": reslist}
