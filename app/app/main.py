from fastapi import FastAPI, File, UploadFile
from .utils import ImagePretrainedModel
import json
import os
from pydantic import BaseModel
from typing import List
from starlette.responses import HTMLResponse

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
async def get_embedding(file: UploadFile = File(...), normalize: bool = None):
    if normalize is None:
        normalize = True
    emb = MODEL_INSTANCE.get_embbeding_from_bytes_cv2(file, normalize)
    return {"filename": file.filename, "embedding": emb}

@app.post("/get-embedding-from-list/", response_model=EmbeddingList)
async def get_embedding_from_list(files: List[UploadFile] = File(...), normalize: bool = None):
    if normalize is None:
        normalize = True
    embs = MODEL_INSTANCE.get_embbeding_from_bytes_list_cv2(files, normalize)
    reslist = [{"filename": file.filename, "embedding": embs[i]} for i, file in enumerate(files)]
    return {"embedding_list": reslist}

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}

@app.get("/")
async def main():
    content = """
<body>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)