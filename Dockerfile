FROM tiangolo/uvicorn-gunicorn:python3.6

LABEL maintainer="Fanch <francois.valadier@openvalue.fr>"

RUN pip install fastapi python-multipart tensorflow numpy keras Pillow opencv-python

ARG MODEL_NAME=MobileNetV2
ENV KERAS_PRETRAINED_MODEL=$MODEL_NAME

COPY ./app /app