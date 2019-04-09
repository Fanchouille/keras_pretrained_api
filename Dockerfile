FROM tiangolo/uvicorn-gunicorn:python3.6

LABEL maintainer="Fanch <francois.valadier@openvalue.fr>"

RUN pip install fastapi tensorflow numpy keras Pillow

RUN pip install python-multipart opencv-python

COPY ./app /app