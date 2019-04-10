# Generic API

This a dockerized API to expose a scikit model with fastAPI.

Based on https://github.com/tiangolo/uvicorn-gunicorn-docker

# Structure 

### *API endpoints are in main but you may add your utils in 'app/app' folder*

/app/app/main.py 

### *Model zoo is in 'app/model' folder*

model_zoo.json lists all pretrained keras models that you can use.

Run with the "-e KERAS_PRETRAINED_MODEL=Xception" flag to run docker 
with the chosen model (here for Xception model).
 
Default is MobileNetV2.

# Run this :

## *Build image :*

cd generic_api

docker build -t keras_pretrained_api .

## *Run image :*

docker run -d -p 80:80 keras_pretrained_api

docker run -d -e "KERAS_PRETRAINED_MODEL=Xception" -p 80:80 keras_pretrained_api 

docker run -p 80:80 keras_pretrained_api

(*remove -d to keep CLI attached*)


### *Dev mode :*

docker run -d -p 80:80 -v $(pwd) keras_pretrained_api /start-reload.sh
 
docker run -d  -e "KERAS_PRETRAINED_MODEL=Xception" -p 80:80 -v $(pwd) keras_pretrained_api /start-reload.sh 

docker run -e "KERAS_PRETRAINED_MODEL=Xception" -p 80:80 -v $(pwd) keras_pretrained_api /start-reload.sh 

(*-v $(pwd) to use local folder as volume*)
