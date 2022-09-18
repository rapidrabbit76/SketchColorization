FROM python:3.8.12-slim-buster
WORKDIR /app

RUN apt-get update && apt install -y \ 
    libgl1-mesa-glx \
    curl  \
    libglib2.0-0 && apt-get clean 

RUN pip install --no-cache-dir \
    fastapi[all]==0.80.0 \
    service-streamer==0.1.2 \
    fastapi-restful==0.4.3 \
    fastapi-health==0.4.0 \
    loguru==0.6.0 


ADD ./requirements.txt /app
RUN pip install -r /app/requirements.txt

ADD . /app
RUN chmod +x /app/docker/start.sh
ENTRYPOINT /app/docker/start.sh
