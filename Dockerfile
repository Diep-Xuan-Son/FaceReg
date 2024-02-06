FROM python:3.8
FROM nvidia/cuda:12.0.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Vietnam
ENV FLASK_APP=modeldbs

RUN apt-get update && \
    apt-get install -y \
        # git \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
        libsm6 \
        libxext6
RUN python3 -m pip install --upgrade pip
# Set the working directory
WORKDIR /FaceReg
# COPY ./requirement .
RUN ls -la

# ADD requirements.txt requirements.txt
COPY ./requirements.txt .
RUN pip3 --no-cache-dir install -r requirements.txt
COPY ./requirements_service.txt .
RUN pip3 --no-cache-dir install -r requirements_service.txt

RUN python3 -m pip install -U numpy==1.23.4

ADD ./src/. /FaceReg

RUN ls -la

# ENV DATABASE_URL=postgresql+psycopg2://vms_op:MQ123456@face_db_op:543/vms_hcm

# RUN flask run
# RUN flask db init

# CMD ["python3", "controllers_face.py"]


# docker build -t mqbot-gradio-web-api --load --rm .
# docker run -d -p 8888:8888 --name mqbot mqbot-gradio-web-api
