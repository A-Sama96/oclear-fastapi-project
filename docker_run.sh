#!/bin/bash

export JOB_NAME="OFA-container"
export IMAGE="asanjaata96/oclear-fastapi"
export TAG="latest"
export PYTHON_ENV="production"
export API_PORT=8080
export WORKERS=2
export TIMEOUT=300
export LOG_FOLDER=/home/asan-jaata/Desktop/oclear-log

echo ${IMAGE}:${TAG}

# Create log folder if not exists
if [ ! -d ${LOG_FOLDER} ]; then
     mkdir ${LOG_FOLDER}
fi

# Add your authentication command for the docker image registry here

# force pull and update the image, use this in remote host only
docker pull ${IMAGE}:${TAG}

# stop running container with same job name, if any
if [ "$(docker ps -a | grep $JOB_NAME)" ]; then
  docker stop ${JOB_NAME} && docker rm ${JOB_NAME}
fi

# start docker container
echo docker run -d \
  --gpus all \
  -p ${API_PORT}:80 \
  -e "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" \
  -e "WORKERS=${WORKERS}" \
  -e "TIMEOUT=${TIMEOUT}" \
  -e "PYTHON_ENV=${PYTHON_ENV}" \
  -v "${LOG_FOLDER}:/app/log" \
  --name="${JOB_NAME}" \
  ${IMAGE}:${TAG}
