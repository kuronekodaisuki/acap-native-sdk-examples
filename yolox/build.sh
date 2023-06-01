#!/bin/sh

export ARCH=armv7hf
export DOCKER_BUILDKIT=1
export APP_NAME=yolox:1.0

docker build --build-arg CHIP="edgetpu" --tag ${APP_NAME} .

docker cp $(docker create ${APP_NAME}):/opt/app ./build
