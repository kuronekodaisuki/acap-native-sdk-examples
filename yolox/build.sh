#!/bin/sh

export ARCH=armv7hf
export DOCKER_BUILDKIT=1
export APP_NAME=snow-depth:1.0

docker build --tag ${APP_NAME} .

docker cp $(docker create ${APP_NAME}):/opt/app ./build
