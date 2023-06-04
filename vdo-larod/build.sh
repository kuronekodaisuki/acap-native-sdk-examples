#!/bin/sh

export ARCH=armv7hf
export CHIP=edgetpu
export APP_IMAGE=vdo-larod

docker build --tag $APP_IMAGE --build-arg CHIP=$CHIP .
docker cp $(docker create $APP_IMAGE):/opt/app ./build
