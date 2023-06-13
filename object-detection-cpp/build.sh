#!/bin/sh

DOCKER_BUILDKIT=1 docker build --build-arg CHIP=edgetpu --tag object-detection-cpp .
