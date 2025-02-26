ARG ARCH=armv7hf
ARG REPO=axisecp
ARG SDK=acap-native-sdk
ARG UBUNTU_VERSION=22.04
ARG VERSION=1.8

FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION}

# Set general arguments
ARG ARCH
ARG SDK_LIB_PATH_BASE=/opt/axis/acapsdk/sysroots/${ARCH}/usr
ARG BUILD_DIR=/opt/build

#-------------------------------------------------------------------------------
# Prepare build environment
#-------------------------------------------------------------------------------

# Install build dependencies for cross compiling OpenCV
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y -f --no-install-recommends \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#-------------------------------------------------------------------------------
# Build OpenCV libraries
#-------------------------------------------------------------------------------

ARG OPENCV_VERSION=4.5.1
ARG OPENCV_DIR=${BUILD_DIR}/opencv
ARG OPENCV_SRC_DIR=${OPENCV_DIR}/opencv-${OPENCV_VERSION}
ARG OPENCV_BUILD_DIR=${OPENCV_DIR}/build

#
# OpenCV aruco module
#
COPY aruco ${OPENCV_SRC_DIR}/modules/aruco

WORKDIR ${OPENCV_DIR}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz | tar -xz

COPY opencv/cap_axis_vdo.cpp modules/videoio/src
COPY opencv/detect_axis_vdo.cmake modules/videoio/cmake
COPY opencv/axisvdo-4.5.1.patch .
#RUN patch -p 1 < axisvdo-4.5.1.patch

WORKDIR ${OPENCV_BUILD_DIR}
ENV COMMON_CMAKE_FLAGS="-S $OPENCV_SRC_DIR \
        -B $OPENCV_BUILD_DIR \
        -D CMAKE_INSTALL_PREFIX=$SDK_LIB_PATH_BASE \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D WITH_OPENEXR=OFF \
        -D WITH_GTK=OFF \
        -D WITH_V4L=OFF \
        -D WITH_FFMPEG=OFF \
        -D WITH_GSTREAMER=OFF \
        -D WITH_GSTREAMER_0_10=OFF \
        -D BUILD_LIST=aruco,calib3d,core,features2d,flann,imgproc,video \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_OPENCV_APPS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_JPEG=ON \
        -D BUILD_PNG=OFF \
        -D WITH_JASPER=OFF \
        -D BUILD_PROTOBUF=OFF \
        -D OPENCV_GENERATE_PKGCONFIG=ON "

# hadolint ignore=SC2086
RUN if [ "$ARCH" = armv7hf ]; then \
        # Source SDK environment to get cross compilation tools
        . /opt/axis/acapsdk/environment-setup* && \
        # Configure build with CMake
        cmake \
        -D CMAKE_CXX_COMPILER=${CXX%-g++*}-g++ \
        -D CMAKE_CXX_FLAGS="${CXX#*-g++}" \
        -D CMAKE_C_COMPILER=${CC%-gcc*}-gcc \
        -D CMAKE_C_FLAGS="${CC#*-gcc}" \
        -D CMAKE_TOOLCHAIN_FILE=${OPENCV_SRC_DIR}/platforms/linux/arm-gnueabi.toolchain.cmake \
        -D CPU_BASELINE=NEON,VFPV3 \
        -D ENABLE_NEON=ON \
        -D ENABLE_VFPV3=ON \
        $COMMON_CMAKE_FLAGS && \
        # Build and install OpenCV
        make -j "$(nproc)" install ; \
    elif [ "$ARCH" = aarch64 ]; then \
        # Source SDK environment to get cross compilation tools
        . /opt/axis/acapsdk/environment-setup* && \
        # Configure build with CMake
        # No need to set NEON and VFP for aarch64 since they are implicitly
        # present in an any standard armv8-a implementation.
        cmake \
        -D CMAKE_CXX_COMPILER=${CXX%-g++*}-g++ \
        -D CMAKE_CXX_FLAGS="${CXX#*-g++}" \
        -D CMAKE_C_COMPILER=${CC%-gcc*}-gcc \
        -D CMAKE_C_FLAGS="${CC#*-gcc}" \
        -D CMAKE_TOOLCHAIN_FILE=${OPENCV_SRC_DIR}/platforms/linux/aarch64-gnu.toolchain.cmake \
        $COMMON_CMAKE_FLAGS && \
        # Build and install OpenCV
        make -j "$(nproc)" install ; \
    else \
        printf "Error: '%s' is not a valid value for the ARCH variable\n", "$ARCH"; \
        exit 1; \
    fi

#-------------------------------------------------------------------------------
# Copy the built library files to application directory
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .
RUN mkdir lib && \
    cp -P ${OPENCV_BUILD_DIR}/lib/lib*.so* ./lib/

#-------------------------------------------------------------------------------
# Finally build the ACAP application
#-------------------------------------------------------------------------------
WORKDIR /opt/app/SnowMeasurement
#
# build SnowMeasurement library
#
RUN . /opt/axis/acapsdk/environment-setup* && \
    cmake \
        -D CMAKE_CXX_COMPILER=${CXX%-g++*}-g++ \
        -D CMAKE_CXX_FLAGS="${CXX#*-g++}" \
        -D CMAKE_C_COMPILER=${CC%-gcc*}-gcc \
        -D CMAKE_C_FLAGS="${CC#*-gcc}" \
        -D OPENCV_INCLUDE_DIR=$SDK_LIB_PATH_BASE/include/opencv4 \
        -D OPENCV_LIBRARIES=$SDK_LIB_PATH_BASE/lib \
        .

#
# Build Library
#
RUN make -j 8;
RUN cp lib*.* ../lib/

WORKDIR /opt/app

# Get pretrained models
ARG CHIP=
RUN if [ "$CHIP" = cpu ] || [ "$CHIP" = artpec8 ]; then \
        curl -L -o converted_model.tflite \
            https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite ; \
    elif [ "$CHIP" = edgetpu ]; then \
        curl -L -o converted_model.tflite \
            https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite ; \
    else \
        printf "Error: '%s' is not a valid value for the CHIP variable\n", "$CHIP"; \
        exit 1; \
    fi

# Download the labels
RUN mkdir -p label && \
    curl -L -o label/labels.txt https://github.com/google-coral/test_data/raw/master/coco_labels.txt

# Setup the model directory
RUN mkdir -p model && \
    cp converted_model.tflite model/converted_model.tflite

#-------------------------------------------------------------------------------
# Finally build the ACAP application
#-------------------------------------------------------------------------------
WORKDIR /opt/app

RUN . /opt/axis/acapsdk/environment-setup* && acap-build .\
    -a 'config/AXIS_Q6215-LE.xml' -a 'label/coco_classes.txt' -a 'model/yolox_nano_int8_quantize.tflite'
