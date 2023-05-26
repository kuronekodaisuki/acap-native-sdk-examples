cmake_minimum_required(VERSION 3.8)

project(SnowDepth DESCRIPTION "Snow measurement test with OpenCV")

add_library(SnowDepth STATIC
	RotateAndCrop.cpp
	SnowMeasurement.cpp
	SnowDepth.cpp
	MovingAvg.cpp
)

#find_package(Eigen3 REQUIRED)

target_include_directories(SnowDepth PUBLIC
	${OPENCV_INCLUDE_DIR}
	PUBLIC ${PROJECT_SOURCE_DIR}
	#${Eigen3_INCLUDE_DIRS}
)

target_link_libraries(SnowDepth PUBLIC
	${OPENCV_LIBRARIES}
    opencv_aruco
    opencv_calib3d
    opencv_core
    opencv_features2d
    opencv_flann
    opencv_video
	#Eigen3::Eigen
)
