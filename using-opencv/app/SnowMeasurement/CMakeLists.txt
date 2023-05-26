cmake_minimum_required(VERSION 3.8)

project(SnowMeasurement DESCRIPTION "Snow measurement test with OpenCV")

add_library(SnowMeasurement STATIC
	RotateAndCrop.cpp
	SnowMeasurement.cpp
	SnowDepth.cpp
	MovingAvg.cpp
	)

#find_package(OpenCV REQUIRED)
#find_package(Eigen3 REQUIRED)

target_include_directories(SnowMeasurement PUBLIC
	${OPENCV_INCLUDE_DIR}
	#${Eigen3_INCLUDE_DIRS}
	PUBLIC ${PROJECT_SOURCE_DIR}
	)

target_link_libraries(SnowMeasurement PUBLIC
	${OPENCV_LIBRARIES}
    opencv_aruco
    opencv_calib3d
    opencv_core
    opencv_features2d
    opencv_flann
    opencv_video
	#Eigen3::Eigen
	)
