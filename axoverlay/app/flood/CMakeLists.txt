cmake_minimum_required(VERSION 3.8)

project(FloodDepth DESCRIPTION "Snow measurement test with OpenCV")

add_library(FloodDepth SHARED
	Flood.cpp
    Marker.cpp
)

target_include_directories(FloodDepth PUBLIC
	${OPENCV_INCLUDE_DIR}
	PUBLIC ${PROJECT_SOURCE_DIR}
)

target_link_libraries(FloodDepth PUBLIC
	${OPENCV_LIBRARIES}
    opencv_aruco
    opencv_calib3d
    opencv_core
    opencv_features2d
    opencv_flann
    opencv_video
)
