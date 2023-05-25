cmake_minimum_required(VERSION 3.8)

project(aruco DESCRIPTION "OpenCV aruco module")

add_library(aruco STATIC
	src/aruco.cpp
	src/apriltag_quad_thresh.cpp
	src/charuco.cpp
	src/dictionary.cpp
	src/zmaxheap.cpp
	)

find_package(OpenCV REQUIRED)

target_include_directories(aruco PUBLIC
	${OpenCV_INCLUDE_DIRS}
	)

target_link_libraries(aruco PUBLIC
	${OpenCV_LIBRARIES}
	)
