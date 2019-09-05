cmake_minimum_required(VERSION 2.8.12)

project(face_detection)

find_package(OpenCV REQUIRED)

include_directories(include
	${OpenCV_INCLUDE_DIRS})

set(FD_SOURCES src/main.cpp src/faceDetector.cpp)
add_executable(faceDetector ${FD_SOURCES})
target_link_libraries(faceDetector ${OpenCV_LIBS})

set(3D_SOURCES src/3dLoader.cpp)
add_executable(3dLoader ${3D_SOURCES})
target_link_libraries(3dLoader ${OpenCV_LIBS})