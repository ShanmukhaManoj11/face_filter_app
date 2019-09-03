cmake_minimum_required(VERSION 2.8.12)

project(face_detection)

find_package(OpenCV REQUIRED)

include_directories(include
	${OpenCV_INCLUDE_DIRS})

file(GLOB PROJECT_SRCS src/*.cpp)

add_executable(faceDetector ${PROJECT_SRCS})
target_link_libraries(faceDetector ${OpenCV_LIBS})