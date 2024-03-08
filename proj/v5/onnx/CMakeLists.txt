cmake_minimum_required(VERSION 3.16)
project(onte)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /usr/local/lib)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_INCLUDE /home/lxai/tool/onnxtool/onnxruntime-linux-x64-gpu-1.10.0/include)
set(ONNXRUNTIME_LIB /home/lxai/tool/onnxtool/onnxruntime-linux-x64-gpu-1.10.0/lib/libonnxruntime.so)

include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE})
add_executable(onte main.cpp)

target_link_libraries(onte ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
