cmake_minimum_required(VERSION 3.16)

project(onte)

set(CMAKE_CXX_STANDARD 17)

# set(OpenCV_DIR /usr/local/bin)
set(OpenCV_DIR /home/lxai/.conda/envs/OpcvEnv/share/opencv4)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(onte main.cpp)

target_link_libraries(onte ${OpenCV_LIBS})
