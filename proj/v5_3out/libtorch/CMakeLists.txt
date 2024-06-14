cmake_minimum_required(VERSION 3.16)

project(onte)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
set(CMAKE_CXX_FLAGS ${CAMKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(OpenCV_DIR /usr/local/bin)
set(Torch_DIR /home/lxai/tool/libtorchtool/libtorch/share/cmake/Torch)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PRIVATE} ${CUDA_INCLUDE_DIRS})

add_executable(onte main.cpp)

# target_link_libraries(onte ${TORCH_LIBRARIES})
target_link_libraries(onte ${OpenCV_LIBS} ${TORCH_LIBRARIES})
