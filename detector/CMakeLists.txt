cmake_minimum_required(VERSION 3.22)
project(detect)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置构建类型为 Release
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")

find_package( OpenCV REQUIRED ) # 找到需求的库或者组件
find_package(OpenVINO REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ) # 添加一个用于包含的文件去构建的文件夹
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(aha
    Armor.cpp
    InferCore.cpp
    ../carman/carman.cpp
)
target_link_libraries(aha ${OpenCV_LIBS} openvino::runtime)

add_executable( detect 
    detector.cpp
)
target_link_libraries(detect
    ${OpenCV_LIBS}
    openvino::runtime
    aha
)