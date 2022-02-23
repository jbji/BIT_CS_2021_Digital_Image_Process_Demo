cmake_minimum_required(VERSION 3.21)
project(2022_opencv_seamCarving)

set(CMAKE_CXX_STANDARD 14)

add_executable(2022_opencv_seamCarving main.cpp SeamCarvingResize.cpp SeamCarvingResize.h SeamCarvingEnergyFunctions.h SeamCarvingEnergyFunctions.cpp ImprovedSeamCarvingResize.cpp ImprovedSeamCarvingResize.h)

# For OpenCV
# 设置OpenCV DIR
set(OpenCV_DIR /usr/local/opt/opencv)
# 寻找OpenCV.CMakeLists,以此找到包,并赋值各库相关变量
find_package(OpenCV REQUIRED)
# OpenCV_INCLUDE_DIRS是关于find_package的变量,
# 包含了一个路径，这样可以在代码中的#include做根目录
include_directories( /usr/local/Cellar/opencv/4.5.4_2/include/opencv4 )
# 链接OpenCV库,OpenCV_LIBS为代表库可执行文件的变量
target_link_libraries( 2022_opencv_seamCarving ${OpenCV_LIBS} )