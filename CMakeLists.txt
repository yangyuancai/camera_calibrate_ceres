cmake_minimum_required(VERSION 3.0)
#cmake版本需求

project(CeresMonoCali C CXX)
#工程名字

set(CMAKE_CXX_STANDARD 17)
#c++标准



option(DEBUG_MODE "flag for debuging output" ON)

set(gcc_like_compilers GNU Clang Intel)
#？？？

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
	include_directories(${CMAKE_SOURCE_DIR}/source)
endif (OpenCV_FOUND)


set(Eigen3_DIR "D:/ThridPartyLib/Eigen3/share/eigen3/cmake")
set(gflags_DIR "D:/ThridPartyLib/gflags/lib/cmake/gflags")
set(glog_DIR "D:/ThridPartyLib/glog/lib/cmake/glog")
set(Ceres_DIR "D:/ThridPartyLib/ceres/lib/cmake/Ceres")

find_package(Ceres REQUIRED)

message(${OpenCV_LIBS})

include_directories(include)
#向工程添加多个特定的头文件搜索路径
set(EXAMPLENAME fourinone_calibration_v2_example)
#设置项目名称
add_subdirectory(example)
#向当前工程添加存放源文件的子目录
