cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(demo_coopcl_sift)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
if(NOT ${OpenCL_VERSION_MAJOR} MATCHES "2")
message(FATAL_ERROR "OpenCL 2.x not found")
endif(NOT ${OpenCL_VERSION_MAJOR} MATCHES "2")


# -----------------
# SIFT_lib
# -----------------
if (UNIX)
find_package(Threads REQUIRED)
endif (UNIX)

add_library(coopcl_Sift sift.cpp)
set_target_properties(coopcl_Sift PROPERTIES CXX_STANDARD 14)
target_include_directories(coopcl_Sift PRIVATE ${OpenCL_INCLUDE_DIRS})
target_include_directories(coopcl_Sift PRIVATE ../clDriver/lib/)
target_link_libraries(coopcl_Sift PRIVATE ${OpenCL_LIBRARIES})
if (UNIX)
#target_link_libraries(coopcl_Sift pthread)
endif (UNIX)


# -----------------
# DemoAPP
# -----------------
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
target_include_directories(${PROJECT_NAME} PRIVATE .)
target_include_directories(${PROJECT_NAME} PRIVATE ../common)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE coopcl_Sift)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4)#/WX)
else(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)#-Werror)
endif(MSVC)
