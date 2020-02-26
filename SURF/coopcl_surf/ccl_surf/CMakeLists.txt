cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(ccl_Surf)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} ccl_surf.cpp)

#-------------
# surf_lib
#-------------
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
target_include_directories(${PROJECT_NAME} PRIVATE ../../../clDriver/lib/)

#open_cl
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})

#open_cv
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

if(WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE wsock32 ws2_32)
endif()