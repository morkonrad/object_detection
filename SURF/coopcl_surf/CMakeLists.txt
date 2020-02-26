cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(demo_coopcl_Surf)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(ccl_surf)

#-------------
# executable
#-------------
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
target_include_directories(${PROJECT_NAME} PRIVATE ../../common)
target_include_directories(${PROJECT_NAME} PRIVATE ../../clDriver/lib)

#ccl_surf.lib
target_include_directories(${PROJECT_NAME} PRIVATE ccl_surf)
target_link_libraries(${PROJECT_NAME} PRIVATE ccl_Surf)

#open_cv
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

#open_cl
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})