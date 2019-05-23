cmake_minimum_required(VERSION 2.8)

project(pathtracer)

add_library(math STATIC "src/math/vector3d.cpp" "src/math/matrix3d.cpp")
target_include_directories(math PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>)

add_executable(${PROJECT_NAME} "main.cpp")
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>)
target_link_libraries(${PROJECT_NAME} math)
