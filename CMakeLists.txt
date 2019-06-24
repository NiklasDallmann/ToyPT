cmake_minimum_required(VERSION 2.8)

project(pathtracer)

find_package(OpenMP)

set(CMAKE_CXX_FLAGS "-o3 -flto -march=native -mtune=native -s")

add_library(math STATIC "src/math/vector4.cpp" "src/math/matrix4x4.cpp" "src/math/algorithms.cpp")
target_include_directories(math
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
)
target_link_libraries(math OpenMP::OpenMP_CXX)
#target_compile_options(math PUBLIC "-o3" "-flto" "-s")

add_library(rendering STATIC
	"src/rendering/framebuffer.cpp"
	"src/rendering/color.cpp"
	"src/rendering/renderer.cpp"
	"src/rendering/triangle.cpp"
	"src/rendering/material.cpp"
	"src/rendering/abstractmesh.cpp"
	"src/rendering/cube.cpp"
	"src/rendering/pointlight.cpp"
	"src/rendering/square")
target_include_directories(math
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
)
target_link_libraries(rendering math OpenMP::OpenMP_CXX)
#target_compile_options(rendering PUBLIC "-o3" "-flto" "-s")

add_executable(${PROJECT_NAME} "src/application/main.cpp")
target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
)
target_link_libraries(${PROJECT_NAME} math)
target_link_libraries(${PROJECT_NAME} rendering)
#target_compile_options(${PROJECT_NAME} PUBLIC "-o3" "-flto" "-s")
