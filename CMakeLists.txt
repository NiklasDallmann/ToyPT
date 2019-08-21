cmake_minimum_required(VERSION 2.8)

project(pathtracer)

find_package(OpenMP)

set(CMAKE_CXX_FLAGS "-o3 -flto -march=native -mtune=native -mavx2 -std=c++17")
#set(CMAKE_CXX_FLAGS "-g -mavx2 -std=c++17")

#add_library(math STATIC )
#target_include_directories(math
#	PUBLIC
#		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
#)
#target_link_libraries(math OpenMP::OpenMP_CXX)

add_library(rendering STATIC
	"src/rendering/framebuffer.cpp"
	"src/rendering/color.cpp"
	"src/rendering/renderer.cpp"
	"src/rendering/triangle.cpp"
	"src/rendering/material.cpp"
	"src/rendering/mesh.cpp"
	"src/rendering/pointlight.cpp"
	"src/rendering/geometrycontainer.cpp"
	"src/rendering/ray.cpp"
	"src/rendering/simdtypes.cpp"
	"src/rendering/randomnumbergenerator.cpp")
target_include_directories(rendering
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
)
target_link_libraries(rendering OpenMP::OpenMP_CXX)

add_executable(${PROJECT_NAME} "src/application/main.cpp")
target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
)
target_link_libraries(${PROJECT_NAME} rendering)
