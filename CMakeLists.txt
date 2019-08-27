cmake_minimum_required(VERSION 2.8)

project(pathtracer)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(OpenMP)
find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(CMAKE_CXX_FLAGS "-mavx2 -mfma -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-o3 -flto -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

add_library(rendering STATIC
	"src/rendering/framebuffer.cpp"
	"src/rendering/color.cpp"
	"src/rendering/simdrenderer.cpp"
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

add_executable(${PROJECT_NAME}
	"src/application/main.cpp"
	"src/application/application.cpp"
	"include/application/application.h"
	"src/application/renderthread.cpp"
	"include/application/renderthread.h")
target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/application>
)
target_link_libraries(${PROJECT_NAME} rendering Qt5::Widgets)
