cmake_minimum_required(VERSION 2.8)

project(pathtracer LANGUAGES CXX CUDA)

find_package(OpenMP)
find_package(OpenCL)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenImageDenoise)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-mavx2 -mfma")
set(CMAKE_CXX_FLAGS_RELEASE "-o3 -flto -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

add_definitions(-DKERNEL_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/kernels/kernel.cl")

add_library(utility SHARED
	"src/utility/debugstream.cpp"
	"include/utility/debugstream.h"
	"include/utility/globals.h")
target_include_directories(utility
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/utility>
)

add_library(rendering SHARED
	"src/rendering/framebuffer.cpp"
	"include/rendering/framebuffer.h"
	"src/rendering/color.cpp"
	"include/rendering/color.h"
	"src/rendering/triangle.cpp"
	"include/rendering/triangle.h"
	"src/rendering/material.cpp"
	"include/rendering/material.h"
	"src/rendering/mesh.cpp"
	"include/rendering/mesh.h"
	"src/rendering/pointlight.cpp"
	"include/rendering/pointlight.h"
	"src/rendering/geometrycontainer.cpp"
	"include/rendering/geometrycontainer.h"
	"src/rendering/ray.cpp"
	"include/rendering/ray.h"
	"src/rendering/storage.cpp"
	"include/rendering/storage.h"
	"src/rendering/randomnumbergenerator.cpp"
	"include/rendering/randomnumbergenerator.h"
	"include/rendering/abstractrenderer.h"
	"src/rendering/simd/simdrenderer.cpp"
	"include/rendering/simd/simdrenderer.h"
	"src/rendering/cuda/cudarenderer.cpp"
	"include/rendering/cuda/cudarenderer.h"
	"src/rendering/cuda/cudaarrayprivate.cu"
	"include/rendering/cuda/cudaarrayprivate.h"
	"include/rendering/cuda/cudaarray.h"
	"src/rendering/cuda/kernelwrapper.cpp"
	"include/rendering/cuda/kernelwrapper.h"
	"src/rendering/cuda/kernel.cu"
	"include/rendering/cuda/cudatypes.h")
target_include_directories(rendering
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
		$<BUILD_INTERFACE:/opt/cuda/include>
)
target_link_libraries(rendering utility OpenMP::OpenMP_CXX OpenImageDenoise)

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
