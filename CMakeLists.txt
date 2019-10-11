cmake_minimum_required(VERSION 3.15)

project(toypt
	LANGUAGES									CXX CUDA
	VERSION										0.1.0)

# Find packages
find_package(OpenMP)
find_package(OpenCL)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenImageDenoise)
find_package(cxxutility REQUIRED)

# Qt MOC
set(CMAKE_AUTOMOC								ON)
set(CMAKE_AUTORCC								ON)
set(CMAKE_AUTOUIC								ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY				"${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY				"${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY				"${CMAKE_CURRENT_SOURCE_DIR}/bin")

# Compiler flags
set(CMAKE_CXX_STANDARD							17)
set(CMAKE_CXX_FLAGS								"-mavx2 -mfma")
set(CMAKE_CXX_FLAGS_RELEASE						"-O3 -flto -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG						"-g")

# Installation settings
set(CMAKE_SKIP_BUILD_RPATH						FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH				TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH			TRUE)
set(CMAKE_INSTALL_RPATH							${CMAKE_INSTALL_PREFIX}/lib/toypt)

# Definitions
add_definitions(-DKERNEL_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/kernels/kernel.cl")

# Targets
set(RENDERING_PUBLIC_HEADERS
	"include/rendering/abstractrenderer.h"
	"include/rendering/color.h"
	"include/rendering/cuda/cudaarray.h"
	"include/rendering/cuda/cudaarrayprivate.h"
	"include/rendering/cuda/cudarenderer.h"
	"include/rendering/cuda/cudatypes.h"
	"include/rendering/cuda/kernelwrapper.h"
	"include/rendering/framebuffer.h"
	"include/rendering/geometrycontainer.h"
	"include/rendering/material.h"
	"include/rendering/mesh.h"
	"include/rendering/pointlight.h"
	"include/rendering/randomnumbergenerator.h"
	"include/rendering/ray.h"
	"include/rendering/simd/simdrenderer.h"
	"include/rendering/storage.h"
	"include/rendering/triangle.h")

set(RENDERING_SOURCES
	"src/rendering/color.cpp"
	"src/rendering/cuda/cudaarrayprivate.cu"
	"src/rendering/cuda/cudarenderer.cpp"
	"src/rendering/cuda/kernel.cu"
	"src/rendering/cuda/kernelwrapper.cpp"
	"src/rendering/framebuffer.cpp"
	"src/rendering/geometrycontainer.cpp"
	"src/rendering/material.cpp"
	"src/rendering/mesh.cpp"
	"src/rendering/pointlight.cpp"
	"src/rendering/randomnumbergenerator.cpp"
	"src/rendering/ray.cpp"
	"src/rendering/simd/simdrenderer.cpp"
	"src/rendering/storage.cpp"
	"src/rendering/triangle.cpp")

# Rendering library
add_library(rendering SHARED					${RENDERING_SOURCES})

target_include_directories(rendering
	PUBLIC										$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
												$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
												$<BUILD_INTERFACE:/opt/cuda/include>)

set_target_properties(rendering PROPERTIES
	PRIVATE_HEADER								"${RENDERING_PRIVATE_HEADERS}")

target_link_libraries(rendering
	PUBLIC										cxxutility PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise)

# Application target
set(APP_HEADERS
	"include/application/application.h"
	"include/application/renderthread.h")

set(APP_SOUCRES
	"src/application/main.cpp"
	"src/application/application.cpp"
	"src/application/renderthread.cpp")

# Note: This seems not to work with Qt's MOC.
#add_executable(${PROJECT_NAME}					${APP_SOURCES}
#												${APP_HEADERS})

add_executable(${PROJECT_NAME}
	"include/application/application.h"
	"include/application/renderthread.h"
	"src/application/main.cpp"
	"src/application/application.cpp"
	"src/application/renderthread.cpp")

target_include_directories(${PROJECT_NAME}
	PUBLIC										$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
												$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
												$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/application>)

target_link_libraries(${PROJECT_NAME}
	PRIVATE										rendering Qt5::Widgets)

# Installation
install(FILES									${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/toypt
		PERMISSIONS								OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
		DESTINATION								${CMAKE_INSTALL_PREFIX}/bin/)

install(FILES									${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/librendering.so
		PERMISSIONS								OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
		DESTINATION								${CMAKE_INSTALL_PREFIX}/lib/toypt/)
