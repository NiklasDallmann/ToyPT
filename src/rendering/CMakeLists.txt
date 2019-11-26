find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenImageDenoise REQUIRED)

set(RENDERING_PUBLIC_HEADERS
	"${TOYPT_PROJECT_DIR}/include/rendering/abstractrenderer.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/color.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/cuda/cudaarray.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/cuda/cudaarrayprivate.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/cuda/cudarenderer.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/cuda/cudatypes.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/framebuffer.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/geometrycontainer.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/material.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/mesh.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/pointlight.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/randomnumbergenerator.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/ray.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/simd/simdrenderer.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/storage.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/triangle.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/simd/kdtree.h"
	"${TOYPT_PROJECT_DIR}/include/rendering/shader.h")

set(RENDERING_SOURCES
	"${TOYPT_PROJECT_DIR}/src/rendering/color.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/cuda/cudaarrayprivate.cu"
	"${TOYPT_PROJECT_DIR}/src/rendering/cuda/cudarenderer.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/cuda/cudarendererprivate.cu"
	"${TOYPT_PROJECT_DIR}/src/rendering/framebuffer.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/geometrycontainer.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/material.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/mesh.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/pointlight.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/randomnumbergenerator.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/ray.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/simd/simdrenderer.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/storage.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/triangle.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/simd/kdtree.cpp"
	"${TOYPT_PROJECT_DIR}/src/rendering/shader.cpp")

# Rendering library
add_library(rendering SHARED					${RENDERING_SOURCES})

target_include_directories(rendering
	PUBLIC										$<BUILD_INTERFACE:${TOYPT_PROJECT_DIR}/include>
												$<BUILD_INTERFACE:${TOYPT_PROJECT_DIR}/include/rendering>
												$<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

set_target_properties(rendering PROPERTIES
	PRIVATE_HEADER								"${RENDERING_PRIVATE_HEADERS}")

target_link_libraries(rendering
	PUBLIC										cxxutility PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise) 
