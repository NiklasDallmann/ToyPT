cmake_minimum_required(VERSION 2.8)

project(pathtracer)

add_library(math STATIC "src/math/vector3d.cpp" "src/math/matrix3d.cpp")
target_include_directories(math
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
)
target_compile_options(math PUBLIC "-o3 -flto")

add_library(rendering STATIC "src/rendering/framebuffer.cpp" "src/rendering/color.cpp" "src/rendering/renderer.cpp" "src/rendering/triangle.cpp" "src/rendering/material.cpp")
target_include_directories(math
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
)
target_link_libraries(rendering math)
target_compile_options(rendering PUBLIC "-o3 -flto")

add_executable(${PROJECT_NAME} "src/application/main.cpp")
target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/math>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rendering>
)
target_link_libraries(${PROJECT_NAME} math)
target_link_libraries(${PROJECT_NAME} rendering)
target_compile_options(${PROJECT_NAME} PUBLIC "-o3 -flto")
