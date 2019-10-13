cmake_minimum_required(VERSION 3.15)

project(toypt
	LANGUAGES									CXX CUDA
	VERSION										0.1.0)

# Find packages
find_package(cxxutility REQUIRED)

# Project settings
set(TOYPT_PROJECT_DIR							"${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY				"${CMAKE_CURRENT_SOURCE_DIR}/lib")
# Note: This does not work with the default build RPATH. Put shared libraries next to executables until that is fixed.
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY				"${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY				"${CMAKE_CURRENT_SOURCE_DIR}/bin")
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

# Subdirectories
add_subdirectory("src/rendering")
add_subdirectory("src/application")

# Installation
install(FILES									${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/toypt
		PERMISSIONS								OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
		DESTINATION								${CMAKE_INSTALL_PREFIX}/bin/)

install(FILES									${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/librendering.so
		PERMISSIONS								OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
		DESTINATION								${CMAKE_INSTALL_PREFIX}/lib/toypt/)
