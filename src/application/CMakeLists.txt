find_package(Qt5 COMPONENTS Widgets REQUIRED)

# Qt MOC
set(CMAKE_AUTOMOC								ON)
set(CMAKE_AUTORCC								ON)
set(CMAKE_AUTOUIC								ON)

set(APP_HEADERS
	"${TOYPT_PROJECT_DIR}/include/application/application.h"
	"${TOYPT_PROJECT_DIR}/include/application/renderthread.h")

set(APP_SOUCRES
	"${TOYPT_PROJECT_DIR}/src/application/main.cpp"
	"${TOYPT_PROJECT_DIR}/src/application/application.cpp"
	"${TOYPT_PROJECT_DIR}/src/application/renderthread.cpp")

# Note: This seems not to work with Qt's MOC.
#add_executable(${PROJECT_NAME}					${APP_SOURCES}
#												${APP_HEADERS})

add_executable(${PROJECT_NAME}
	"${TOYPT_PROJECT_DIR}/include/application/application.h"
	"${TOYPT_PROJECT_DIR}/include/application/renderthread.h"
	"${TOYPT_PROJECT_DIR}/src/application/main.cpp"
	"${TOYPT_PROJECT_DIR}/src/application/application.cpp"
	"${TOYPT_PROJECT_DIR}/src/application/renderthread.cpp")

target_include_directories(${PROJECT_NAME}
	PUBLIC										$<BUILD_INTERFACE:${TOYPT_PROJECT_DIR}/include/math>
												$<BUILD_INTERFACE:${TOYPT_PROJECT_DIR}/include/rendering>
												$<BUILD_INTERFACE:${TOYPT_PROJECT_DIR}/include/application>)

target_link_libraries(${PROJECT_NAME}
	PRIVATE										rendering Qt5::Widgets) 
