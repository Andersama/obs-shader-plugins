project(obs-shader-filter)

set(CMAKE_PREFIX_PATH "${QTDIR}")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

#find_package(LibObs REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)

include_directories(
#	SYSTEM "${CMAKE_SOURCE_DIR}/libobs"
#	"${CMAKE_SOURCE_DIR}/UI/obs-frontend-api"
	${Qt5Core_INCLUDES}
	${Qt5Widgets_INCLUDES}
)

find_package(FFmpeg REQUIRED
	COMPONENTS avcodec)
include_directories(${FFMPEG_INCLUDE_DIRS})

if(MSVC)
	set(obs-shader-filter_PLATFORM_DEPS
		w32-pthreads)
endif()

set(obs-shader-filter_HEADERS
	fft.h
	tinyexpr.h
	mtrandom.h
	obs-shader-filter.hpp
)

set(obs-shader-filter_SOURCES
	fft.c
	obs-shader-filter.cpp
	tinyexpr.c
	mtrandom.cpp
)

add_library(obs-shader-filter MODULE
	${obs-shader-filter_SOURCES}
	${obs-shader-filter_HEADERS}
)
	
target_link_libraries(obs-shader-filter
	libobs
	Qt5::Core
	Qt5::Widgets
	${obs-shader-filter_PLATFORM_DEPS}
	${FFMPEG_LIBRARIES}
#	obs-frontend-api
#	Qt5::Core
#	Qt5::Widgets
)

install_obs_plugin_with_data(obs-shader-filter data)
