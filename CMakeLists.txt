cmake_minimum_required(VERSION 2.8.8)
project(lua_wrapper)

set (CMAKE_CXX_FLAGS "-g -Wall -Wextra -fopenmp -fPIC -std=c++11 -O2")


# Torch dependencies
find_package(TH REQUIRED)
if (TH_FOUND)
    message(STATUS "TH found (${TH_INCLUDE_DIR})")
    message(STATUS "TH found (${TH_LIBRARIES})")
endif()

find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch found (${Torch_INSTALL_INCLUDE})")
    message(STATUS "Torch found (${Torch_INSTALL_LIB})")
endif()

include_directories("${TH_INCLUDE_DIR}")
include_directories("${Torch_INSTALL_INCLUDE}")
set(TH_LIB_DIR ${TH_LIBRARIES} ${LUA_LIBRARIES} ${LUAT_LIBRARIES})
set(TH_LIB_DEPS -lTH -lluajit -lluaT) # -ltorch-lua-static -lnn -ltorch -lnnx -limage -limgraph


# OpenCv Dependencies (TODO: Make it optional)
find_package(OpenCV REQUIRED)
set(OPENCV_VERSION ${OpenCV_VERSION_MAJOR})
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV ${OpenCV_VERSION_MAJOR} (${OpenCV_VERSION}) found (${OpenCV_CONFIG_PATH})")


include_directories(include/)

link_directories(${TH_LIB_DIR})

#### Compile the library ####

set(LIB_SOURCES
    # Do NOT include generics here (specially ,cpp), should be dynamically compiled when they
    # are included in lua WRAP
    include/lua_wrap_generic.hpp
    include/lua_wrap.hpp
    src/lua_wrap.cpp
)

set(LUAW_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/")
message(STATUS "Generating lua_wrap library in: ${LUAW_LIB_DIR}")

add_library(lua_wrap SHARED ${LIB_SOURCES})
set_target_properties(lua_wrap PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LUAW_LIB_DIR})
target_link_libraries (lua_wrap ${OpenCV_LIBS} ${TH_LIB_DEPS})

#### Compile the example program ####

set(MAIN_SOURCES
    main.cpp
)

link_directories(${LUAW_LIB_DIR})

add_executable (a.out ${MAIN_SOURCES})
target_link_libraries (a.out ${OpenCV_LIBS} ${TH_LIB_DEPS} -llua_wrap)
