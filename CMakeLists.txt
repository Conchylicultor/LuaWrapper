cmake_minimum_required(VERSION 2.8.8)
project(lua_wrapper)

set (CMAKE_CXX_FLAGS "-g -Wall -Wextra -fopenmp -fPIC -std=c++11 -O2")

find_package(Torch REQUIRED)
if (TH_FOUND)
    message(STATUS "TH found (${TH_INCLUDE_DIR})")
    message(STATUS "TH found (${TH_LIBRARIES})")
endif()

find_package(TH REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch found (${Torch_INSTALL_INCLUDE})")
    message(STATUS "Torch found (${Torch_INSTALL_LIB})")
endif()

include_directories("${TH_INCLUDE_DIR}")
include_directories("${Torch_INSTALL_INCLUDE}")
set(TH_LIB_DIR ${TH_LIBRARIES} ${LUA_LIBRARIES} ${LUAT_LIBRARIES})
set(TH_LIB_DEPS -lTH -lluajit -lluaT) # -ltorch-lua-static -lnn -ltorch -lnnx -limage -limgraph


set(MAIN_SOURCES
    lua_wrap.cpp
    lua_wrap.hpp
    main.cpp
)

link_directories(${TH_LIB_DIR})

add_executable (a.out ${MAIN_SOURCES})
target_link_libraries (a.out ${TH_LIB_DEPS})
