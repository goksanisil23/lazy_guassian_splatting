cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gsplat)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(COMPILER_FLAGS -Wall -std=c++17 -g -O2)

set(SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(gsplat ${SRCS})

target_include_directories(gsplat SYSTEM PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(gsplat ${TORCH_LIBRARIES} ${OpenCV_LIBS} pthread dl rt TBB::tbb)

target_include_directories(gsplat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gsplat PUBLIC ${COMPILER_FLAGS})

# Rendering
set(RAYLIB_CPP_INCLUDE_DIR "/home/s0001734/Downloads/raylib-cpp/include")
set(RAYLIB_INCLUDE_DIR "/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib/include")
set(RAYLIB_LINK_DIR "/home/s0001734/Downloads/raylib-cpp/build/_deps/raylib-build/raylib")

add_executable(render render.cpp)

target_include_directories(render SYSTEM PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(render PRIVATE ${RAYLIB_LINK_DIR})

target_link_libraries(render ${TORCH_LIBRARIES} ${OpenCV_LIBS} pthread dl rt TBB::tbb raylib)

target_include_directories(render PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RAYLIB_CPP_INCLUDE_DIR}
    ${RAYLIB_INCLUDE_DIR}

)

target_compile_options(render PUBLIC ${COMPILER_FLAGS})