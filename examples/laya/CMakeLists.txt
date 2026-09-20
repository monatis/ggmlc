cmake_minimum_required(VERSION 3.18)
project(laya LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LAYA_SOURCES
    src/questions.cpp
    src/sequence.cpp
    src/presets.cpp
    src/language.cpp
    src/router.cpp
    src/engine.cpp
    src/server.cpp
    src/main.cpp
)

add_executable(laya ${LAYA_SOURCES})

target_include_directories(laya PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/src
)

target_link_libraries(laya PRIVATE ggmlc_runtime)
if (WIN32)
    target_link_libraries(laya PRIVATE ws2_32)
endif()
