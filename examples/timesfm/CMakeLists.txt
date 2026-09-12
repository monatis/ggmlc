cmake_minimum_required(VERSION 3.18)
project(timesfm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TIMESFM_SOURCES
    src/revin.cpp
    src/detrending.cpp
    src/patching.cpp
    src/data_loader.cpp
    src/export.cpp
    src/visualizer.cpp
    src/backtest.cpp
    src/doctor.cpp
    src/server.cpp
    src/forecaster.cpp
    src/main.cpp
)

add_executable(timesfm ${TIMESFM_SOURCES})

target_include_directories(timesfm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/src
)

target_link_libraries(timesfm PRIVATE ggmlc_runtime)
if (WIN32)
    target_link_libraries(timesfm PRIVATE ws2_32)
endif()

add_executable(timesfm_bench
    src/revin.cpp
    src/detrending.cpp
    src/patching.cpp
    src/data_loader.cpp
    src/export.cpp
    src/visualizer.cpp
    src/backtest.cpp
    src/forecaster.cpp
    src/benchmark_batch.cpp
)

target_include_directories(timesfm_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/src
)

target_link_libraries(timesfm_bench PRIVATE ggmlc_runtime)
if (WIN32)
    target_link_libraries(timesfm_bench PRIVATE ws2_32)
endif()

