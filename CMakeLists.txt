cmake_minimum_required(VERSION 3.10)
project(ggmlc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/arch/x86
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/llamafile
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)

# GGML Library source files
set(GGML_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-alloc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-backend-reg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-backend-meta.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-quants.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-threading.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/ggml-cpu.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/ggml-cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/quants.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/traits.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/unary-ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/binary-ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/vec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/repack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/arch/x86/quants.c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/arch/x86/repack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/ggml-cpu/llamafile/sgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/src/gguf.cpp
)

add_library(ggml_lib STATIC ${GGML_SOURCES})
target_compile_definitions(ggml_lib PUBLIC
    _GNU_SOURCE
    GGML_USE_CPU
    GGML_VERSION="0.1.0"
    GGML_COMMIT="master"
)
find_package(Threads REQUIRED)
target_link_libraries(ggml_lib PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

add_subdirectory(runtime)
