cmake_minimum_required(VERSION 3.18)
project(tab_completion LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enable CUDA backend acceleration" OFF)
option(ENABLE_METAL "Enable Apple Metal backend acceleration" OFF)

set(TAB_COMPLETION_SOURCES
    src/infilling.cpp
    src/sampler.cpp
    src/engine.cpp
    src/main.cpp
)

add_executable(tab_completion ${TAB_COMPLETION_SOURCES})

target_include_directories(tab_completion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Check if part of the parent ggmlc build or standalone
if (TARGET ggmlc_runtime)
    target_link_libraries(tab_completion PRIVATE
        ggmlc_runtime
    )
    if (WIN32)
        target_link_libraries(tab_completion PRIVATE ws2_32)
    endif()
    target_include_directories(tab_completion PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/include/ggml
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ggml/src
    )
else()
    # Standalone build configuration
    set(GGMLC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    
    target_include_directories(tab_completion PRIVATE
        ${GGMLC_ROOT}/runtime/include
        ${GGMLC_ROOT}/third_party/ggml/include/ggml
        ${GGMLC_ROOT}/third_party/ggml/src
    )

    if (ENABLE_CUDA)
        enable_language(CUDA)
        add_compile_definitions(GGML_USE_CUDA)
        set(CUDA_GRAPH_SOURCES
            ${GGMLC_ROOT}/runtime/src/cuda_graph.cu
            ${GGMLC_ROOT}/runtime/src/vmm_pool.cu
        )
        set(BACKEND_LIBS ggml ggml-base ggml-cpu ggml-cuda cuda)
    elseif (ENABLE_METAL)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        find_library(METAL_FRAMEWORK Metal REQUIRED)
        find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
        add_compile_definitions(GGML_USE_METAL)
        set(BACKEND_LIBS ggml ggml-base ggml-cpu ggml-metal ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})
        set(CUDA_GRAPH_SOURCES
            ${GGMLC_ROOT}/runtime/src/cuda_graph_stub.cpp
            ${GGMLC_ROOT}/runtime/src/vmm_pool_stub.cpp
        )
    else()
        set(CUDA_GRAPH_SOURCES
            ${GGMLC_ROOT}/runtime/src/cuda_graph_stub.cpp
            ${GGMLC_ROOT}/runtime/src/vmm_pool_stub.cpp
        )
        set(BACKEND_LIBS ggml_lib)
    endif()

    target_sources(tab_completion PRIVATE
        ${GGMLC_ROOT}/runtime/src/loader.cpp
        ${GGMLC_ROOT}/runtime/src/executor.cpp
        ${GGMLC_ROOT}/runtime/src/radix_tree.cpp
        ${GGMLC_ROOT}/runtime/src/batch_scheduler.cpp
        ${GGMLC_ROOT}/runtime/src/stdlib_kernels.cpp
        ${GGMLC_ROOT}/runtime/src/pipeline/tokenizer.cpp
        ${CUDA_GRAPH_SOURCES}
    )

    target_link_libraries(tab_completion PRIVATE ${BACKEND_LIBS})
endif()
