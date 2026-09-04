if (GGMLC_ENABLE_CUDA)
    set(GGML_BACKEND_LIBS ggml ggml-base ggml-cpu ggml-cuda)
    add_compile_definitions(GGML_USE_CUDA)
else()
    set(GGML_BACKEND_LIBS ggml_lib)
endif()

add_library(ggmlc_runtime STATIC
    src/loader.cpp
    src/executor.cpp
    src/stdlib_kernels.cpp
    src/pipeline/image.cpp
    src/pipeline/tokenizer.cpp
)

target_link_libraries(ggmlc_runtime PUBLIC
    ${GGML_BACKEND_LIBS}
)

# CLI Tool
add_executable(ggmlc-run
    tools/ggmlc_run.cpp
)

target_link_libraries(ggmlc-run PRIVATE
    ggmlc_runtime
)

# Shared library for C-FFI
add_library(ggmlc_runtime_shared SHARED
    src/loader.cpp
    src/executor.cpp
    src/stdlib_kernels.cpp
    src/pipeline/image.cpp
    src/pipeline/tokenizer.cpp
)

target_link_libraries(ggmlc_runtime_shared PUBLIC
    ${GGML_BACKEND_LIBS}
)

# Optional Python Bindings with nanobind (targeting Python Limited API / abi3)
find_package(Python 3.11 COMPONENTS Interpreter Development.Module QUIET)
if (Python_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (EXISTS "${NANOBIND_CMAKE_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
    endif()

    find_package(nanobind CONFIG QUIET)
    if (nanobind_FOUND)
        nanobind_add_module(_runtime
            bindings/python_bindings.cpp
            src/loader.cpp
            src/executor.cpp
            src/stdlib_kernels.cpp
            src/pipeline/image.cpp
            src/pipeline/tokenizer.cpp
        )
        target_link_libraries(_runtime PRIVATE ${GGML_BACKEND_LIBS})
        
        # Set output directory to python/ggmlc for direct import
        set_target_properties(_runtime PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/ggmlc"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/ggmlc"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/ggmlc"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/ggmlc"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/ggmlc"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/ggmlc"
        )

        install(TARGETS _runtime DESTINATION ggmlc)
    endif()
endif()
