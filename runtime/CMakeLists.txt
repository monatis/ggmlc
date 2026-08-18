add_library(ggmlc_runtime STATIC
    src/loader.cpp
    src/executor.cpp
)

target_link_libraries(ggmlc_runtime PUBLIC
    ggml_lib
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
)

target_link_libraries(ggmlc_runtime_shared PUBLIC
    ggml_lib
)
