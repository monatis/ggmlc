cmake_minimum_required(VERSION 3.18)

option(GGMLC_BUILD_EXAMPLE_TAB_COMPLETION "Build tab completion example" ON)
option(GGMLC_BUILD_EXAMPLE_TIMESFM "Build TimesFM foundation forecasting example" ON)
option(GGMLC_BUILD_EXAMPLE_LAYA "Build Laya System 1 decision example" ON)

if (GGMLC_BUILD_EXAMPLE_TAB_COMPLETION AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tab_completion/CMakeLists.txt")
    add_subdirectory(tab_completion)
endif()

if (GGMLC_BUILD_EXAMPLE_TIMESFM AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/timesfm/CMakeLists.txt")
    add_subdirectory(timesfm)
endif()

if (GGMLC_BUILD_EXAMPLE_LAYA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/laya/CMakeLists.txt")
    add_subdirectory(laya)
endif()
