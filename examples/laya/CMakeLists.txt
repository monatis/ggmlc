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

if (GGMLC_ENABLE_AMDXDNA)
    file(RELATIVE_PATH _laya_library_dir
        "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
    get_filename_component(_laya_xrt_library_dir "${GGMLC_XRT_COREUTIL_LIBRARY}" DIRECTORY)
    set_target_properties(laya PROPERTIES
        INSTALL_RPATH "$ORIGIN/${_laya_library_dir};${_laya_xrt_library_dir}"
        INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()
install(TARGETS laya RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Laya)
install(FILES "${CMAKE_SOURCE_DIR}/third_party/ggml/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/laya/licenses"
    RENAME ggml-MIT.txt COMPONENT Laya)
if (EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
    install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/laya/licenses"
        RENAME ggmlc-LICENSE COMPONENT Laya)
endif()

if (GGMLC_ENABLE_AMDXDNA)
    set(GGMLC_LAYA_PACKAGE_NAME "laya-linux-x86_64-amdxdna"
        CACHE STRING "Base name of the local Laya release archive")
    set(GGMLC_LAYA_PACKAGE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/dist"
        CACHE PATH "Directory for the local Laya release archive")
    if (NOT GGMLC_LAYA_PACKAGE_NAME MATCHES "^[A-Za-z0-9][A-Za-z0-9._+-]*$")
        message(FATAL_ERROR "GGMLC_LAYA_PACKAGE_NAME must be a filename without directory separators")
    endif()
    set(_laya_build_host "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
    if (EXISTS "/etc/os-release" AND NOT CMAKE_CROSSCOMPILING)
        file(STRINGS "/etc/os-release" _laya_os_line REGEX "^PRETTY_NAME=")
        string(REGEX REPLACE "^PRETTY_NAME=\"|\"$" "" _laya_build_host "${_laya_os_line}")
    endif()
    configure_file(packaging/package.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/package-laya.cmake" @ONLY)
    add_custom_target(laya-package
        COMMAND "${CMAKE_COMMAND}" "-DBUILD_CONFIG=$<CONFIG>"
            -P "${CMAKE_CURRENT_BINARY_DIR}/package-laya.cmake"
        DEPENDS laya
        COMMENT "Packaging Laya and AMD XDNA runtime assets locally"
        VERBATIM)
endif()

if (WIN32)
    target_link_libraries(laya PRIVATE ws2_32)
endif()
