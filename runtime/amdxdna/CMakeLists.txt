if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
    NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    message(FATAL_ERROR "GGMLC_ENABLE_AMDXDNA requires Linux x86_64 and an AMD XDNA2 NPU")
endif()

get_filename_component(_amdxdna_vendor "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/amdxdna" ABSOLUTE)

# Migrate the former default in existing build trees. Keep an explicitly
# selected, different xclbin; its ABI must match the bundled GEMM library.
if (GGMLC_FLM_INCLUDE_DIR)
    get_filename_component(_old_flm_src "${GGMLC_FLM_INCLUDE_DIR}/.." ABSOLUTE)
    if (GGMLC_AMDXDNA_XCLBIN STREQUAL "${_old_flm_src}/xclbins/Qwen3-0.6B-NPU2/mm.xclbin")
        unset(GGMLC_AMDXDNA_XCLBIN CACHE)
    endif()
endif()
foreach (_obsolete GGMLC_FASTFLOWLM_ROOT GGMLC_FLM_INCLUDE_DIR GGMLC_FLM_GEMM_LIBRARY GGMLC_FLM_AIEBU_LIBRARY)
    unset(${_obsolete} CACHE)
endforeach()
set(GGMLC_AMDXDNA_XCLBIN "${_amdxdna_vendor}/xclbins/mm.xclbin"
    CACHE FILEPATH "XDNA2 BF16 GEMM xclbin matching the bundled libgemm.so")
foreach (_file include/modules/gemm.hpp lib/libgemm.so lib/libaiebu.a)
    if (NOT EXISTS "${_amdxdna_vendor}/${_file}")
        message(FATAL_ERROR "Bundled amdxdna file missing: third_party/amdxdna/${_file}; restore it from the ggmlc checkout")
    endif()
endforeach()
if (NOT EXISTS "${GGMLC_AMDXDNA_XCLBIN}")
    message(FATAL_ERROR "AMD XDNA2 GEMM xclbin missing: ${GGMLC_AMDXDNA_XCLBIN}")
endif()

# Copy runtime assets into the build tree. Compiled programs need neither the
# upstream checkout nor the vendored source directory to load these assets.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib" "${CMAKE_CURRENT_BINARY_DIR}/assets")
configure_file("${_amdxdna_vendor}/lib/libgemm.so" "lib/libgemm.so" COPYONLY)
configure_file("${GGMLC_AMDXDNA_XCLBIN}" "assets/mm.xclbin" COPYONLY)
add_library(ggmlc_amdxdna_gemm SHARED IMPORTED GLOBAL)
set_target_properties(ggmlc_amdxdna_gemm PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/lib/libgemm.so"
    IMPORTED_NO_SONAME TRUE)
add_library(ggmlc_amdxdna_aiebu STATIC IMPORTED GLOBAL)
set_target_properties(ggmlc_amdxdna_aiebu PROPERTIES
    IMPORTED_LOCATION "${_amdxdna_vendor}/lib/libaiebu.a")

set(GGMLC_XRT_ROOT "" CACHE PATH "XRT development prefix (normally supplied by libxrt-dev)")
find_path(GGMLC_XRT_INCLUDE_DIR xrt/xrt_device.h
    HINTS "${GGMLC_XRT_ROOT}/include" "$ENV{XILINX_XRT}/include" /opt/xilinx/xrt/include)
find_library(GGMLC_XRT_COREUTIL_LIBRARY NAMES libxrt_coreutil.so.2 xrt_coreutil
    HINTS "${GGMLC_XRT_ROOT}/lib" "${GGMLC_XRT_ROOT}/lib/x86_64-linux-gnu"
          "$ENV{XILINX_XRT}/lib" /opt/xilinx/xrt/lib)
find_path(GGMLC_UUID_INCLUDE_DIR uuid/uuid.h HINTS "${GGMLC_XRT_ROOT}/include")
find_path(GGMLC_DRM_INCLUDE_DIR drm/drm.h)
if (NOT GGMLC_XRT_INCLUDE_DIR OR NOT GGMLC_XRT_COREUTIL_LIBRARY OR
    NOT GGMLC_UUID_INCLUDE_DIR OR NOT GGMLC_DRM_INCLUDE_DIR)
    message(FATAL_ERROR "XRT development files missing: install libxrt-dev uuid-dev libdrm-dev, or set GGMLC_XRT_ROOT")
endif()

# Keep the vendor C++ headers and their CPU ISA requirements in this adapter.
# The rest of ggmlc remains C++17 and has no XRT dependency in CPU-only builds.
add_library(ggmlc_amdxdna STATIC amdxdna.cpp)
target_compile_features(ggmlc_amdxdna PRIVATE cxx_std_20)
target_compile_options(ggmlc_amdxdna PRIVATE -mavx2)
target_include_directories(ggmlc_amdxdna SYSTEM PRIVATE
    "${_amdxdna_vendor}/include" "${GGMLC_XRT_INCLUDE_DIR}"
    "${GGMLC_UUID_INCLUDE_DIR}" "${GGMLC_DRM_INCLUDE_DIR}")
file(RELATIVE_PATH _amdxdna_installed_xclbin
    "${CMAKE_INSTALL_FULL_BINDIR}"
    "${CMAKE_INSTALL_FULL_DATADIR}/laya/amdxdna/mm.xclbin")
target_compile_definitions(ggmlc_amdxdna PRIVATE
    GGMLC_AMDXDNA_DEFAULT_XCLBIN="${CMAKE_CURRENT_BINARY_DIR}/assets/mm.xclbin"
    GGMLC_AMDXDNA_INSTALLED_XCLBIN="${_amdxdna_installed_xclbin}")
target_link_libraries(ggmlc_amdxdna PUBLIC ${GGML_BACKEND_LIBS}
    ggmlc_amdxdna_gemm ggmlc_amdxdna_aiebu
    "${GGMLC_XRT_COREUTIL_LIBRARY}" ${CMAKE_DL_LIBS})
message(STATUS "ggmlc amdxdna: bundled GEMM=${_amdxdna_vendor}, XRT=${GGMLC_XRT_COREUTIL_LIBRARY}")

# aiebu is linked statically. XRT and the XDNA plugin are system dependencies.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/lib/libgemm.so"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Laya)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/assets/mm.xclbin"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/laya/amdxdna" COMPONENT Laya)
install(DIRECTORY "${_amdxdna_vendor}/licenses"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/laya/amdxdna" COMPONENT Laya)
install(FILES "${_amdxdna_vendor}/README.md" "${_amdxdna_vendor}/PROVENANCE.json"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/laya/amdxdna" COMPONENT Laya)
