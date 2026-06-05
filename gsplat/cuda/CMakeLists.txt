cmake_minimum_required(VERSION 3.20)

project(gsplat_cuda LANGUAGES CXX CUDA)

set(RAST_NAME GSplatCUDA)

option(ENABLE_CUDA_DEBUG "Enable CUDA kernel debugging flags" ON)
option(GSPLAT_FAST_MATH "Match build.py FAST_MATH=1 and pass -use_fast_math to nvcc" ON)
option(GSPLAT_BUILD_CAMERA_WRAPPERS "Build Python-oriented camera wrapper debug helpers" OFF)
option(GSPLAT_BUILD_DEBUG_RUNNER "Build tests/run.cu as a native debug entry" ON)

set(GSPLAT_CUDA_ARCHITECTURES "70;75;86" CACHE STRING "CUDA architectures")
set(GSPLAT_NUM_CHANNELS "" CACHE STRING "Optional GSPLAT_NUM_CHANNELS override, for example: 1,2,3,4")
set(TORCH_CUDA_ARCH_LIST "7.0;7.5;8.6" CACHE STRING "CUDA architectures passed to TorchConfig")

set(GSPLAT_BUILD_2DGS "0" CACHE STRING "Optional override: 0 or 1")
set(GSPLAT_BUILD_3DGS "0" CACHE STRING "Optional override: 0 or 1")
set(GSPLAT_BUILD_RADEGS "0" CACHE STRING "Optional override: 0 or 1")
set(GSPLAT_BUILD_3DGUT "0" CACHE STRING "Optional override: 0 or 1")
set(GSPLAT_BUILD_ADAM "0" CACHE STRING "Optional override: 0 or 1")
set(GSPLAT_BUILD_RELOC "0" CACHE STRING "Optional override: 0 or 1")
set(GSPLAT_BUILD_LOSSES "0" CACHE STRING "Optional override: 0 or 1")

find_package(Torch REQUIRED)
find_package(OpenMP)

separate_arguments(TORCH_CXX_FLAGS_LIST UNIX_COMMAND "${TORCH_CXX_FLAGS}")

set(GSPLAT_SOURCES
    csrc/Adam.cpp
    csrc/AdamCUDA.cu
    csrc/Intersect.cpp
    csrc/IntersectTile.cu
    csrc/Null.cpp
    csrc/NullCUDA.cu
    csrc/Projection.cpp
    csrc/Projection2DGSFused.cu
    csrc/Projection2DGSPacked.cu
    csrc/ProjectionEWA3DGSFused.cu
    csrc/ProjectionEWA3DGSPacked.cu
    csrc/ProjectionEWASimple.cu
    csrc/ProjectionRaDeGSFused.cu
    csrc/ProjectionRaDeGSPacked.cu
    csrc/ProjectionUT3DGSFused.cu
    csrc/QuatScaleToCovar.cpp
    csrc/QuatScaleToCovarCUDA.cu
    csrc/Rasterization.cpp
    csrc/RasterizeToIndices2DGS.cu
    csrc/RasterizeToIndices3DGS.cu
    csrc/RasterizeToPixels2DGSBwd.cu
    csrc/RasterizeToPixels2DGSFwd.cu
    csrc/RasterizeToPixels3DGSBwd.cu
    csrc/RasterizeToPixels3DGSFwd.cu
    csrc/RasterizeToPixelsRaDeGSBwd.cu
    csrc/RasterizeToPixelsRaDeGSFwd.cu
    csrc/RasterizeToPixelsFromWorld3DGSBwd.cu
    csrc/RasterizeToPixelsFromWorld3DGSFwd.cu
    csrc/Relocation.cpp
    csrc/RelocationCUDA.cu
    csrc/SphericalHarmonics.cpp
    csrc/SphericalHarmonicsCUDA.cu
)

if(GSPLAT_BUILD_CAMERA_WRAPPERS)
    list(APPEND GSPLAT_SOURCES
        csrc/CameraWrappers.cu
        csrc/ExternalDistortionWrappers.cu
    )
endif()

add_library(${RAST_NAME} STATIC ${GSPLAT_SOURCES})
add_library(gsplat::cuda ALIAS ${RAST_NAME})

set_target_properties(${RAST_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${GSPLAT_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(${RAST_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/third_party/glm
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${RAST_NAME}
    PUBLIC
        ${TORCH_LIBRARIES}
)

target_compile_options(${RAST_NAME} PRIVATE
    ${TORCH_CXX_FLAGS_LIST}
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wno-attributes;-Wno-unknown-pragmas>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20;--forward-unknown-opts;--expt-relaxed-constexpr;-diag-suppress=20012,186>
)

if(GSPLAT_FAST_MATH)
    target_compile_options(${RAST_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    )
endif()

if(ENABLE_CUDA_DEBUG)
    message(STATUS "Enable CUDA Kernel Debugging")
    target_compile_options(${RAST_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-g;-O0>
        $<$<COMPILE_LANGUAGE:CUDA>:-G;-g;-lineinfo;-Xptxas=-O0>
    )
else()
    target_compile_options(${RAST_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    target_compile_definitions(${RAST_NAME} PRIVATE NDEBUG)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${RAST_NAME} PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(${RAST_NAME} PUBLIC AT_PARALLEL_OPENMP)
else()
    message(STATUS "Compiling without OpenMP...")
endif()

foreach(_feature IN ITEMS 2DGS 3DGS RADEGS 3DGUT ADAM RELOC LOSSES)
    if(NOT "${GSPLAT_BUILD_${_feature}}" STREQUAL "")
        target_compile_definitions(${RAST_NAME} PUBLIC "GSPLAT_BUILD_${_feature}=${GSPLAT_BUILD_${_feature}}")
    endif()
endforeach()

if(GSPLAT_BUILD_CAMERA_WRAPPERS)
    target_compile_definitions(${RAST_NAME} PUBLIC GSPLAT_BUILD_CAMERA_WRAPPERS=1)
endif()

if(NOT "${GSPLAT_NUM_CHANNELS}" STREQUAL "")
    target_compile_definitions(${RAST_NAME} PUBLIC "GSPLAT_NUM_CHANNELS=${GSPLAT_NUM_CHANNELS}")
endif()
