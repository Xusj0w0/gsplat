cmake_minimum_required(VERSION 3.20)

project(GSplatDEBUG LANGUAGES CUDA CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(Torch REQUIRED)
message(STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")
message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")

add_subdirectory(gsplat/cuda)
add_executable(GSplatDEBUG gsplat/cuda/tests/run.cu)
target_sources(GSplatDEBUG
PRIVATE
    gsplat/cuda/tests/utils/stbi_image_write.h
    gsplat/cuda/tests/utils/utils.h
)
target_include_directories(GSplatDEBUG
PUBLIC
    ${TORCH_INCLUDE_DIRS}
)
target_compile_options(GSplatDEBUG
PRIVATE
    ${TORCH_CXX_FLAGS_LIST}
)
target_link_libraries(GSplatDEBUG
PRIVATE
    gsplat::cuda
)