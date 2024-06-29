CMAKE_MINIMUM_REQUIRED(VERSION 3.22)

set(PRJ blockedAttention)
project(${PRJ} CXX CUDA)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "csrc/*.h" "csrc/*.cpp" "csrc/cuda/*.cu" "csrc/cuda/*.cuh")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# using torch.utils.cmake_prefix_path to get the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Anaconda REQUIRED)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CONDA_LIBRARY_DIR}/${CONDA_PYTHON}/site-packages/torch/share/cmake")

message("CMAKE_CURRENT_PATH: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_PREFIX: ${CMAKE_PREFIX_PATH}")
message("CUTLASS_INCLUDE: ${CUTLASS_INCLUDE_DIR}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${PYTHON_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/csrc" "{CMAKE_CURRENT_SOURCE_DIR}/csrc/cuda")

add_executable(${PRJ} ${srcs})
target_link_libraries(${PRJ} ${TORCH_LIBRARIES} ${PYTHON_LIB})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PRJ} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()