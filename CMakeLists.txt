cmake_minimum_required(VERSION 3.24)

project(gaiazero_cpp VERSION 0.1.0 LANGUAGES CXX)

option(GAIA_BUILD_TESTS "Build the C++ contract and inference smoke tests" ON)
option(GAIA_ENABLE_CUDA "Enable optional CUDA runtime wiring" OFF)
option(GAIA_ENABLE_ORT_CPU "Enable optional ONNX Runtime SDK wiring" OFF)
option(GAIA_ENABLE_TENSORRT "Enable optional TensorRT SDK wiring" OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime SDK root (contains include/ and lib/)")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT SDK root (contains include/ and lib/)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(GAIA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(gaiazero_core
    cpp/src/contracts.cpp
    cpp/src/inference.cpp
)
add_library(gaiazero::core ALIAS gaiazero_core)

target_include_directories(gaiazero_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
        $<INSTALL_INTERFACE:include>
)

if(GAIA_ENABLE_CUDA)
    target_compile_definitions(gaiazero_core PUBLIC GAIA_HAS_CUDA=1)
    target_link_libraries(gaiazero_core PUBLIC CUDA::cudart)
endif()

if(MSVC)
    target_compile_options(gaiazero_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(gaiazero_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GAIA_ENABLE_ORT_CPU)
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "GAIA_ENABLE_ORT_CPU=ON requires -DONNXRUNTIME_ROOT=<SDK root>")
    endif()
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNXRUNTIME_ROOT does not contain include/onnxruntime_cxx_api.h")
    endif()
    add_library(gaiazero_onnxruntime INTERFACE)
    target_include_directories(gaiazero_onnxruntime INTERFACE "${ONNXRUNTIME_ROOT}/include")
    target_compile_definitions(gaiazero_core PUBLIC GAIA_HAS_ONNXRUNTIME=1)
    target_link_libraries(gaiazero_core PUBLIC gaiazero_onnxruntime)
endif()

if(GAIA_ENABLE_TENSORRT)
    if(NOT TENSORRT_ROOT)
        message(FATAL_ERROR "GAIA_ENABLE_TENSORRT=ON requires -DTENSORRT_ROOT=<SDK root>")
    endif()
    if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
        message(FATAL_ERROR "TENSORRT_ROOT does not contain include/NvInfer.h")
    endif()
    add_library(gaiazero_tensorrt INTERFACE)
    target_include_directories(gaiazero_tensorrt INTERFACE "${TENSORRT_ROOT}/include")
    target_compile_definitions(gaiazero_core PUBLIC GAIA_HAS_TENSORRT=1)
    target_link_libraries(gaiazero_core PUBLIC gaiazero_tensorrt)
endif()

if(GAIA_BUILD_TESTS)
    enable_testing()
    add_executable(gaiazero_cpp_smoke cpp/tests/smoke.cpp)
    target_link_libraries(gaiazero_cpp_smoke PRIVATE gaiazero::core)
    if(MSVC)
        target_compile_options(gaiazero_cpp_smoke PRIVATE /W4 /permissive-)
    else()
        target_compile_options(gaiazero_cpp_smoke PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME gaiazero_cpp_smoke COMMAND gaiazero_cpp_smoke)
endif()
