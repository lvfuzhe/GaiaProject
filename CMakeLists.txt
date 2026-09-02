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
    cpp/src/onnxruntime_backend.cpp
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
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            "${ONNXRUNTIME_ROOT}/include"
            "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
        NO_DEFAULT_PATH
    )
    if(NOT ONNXRUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR "ONNXRUNTIME_ROOT does not contain onnxruntime_cxx_api.h")
    endif()
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
            "${ONNXRUNTIME_ROOT}/lib"
            "${ONNXRUNTIME_ROOT}/lib/win-x64"
            "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native"
            "${ONNXRUNTIME_ROOT}/build/Windows/Release"
        NO_DEFAULT_PATH
    )
    if(NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "Could not find the ONNX Runtime library under ONNXRUNTIME_ROOT")
    endif()
    add_library(gaiazero_onnxruntime INTERFACE)
    target_include_directories(gaiazero_onnxruntime INTERFACE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(gaiazero_onnxruntime INTERFACE "${ONNXRUNTIME_LIBRARY}")
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
    if(GAIA_ENABLE_ORT_CPU AND WIN32)
        find_file(ONNXRUNTIME_RUNTIME_LIBRARY
            NAMES onnxruntime.dll
            PATHS
                "${ONNXRUNTIME_ROOT}/lib"
                "${ONNXRUNTIME_ROOT}/lib/win-x64"
                "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native"
                "${ONNXRUNTIME_ROOT}/build/Windows/Release"
            NO_DEFAULT_PATH
        )
        if(ONNXRUNTIME_RUNTIME_LIBRARY)
            add_custom_command(TARGET gaiazero_cpp_smoke POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_RUNTIME_LIBRARY}"
                    "$<TARGET_FILE_DIR:gaiazero_cpp_smoke>"
                VERBATIM
            )
        endif()
    endif()
    add_test(NAME gaiazero_cpp_smoke COMMAND gaiazero_cpp_smoke)
endif()
