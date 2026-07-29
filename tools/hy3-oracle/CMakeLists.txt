cmake_minimum_required(VERSION 3.20)
project(lightbridge_hy3_llama_oracle LANGUAGES C CXX)

if(NOT DEFINED LLAMA_CPP_SOURCE)
    message(FATAL_ERROR "LLAMA_CPP_SOURCE must point to the pinned llama.cpp checkout")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_SOURCE}" "${CMAKE_BINARY_DIR}/llama.cpp")

add_executable(bridge-hy3-llama-oracle llama-oracle.cpp)
target_compile_features(bridge-hy3-llama-oracle PRIVATE cxx_std_17)
target_link_libraries(bridge-hy3-llama-oracle PRIVATE llama ggml)
target_include_directories(
    bridge-hy3-llama-oracle
    PRIVATE
        "${LLAMA_CPP_SOURCE}/include"
        "${LLAMA_CPP_SOURCE}/ggml/include"
)
set_target_properties(
    bridge-hy3-llama-oracle
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)

add_executable(bridge-hy3-full-model-oracle full-model-oracle.cpp)
target_compile_features(bridge-hy3-full-model-oracle PRIVATE cxx_std_17)
target_link_libraries(bridge-hy3-full-model-oracle PRIVATE llama ggml)
target_include_directories(
    bridge-hy3-full-model-oracle
    PRIVATE
        "${LLAMA_CPP_SOURCE}/include"
        "${LLAMA_CPP_SOURCE}/ggml/include"
)
set_target_properties(
    bridge-hy3-full-model-oracle
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)
