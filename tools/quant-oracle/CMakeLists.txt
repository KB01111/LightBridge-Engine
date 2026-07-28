cmake_minimum_required(VERSION 3.20)

project(lightbridge_quant_oracle LANGUAGES C CXX)

find_package(Git REQUIRED)

if(NOT DEFINED LLAMA_CPP_SOURCE)
  message(FATAL_ERROR "LLAMA_CPP_SOURCE must name the authenticated llama.cpp b10153 checkout")
endif()

cmake_path(ABSOLUTE_PATH LLAMA_CPP_SOURCE NORMALIZE OUTPUT_VARIABLE LLAMA_CPP_SOURCE_ABS)
if(NOT EXISTS "${LLAMA_CPP_SOURCE_ABS}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "LLAMA_CPP_SOURCE does not contain ggml/CMakeLists.txt")
endif()

set(LLAMA_CPP_REVISION "b77d646751d01c0962bc203b6809e9d94f7d50b7")
set(LLAMA_CPP_RELEASE "b10153")
set(LLAMA_CPP_REPOSITORY "https://github.com/ggml-org/llama.cpp.git")
file(TO_CMAKE_PATH "${LLAMA_CPP_SOURCE_ABS}" LLAMA_CPP_SAFE_DIRECTORY)

function(run_authenticated_git output_variable)
  execute_process(
    COMMAND
      "${GIT_EXECUTABLE}"
      -c "safe.directory=${LLAMA_CPP_SAFE_DIRECTORY}"
      -C "${LLAMA_CPP_SOURCE_ABS}"
      ${ARGN}
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE git_output
    ERROR_VARIABLE git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT git_result EQUAL 0)
    message(FATAL_ERROR "Authenticated git command failed: ${git_error}")
  endif()
  set("${output_variable}" "${git_output}" PARENT_SCOPE)
endfunction()

run_authenticated_git(checkout_head rev-parse HEAD)
if(NOT checkout_head STREQUAL LLAMA_CPP_REVISION)
  message(FATAL_ERROR "LLAMA_CPP_SOURCE HEAD is not the pinned b10153 revision")
endif()

run_authenticated_git(release_head rev-parse "refs/tags/${LLAMA_CPP_RELEASE}^{}")
if(NOT release_head STREQUAL LLAMA_CPP_REVISION)
  message(FATAL_ERROR "llama.cpp release tag b10153 does not resolve to the pinned revision")
endif()

run_authenticated_git(checkout_remote remote get-url origin)
if(NOT checkout_remote STREQUAL LLAMA_CPP_REPOSITORY)
  message(FATAL_ERROR "LLAMA_CPP_SOURCE origin is not the authenticated repository")
endif()

run_authenticated_git(checkout_status status --porcelain=v1 --untracked-files=all)
if(NOT checkout_status STREQUAL "")
  message(FATAL_ERROR "LLAMA_CPP_SOURCE must have a clean tracked and untracked worktree")
endif()

set(
  AUTHENTICATED_SOURCE_IDENTITIES
  "ggml/CMakeLists.txt|a766e49ea11d2630e7d0acd6407e80d664e3f283"
  "ggml/src/CMakeLists.txt|82e9480c2f240a66d2269198d5340b4e8565da3a"
  "ggml/src/ggml.c|a7d1fe7d94be4bee3df47f0d710fbfdb62087d1f"
  "ggml/src/ggml-common.h|83f9118da84a6a61967a5c8a04af9893130a4e95"
  "ggml/src/ggml-quants.c|1ebc50a763f16db909de37090da38cc8c0fdde94"
  "ggml/src/ggml-quants.h|75188f1af180e3592f3c94bc4077989fb817c359"
  "ggml/src/ggml-impl.h|62b76abbcec9e71c860ba1a99d79b501bad26b93"
  "ggml/src/ggml-cpu/ggml-cpu.c|491316f7491252248d6f74a60440d3efa7aa6177"
  "ggml/src/ggml-cpu/ggml-cpu-impl.h|5d1ca5ffcc368b9f0249d6cf6ccc4549bb9a3ab4"
  "ggml/src/ggml-cpu/quants.c|5e36459f8cbc5900b375d2189414307393471a6b"
  "ggml/src/ggml-cpu/quants.h|93ea7eeffe5b00ad2c612aac49b7983c12949525"
  "src/models/hy-v3.cpp|47a0beaf217f19219e3e8fb8d5c35664625d7c73"
  "tests/test-quantize-fns.cpp|9510ac14ce00805e1689a8c8b16b6dd6c329911c"
)
foreach(source_identity IN LISTS AUTHENTICATED_SOURCE_IDENTITIES)
  string(REPLACE "|" ";" source_fields "${source_identity}")
  list(GET source_fields 0 source_path)
  list(GET source_fields 1 expected_blob)
  run_authenticated_git(commit_blob rev-parse "HEAD:${source_path}")
  if(NOT commit_blob STREQUAL expected_blob)
    message(FATAL_ERROR "Pinned commit blob identity drift for ${source_path}")
  endif()
  run_authenticated_git(worktree_blob hash-object "${LLAMA_CPP_SOURCE_ABS}/${source_path}")
  if(NOT worktree_blob STREQUAL expected_blob)
    message(FATAL_ERROR "Working-tree source drift for ${source_path}")
  endif()
endforeach()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(
  "${LLAMA_CPP_SOURCE_ABS}/ggml"
  "${CMAKE_BINARY_DIR}/ggml"
  EXCLUDE_FROM_ALL
)

if(TARGET ggml-cpu)
  message(FATAL_ERROR "The scalar oracle must not build or link the ggml-cpu dispatch target")
endif()

add_executable(
  bridge-quant-oracle
  oracle.cpp
  "${LLAMA_CPP_SOURCE_ABS}/ggml/src/ggml-cpu/quants.c"
)

target_include_directories(
  bridge-quant-oracle
  PRIVATE
    "${LLAMA_CPP_SOURCE_ABS}/ggml/include"
    "${LLAMA_CPP_SOURCE_ABS}/ggml/src"
    "${LLAMA_CPP_SOURCE_ABS}/ggml/src/ggml-cpu"
)

target_link_libraries(bridge-quant-oracle PRIVATE ggml-base)
target_compile_features(bridge-quant-oracle PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(ggml-base PRIVATE /fp:strict)
  target_compile_options(bridge-quant-oracle PRIVATE /fp:strict)
else()
  target_compile_options(
    ggml-base
    PRIVATE
      -fno-fast-math
      -ffp-contract=off
  )
  target_compile_options(
    bridge-quant-oracle
    PRIVATE
      -fno-fast-math
      -ffp-contract=off
  )
endif()

set_target_properties(
  bridge-quant-oracle
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin"
)
