# ===----------------------------------------------------------------------
# Astralax General AI Compiler Platform
# 2023-08-26
# ===----------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW) 
endif()

project(astralax LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

message(STATUS "Enabling astralax tooling")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/astralax/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(ASTL_MAIN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/astralax/include")
set(ASTL_GEN_INCLUDE_DIR "${PROJECT_BINARY_DIR}/astralax/include")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${ASTL_MAIN_INCLUDE_DIR})
include_directories(${ASTL_GEN_INCLUDE_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")

add_subdirectory(astralax)
