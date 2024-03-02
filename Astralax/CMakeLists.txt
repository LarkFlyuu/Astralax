cmake_minimum_required(VERSION 3.21)

project(Astralax LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(ABSL_PROPAGATE_CXX_STD ON)

# ===------------------------------------------------------------------------
# Build ProtoBuf
# ===------------------------------------------------------------------------
macro(custom_protobuf_find)
  message(STATUS "Use custom protobuf build.")
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_BUILD_EXAMPLES "" OFF)
  option(protobuf_WITH_ZLIB "" OFF)
  option(protobuf_BUILD_SHARED_LIBS "" ${BUILD_SHARED_LIBS})
  
  set(UBSAN_FLAG "-fsanitize=undefined")

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/protobuf)
  
  if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf ALIAS libprotobuf)
    add_library(protobuf::libprotobuf-lite ALIAS libprotobuf-lite)
    
    if(NOT (ANDROID OR IOS))
      add_executable(protobuf::protoc ALIAS protoc)
    endif()
  endif()
endmacro()
set(ASTL_PROTOC_EXECUTABLE protobuf::protoc)

custom_protobuf_find()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(LLVM_ENABLE_ASSERTIONS ON)

set(MLIR_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/mlir")
set(LLVM_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project/build/lib/cmake/llvm")

find_package(MLIR REQUIRED CONFIG)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(astralax)
add_subdirectory(python)
