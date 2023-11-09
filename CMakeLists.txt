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

set(MLIR_DIR /data/llvm-project/build/lib/cmake/mlir)

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

function(add_astl_mlir_dialect dialect dialect_name)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}Ops.hpp.inc -gen-op-decls)
  mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Dialect.hpp.inc -gen-dialect-decls -dialect=${dialect_name})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_name})
  mlir_tablegen(${dialect}Types.hpp.inc -gen-typedef-decls -typedefs-dialect=${dialect_name})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_name})
  mlir_tablegen(${dialect}Attributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=${dialect_name})
  mlir_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${dialect_name})
  add_public_tablegen_target(OM${dialect}IncGen)
endfunction()

function(add_astl_mlir_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.hpp.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(OM${interface}IncGen)
endfunction()

function(add_astl_mlir_library name)
  cmake_parse_arguments(ARG
    "EXCLUDE_FROM_OM_LIBS;NO_INSTALL"
    ""
    "DEPENDS;INCLUDE_DIRS;ACCEL_INCLUDE_DIRS;LINK_LIBS;LINK_COMPONENTS"
    ${ARGN}
    )

  if (NOT ARG_EXCLUDE_FROM_OM_LIBS)
    set_property(GLOBAL APPEND PROPERTY ONNX_MLIR_LIBS ${name})
  endif()

  add_library(${name} ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})

  if (ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()

  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} ${ARG_INCLUDE_DIRS})
  endif()

  if (ARG_ACCEL_INCLUDE_DIRS)
    target_include_directories(${name} ${ARG_ACCEL_INCLUDE_DIRS})
  endif()

  target_include_directories(${name}
    PUBLIC
    ${ONNX_MLIR_SRC_ROOT}
    ${ONNX_MLIR_BIN_ROOT}
    )

  if (ARG_LINK_LIBS)
    target_link_libraries(${name} ${ARG_LINK_LIBS})
  endif()

  if (ARG_LINK_COMPONENTS)
    set(LinkageMode)
    if (ARG_LINK_COMPONENTS MATCHES "^(PUBLIC|PRIVATE|INTERFACE)")
      list(POP_FRONT ARG_LINK_COMPONENTS LinkageMode)
    endif()

    llvm_map_components_to_libnames(COMPONENT_LIBS ${ARG_LINK_COMPONENTS})

    if (LinkageMode)
      target_link_libraries(${name} ${LinkageMode} ${COMPONENT_LIBS})
    else()
      target_link_libraries(${name} PRIVATE ${COMPONENT_LIBS})
    endif()
  endif()

  if (NOT ARG_NO_INSTALL)
    install(TARGETS ${name}
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin
      )
  endif()
endfunction(add_astl_mlir_library)

add_subdirectory(astralax)
