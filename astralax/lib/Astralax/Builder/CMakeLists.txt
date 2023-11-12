set(ONNX_PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "ONNX_PROTO_GEN_DIR: ${ONNX_PROTO_GEN_DIR}")
include_directories(${ONNX_PROTO_GEN_DIR})

include_directories(${CMAKE_SOURCE_DIR}/third_party/protobuf/third_party/abseil-cpp)
message(STATUS ${CMAKE_SOURCE_DIR}/third_party/protobuf/third_party/abseil-cpp)

add_subdirectory(ONNX)

file(GLOB _srcs *.cpp)

add_mlir_library(ASTLBuilder
    ${_srcs}

    DEPENDS
    MLIRAstlOpsIncGen
    MLIRAstlAttrIncGen
  
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRFuncDialect
    MLIRRewrite
    MLIRTransforms
    ASTLDialect
    ASTLHelper
    ONNXSerializer
)