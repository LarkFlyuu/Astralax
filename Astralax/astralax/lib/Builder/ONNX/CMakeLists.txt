function(protobuf_generate_cpp srcs_var hdrs_var)
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)
    list(APPEND ${srcs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc"
              "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h"
      COMMAND ${ASTL_PROTOC_EXECUTABLE} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I${CMAKE_CURRENT_SOURCE_DIR} ${abs_fil}
      DEPENDS ${ASTL_PROTOC_EXECUTABLE} ${abs_file}
      COMMENT "Running C++ protocol buffer compiler on ${ASTL_PROTOC_EXECUTABLE}  ${fil}" VERBATIM
    )
    set_source_files_properties(${${srcs_var}} ${${hdrs_var}} PROPERTIES GENERATED TRUE)
    set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
    set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  endforeach()
endfunction()

file(GLOB OnnxProtobufFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
set(OnnxSerializerSrc "${CMAKE_CURRENT_SOURCE_DIR}/OnnxConverter.cpp")

protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${OnnxProtobufFiles})
message(STATUS "ONNX_PROTO_SRCS: ${ONNX_PROTO_SRCS}")
message(STATUS "ONNX_PROTO_HDRS: ${ONNX_PROTO_HDRS}")

list(APPEND OnnxSerializerSrc ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})

add_library(ONNXSerializer STATIC ${OnnxSerializerSrc})
target_link_libraries(
  ONNXSerializer PRIVATE
  protobuf::libprotobuf
  HldDialect
  Utils
)
target_include_directories(ONNXSerializer PUBLIC ${CMAKE_SOURCE_DIR}/third_party/protobuf/src)