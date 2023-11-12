add_library(Operator SHARED Operator.cpp)
target_link_libraries(Operator PUBLIC ASTLDialect)

function(add_astl_op_lib op_name)
  add_library(${op_name} SHARED ${op_name}.cpp)
  target_link_libraries(${op_name} Operator)
  set_target_properties(${op_name} PROPERTIES
    OUTPUT_NAME "${op_name}"
    VERSION 1.0
    SOVERSION 1
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
endfunction()

set_target_properties(Operator PROPERTIES  
  OUTPUT_NAME "Operator"
  VERSION 1.0
  SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_astl_op_lib("AddOp")
add_astl_op_lib("SubOp")
add_astl_op_lib("Conv3x3s1p1_i8xi10ti8")
