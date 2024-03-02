llvm_map_components_to_libnames(llvm_libs support core irreader)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  ASTLDialect
)

add_llvm_executable(onnx-parser
  onnx-parser.cpp
  DEPENDS
  ${LIBS}
)
target_link_libraries(onnx-parser PRIVATE ${LIBS})
llvm_update_compile_flags(onnx-parser)
mlir_check_all_link_libraries(onnx-parser)
install(TARGETS onnx-parser DESTINATION bin)
