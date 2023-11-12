get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
)
set(LIBS
  MLIROptLib
  ASTLDialect
  ASTLBuilder
  ASTLConversion
  ASTLTransforms
  ASTLInitAll
  ${dialect_libs}
  ${conversion_libs}
)
add_llvm_executable(astl-run
  astl-run.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(astl-run PRIVATE ${LIBS})
llvm_update_compile_flags(astl-run)
mlir_check_all_link_libraries(astl-run)
install(TARGETS astl-run DESTINATION bin)
