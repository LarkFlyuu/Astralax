add_mlir_library(ASTLHelper
  Helper.cpp

  LINK_COMPONENTS
  Core

  DEPENDS
  ASTLDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRQuantDialect
  ASTLDialect
)