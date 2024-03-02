add_mlir_library(Utils
  MLIRHelper.cpp

  LINK_COMPONENTS
  Core

  # DEPENDS
  # ASTLDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRBytecodeWriter
  MLIRBytecodeReader
  # ASTLDialect
)