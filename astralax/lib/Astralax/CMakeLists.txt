add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transforms)
add_subdirectory(Builder)

add_mlir_library(ASTLInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  DEPENDS
  ASTLConversionIncGen
  ASTLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRQuantDialect
  ASTLDialect
  ASTLConversion
  ASTLTransforms
)