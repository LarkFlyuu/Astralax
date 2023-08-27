add_mlir_library(ASTLIR
  StructuredOpMatcher.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ASTL

  DEPENDS
    MLIRLinalgDialect
)
