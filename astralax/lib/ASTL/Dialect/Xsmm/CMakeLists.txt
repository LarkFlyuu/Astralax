add_mlir_dialect_library(ASTLXsmmDialect
  # Ops and dialects
    XsmmEnum.cpp
    XsmmDialect.cpp
    XsmmOps.cpp
    XsmmUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ASTL

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRXsmmAttrDefIncGen
    MLIRXsmmOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(ASTLXsmmDialect
  PUBLIC
    $<BUILD_INTERFACE:${ASTL_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ASTL_MAIN_INCLUDE_DIR}>
)
