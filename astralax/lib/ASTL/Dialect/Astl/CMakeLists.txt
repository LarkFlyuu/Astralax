add_mlir_dialect_library(ASTLAstlDialect
  # Ops and dialects
    AstlDialect.cpp
    AstlOps.cpp
    AstlUtils.cpp
    AstlAttr.cpp
    AstlTraits.cpp
    AstlInterface.cpp
    BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ASTL

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRAstlOpsIncGen
    MLIRAstlAttrDefIncGen
    ASTLIR
    MLIRAstlInterfaceIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    ASTLIR
)

target_include_directories(ASTLAstlDialect
  PUBLIC
  $<BUILD_INTERFACE:${ASTL_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${ASTL_MAIN_INCLUDE_DIR}>
)
