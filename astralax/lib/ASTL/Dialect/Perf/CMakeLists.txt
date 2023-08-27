add_mlir_dialect_library(ASTLPerfDialect
  # Ops and dialects
    BufferizableOpInterfaceImpl.cpp
    PerfDialect.cpp
    PerfOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ASTL

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRPerfOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(ASTLPerfDialect
  PUBLIC
  $<BUILD_INTERFACE:${ASTL_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${ASTL_MAIN_INCLUDE_DIR}>
)
