add_mlir_dialect_library(ASTLLinalgXTransformOps
    LinalgXTransformOps.cpp
    IteratorCollapsing.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ASTL

  DEPENDS
    ASTLLinalgXTransformOpsIncGen
)

target_include_directories(ASTLLinalgXTransformOps
  PUBLIC
    $<BUILD_INTERFACE:${ASTL_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ASTL_MAIN_INCLUDE_DIR}>
)
