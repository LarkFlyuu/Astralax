add_mlir_dialect_library(TPPLinalgXTransformOps
    LinalgXTransformOps.cpp
    IteratorCollapsing.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPLinalgXTransformOpsIncGen
)

target_include_directories(TPPLinalgXTransformOps
  PUBLIC
    $<BUILD_INTERFACE:${ASTL_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ASTL_MAIN_INCLUDE_DIR}>
)
