add_subdirectory(Dialect)
add_subdirectory(IR)
add_subdirectory(GPU)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(MLIRASTL
  # Passes 
    Bufferize.cpp
    ToBlockLayoutAndBack.cpp
    RewriteToBatchReduceGemm.cpp
    TransformDialectInterpreter.cpp
    RewriteConvToMatmulImpl.cpp
    TileConsumerAndFuseProducers.cpp
    RewriteConvsToMatmulOrBrgemm.cpp
    DefaultAstlPasses.cpp
    GeneralizeTensorPackAndUnPack.cpp
    ConstantFoldPack.cpp
    ConvInitSimplify.cpp
    ConvertForAllToParallelOp.cpp
    CombineAstl.cpp
    RewriteBatchMatmulToMatmul.cpp
    LinalgDeGeneralize.cpp
    ConvertMemRefToAstl.cpp
    DefaultPipeline.cpp
    ConvertLinalgToXsmm.cpp
    DecomposeAggregatedOps.cpp

  # Utils
    MatcherUtils.cpp
    TensorInit.cpp
    TensorInitFloat.cpp
    TensorInitInt.cpp
    BuilderUtils.cpp
    TransformUtils.cpp
    ValueUtils.cpp
    VNNIUtils.cpp

  # Conversions
    ConvertLinalgToAstl.cpp
    ConvertAstlToLoops.cpp
    ConvertAstlToXsmm.cpp
    ConvertXsmmToFunc.cpp
    ConvertCheckToLoops.cpp
    ConvertPerfToLoops.cpp
    ConvertPerfToFunc.cpp
    ConvertPackOptimization.cpp

  # Test Passes
    TestMatchers.cpp
    TestForToForAllRewrite.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ASTL

  DEPENDS
    ASTLCompilerPassIncGen
    ASTLLinalgXTransformOps

    LINK_LIBS PUBLIC
    ASTLIR
    ASTLAstlDialect
    ASTLXsmmDialect
    ASTLPerfDialect
    ASTLGPU

    MLIRIR
    MLIRInferTypeOpInterface
    ${mlir_dialect_libs}
    ${conversion_libs}
)

target_include_directories(MLIRASTL
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${ASTL_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ASTL_MAIN_INCLUDE_DIR}>
)
