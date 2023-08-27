//===- AstlPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_DIALECT_ASTL_PASSES
#define ASTL_DIALECT_ASTL_PASSES

include "mlir/Pass/PassBase.td"

def ConvertLinalgToAstl : Pass<"convert-linalg-to-astl", "func::FuncOp"> {
  let summary = "Convert linalg to astl.";
  let description = [{
    Convert linalg.generic (or named) operations to astl operations. 
    Linalg generic operations are converted using simple pattern 
    matching (i.e., see `isAstlAdd`).
  }];
  let constructor = "mlir::astl::createConvertLinalgToAstlPass()";
  let dependentDialects = ["linalg::LinalgDialect", "astl::AstlDialect"]; 
}

def ConvertMemRefToAstl : Pass<"convert-memref-to-astl", "func::FuncOp"> {
  let summary = "Convert memref ops to astl.";
  let description = [{
    Convert memref operations (i.e., memref.copy) to astl operations.
  }];
  let constructor = "mlir::astl::createConvertMemRefToAstlPass()";
  let dependentDialects = ["memref::MemRefDialect", "astl::AstlDialect"];
}

def ConvertAstlToLoops : Pass<"convert-astl-to-loops", "func::FuncOp"> {
  let summary = "Convert astl to loops";
  let constructor = "mlir::astl::createConvertAstlToLoopsPass()";
  let description = [{
    Convert astl operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect", "memref::MemRefDialect"];
  let options = [
    Option<"parallel", "parallel", "bool", "false", "use parallel loops">
  ];
}

def ConvertAstlToXsmm : Pass<"convert-astl-to-xsmm", "func::FuncOp"> {
  let summary = "Convert astl to xsmm";
  let constructor = "mlir::astl::createConvertAstlToXsmmPass()";
  let description = [{
    Convert astl operations to XSMM operations.
  }];
  let dependentDialects = ["func::FuncDialect", 
                           "memref::MemRefDialect",
                           "xsmm::XsmmDialect"];
}

def ConvertLinalgToXsmm : Pass<"convert-linalg-to-xsmm", "func::FuncOp"> {
  let summary = "Convert linalg to xsmm";
  let constructor = "mlir::astl::createConvertLinalgToXsmmPass()";
  let description = [{
    Convert linalg operations to XSMM operations.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "linalg::LinalgDialect",
                           "xsmm::XsmmDialect", 
                           "tensor::TensorDialect"];
}

def ConvertXsmmToFunc : Pass<"convert-xsmm-to-func", "ModuleOp"> {
  let summary = "Convert xsmm to func";
  let constructor = "mlir::astl::createConvertXsmmToFuncPass()";
  let description = [{
    Convert XSMM operations to libXSMM function calls.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "xsmm::XsmmDialect",
                           "LLVM::LLVMDialect"];
}

def ConvertCheckToLoops : Pass<"convert-check-to-loops", "func::FuncOp"> {
  let summary = "Convert check to loops";
  let constructor = "mlir::astl::createConvertCheckToLoopsPass()";
  let description = [{
    Convert check operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "ModuleOp"> {
  let summary = "Apply transform dialect operations one by one";
  let constructor = "mlir::astl::createTransformDialectInterpreterPass()";
  let description = [{
    Copy and paste from 'TestTransformDialectInterpreter.cpp'.
  }];
}

def ConvertPerfToLoops : Pass<"convert-perf-to-loops", "func::FuncOp"> {
  let summary = "Convert perf to loops";
  let constructor = "mlir::astl::createConvertPerfToLoopsPass()";
  let description = [{
    Convert perf operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertPerfToFunc : Pass<"convert-perf-to-func", "ModuleOp"> {
  let summary = "Convert perf to func";
  let constructor = "mlir::astl::createConvertPerfToFuncPass()";
  let description = [{
    Convert perf operations to function calls.
  }];
  let dependentDialects = ["func::FuncDialect", 
                           "math::MathDialect",
                           "memref::MemRefDialect",
                           "tensor::TensorDialect"];
}

def CombineAstlOps : Pass<"astl-combine", "func::FuncOp"> {
  let summary = "Combine astls into bigger astl";
  let constructor = "mlir::astl::createCombineAstlPass()";
  let description = [{
    Convert astl bias + brgemm + relu op to a larger op.
  }];
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

def TransformDropSchedulePass : Pass<"transform-drop-schedule", "ModuleOp"> {
  let summary = "Drop the transform schedule";
  let constructor = "mlir::astl::createTransformDropSchedulePass()";
}

def PackVNNI : Pass<"pack-vnni", "func::FuncOp"> {
  let summary = "Convert matmul/brgemm to vnni layout";
  let description = [{
    Relayout following matmuls and brgemm as following:
    - VNNI Matmul as: C[M][N]= A[M][K] * B[K/VNNI][N][VNNI]
    - VNNI Blocked Matmul as:
      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb/VNNI][jb][VNNI]
    - VNNI BRGemm as: C[M][N]= A[R][M][K] * B[R][K/VNNI][N][VNNI]
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t", 
               "Blocking factor for vnni layout">
  ];
  let constructor = "mlir::astl::createPackVNNIPass()";
  let dependentDialects = ["tensor::TensorDialect", "astl::AstlDialect"];
}

def PackMatmul : Pass<"pack-matmul", "func::FuncOp"> {
  let summary = "Convert matmul to block layout and back";
  let description = [{
    Block a linalg.matmul 
    as: [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb].
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t", 
               "Blocking factor for relayout">
  ];
  let constructor = "mlir::astl::createPackMatmulPass()";
}

def PackConv2DNchwFchw : Pass<"pack-conv2DNchwFchw", "func::FuncOp"> {
  let summary = "Convert Conv2DNchwFchw to block layout and back";
  let description = [{
    Block Conv2DNchwFchw as: [N][BK][P][Q][bk] += [N][BC][H][W][bc] * [BK][BC][R][S][bk][bc]
                             output            += image             * filter
    Pack the image's channel with a block factor BC.
    Pack the filter's channels C and K with a block factor of BC and BK.
    Pack the output's channel K with a block factor BK.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t",
               "Blocking factor for relayout">
  ];
  let constructor = "mlir::astl::createPackConv2DNchwFchwPass()";
}

def PackConv2DNhwcHwcf : Pass<"pack-conv2DNhwcHwcf", "func::FuncOp"> {
  let summary = "Pack and unpack Conv2DNhwcHwcf";
  let description = [{
    Pack Conv2DNhwcHwcf as [N][K'][P][Q][k] += [N][C'][H][W][c] * [K'][C'][R][S][c][k]
                           output           += image            * filter
    Pack the image and block the image's channel with a factor k.
    Pack the filter and block the filter's channels with k and c.
    Pack the output and block the output's channel with k.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t",
               "Blocking factor for pack and unpack operation">
  ];
  let constructor = "mlir::astl::createPackConv2DNhwcHwcfPass()";
}

def TileConsumerAndFuseProducers : Pass<"tile-consumer-and-fuse-producers", 
                                        "func::FuncOp"> {
  let summary = "Tile consumers and fuse producers";
  let description = [{
    The pass uses `TileConsumerAndFuseProducersUsingSCFForOp` to tile the
    consumer and fuse the consumer with the producers. The fusion anchor to matmul
    or conv-like patterns allows two additional options to control how many
    producers fuse together with the latched operation and how many consumers.
    Precisely, `max-depth` controls how many producers should be considered, while
    `start-from-last-consumer` allows to move the anchor point to the last fusable
    consumer of the conv or matmul-like pattern.
  }];
  let constructor = "mlir::astl::createTileConsumerAndFuseProducersPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"maxDepth", "max-depth", "int64_t", "5", 
           "Get producers till maxDepth">,
    Option<"startFromLastFusableConsumer", "start-from-last-consumer", "bool",
           "true", "Fuse from the last fusable consumer of the current target">,
    Option<"useForAll", "use-for-all", "bool", "true", "Use parallel forAll">
  ];
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect", "astl::AstlDialect"];
}

def RewriteConvToMatmulOrBrgemm : Pass<"rewrite-conv-to-matmul-or-brgemm", 
                                       "func::FuncOp"> {
  let summary = "Rewrite Conv2DNhwcHwcfOp/Conv2DNchwFchwOp to Matmul or Brgemm.";
  let description = [{
    Rewrite a convolution to a matmul or brgemm operation.
  }];
  let options = [
    Option<"enableBrgemm", "enable-brgemm", "bool", "false",
           "Rewrite convolution to BRGEMM if possible">
  ];
  let constructor = "mlir::astl::createRewriteConvToMatmulOrBrgemmPass()";
  let dependentDialects = ["scf::SCFDialect", "linalg::LinalgDialect"];
}

def RewriteBatchMatmulToMatmul : Pass<"rewrite-batch-matmul-to-matmul",
                                      "func::FuncOp"> {
  let summary = "Rewrite a linalg.batch_matmul to linalg.matmul.";
  let constructor = "mlir::astl::createRewriteBatchMatmulToMatmulPass()";
  let dependentDialects = ["scf::SCFDialect", "linalg::LinalgDialect"];
}

def ConvertPackOptimization : Pass<"pack-optimization", "func::FuncOp"> {
  let summary = "Optimize non-transpose pack patterns into faster implementation";
  let description =
      [{Optimize non-transpose pack patterns into faster implementation.}];
  let constructor = "mlir::astl::createConvertPackOptimization()";

  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect"];
}

def DefaultAstlPasses : Pass<"default-astl-passes", "ModuleOp"> {
  let summary = "Collection of default ASTL passes";
  let description = [{
    A collection of passes that lower everything ASTL-related
    to standard low-level dialects.
  }];
  let options= [
    Option<"astlToLoops", "astl-to-loops",
           "bool", /*default=*/"false",
           "By default ASTL ops are lowered to XSMM. Lower ASTL to loops instead.">,
    Option<"linalgToLoops", "linalg-to-loops",
           "bool", /*default=*/"false",
           "Skip all ASTL transformations. Lower linalg directly to loops.">,
    Option<"linalgToXsmm", "linalg-to-xsmm",
           "bool", /*default=*/"false",
           "Skip all ASTL transformations. Lower lianalg directly to xsmm.">
  ];
  let constructor = "mlir::astl::createDefaultAstlPass()";
}

def GeneralizeTensorPackAndUnPack : Pass<"generalize-tensor-pack-unpack",
                                         "func::FuncOp"> {
  let summary = "Generalize tensor.pack and tensor.unpack.";
  let description = [{
    Generalize a pack or unpack operation by first tiling, and then generalize
    it to other linalg operations.
  }];
  let constructor = "mlir::astl::createGeneralizeTensorPackAndUnPackPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def PropagatePackUnPack : Pass<"propagate-pack-and-unpack", "func::FuncOp"> {
  let summary = "Propagate tensor.pack and tensor.unpack";
  let description = [{
    Attempt to push tensor.pack and tensor.unpack at the boundaries. Currently,
    it propagates through linalg element-wise operations. Only one operand in the
    generic must come from a tensor.pack/tensor.unpack.
  }];
  let constructor = "mlir::astl::createPropagatePackUnPackPass()"; 
}

def SimplifyAndCanonicalizePack : Pass<"simplify-pack", "func::FuncOp"> {
  let summary = "Simplify and canonicalize tensor.pack";
  let constructor = "mlir::astl::createSimplifyAndCanonicalizePackPass()";
  let description = [{
    Apply `tensor.pack` and `tensor.unpack` canonicalization and simplification
    patterns.
  }];
}

def ConstantFoldPack : Pass<"constant-fold-pack", "ModuleOp"> {
  let summary = "Constant fold tensor.pack";
  let description = [{
    Reduce pack overhead by folding tensor.pack into constant tensors.
  }];
  let constructor = "mlir::astl::createConstantFoldPackPass()";
}

def ElementWiseFusion : Pass<"element-wise-fusion", "func::FuncOp"> {
  let summary = "Run linalg element-wise fusion";
  let constructor = "mlir::astl::createElementWiseFusionPass()";
}

def ConvInitSimplify : Pass<"conv-init-simplify", "func::FuncOp"> {
  let summary = "Simplify initialization for convolution";
  let description = [{
    Perform a graph-rewrite to simplify initialization for a Conv2DNhwcHwcfOp
    operation. Specifically, instead of initializing the output of a convolution
    with zero and then adding the bias, initialize the output with the bias.  
  }];
  let constructor = "mlir::astl::createConvInitSimplifyPass()";
}

def Bufferize : Pass<"bufferize", "ModuleOp"> {
  let summary = "Bufferize tensor to memref for the entire module";
  let constructor = "mlir::astl::createBufferizePass()";
  let options = [
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Only runs inplaceability analysis (for testing purposes only)">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Annotates IR with RaW conflicts. Requires test-analysis-only.">,
  ];
}

def Cleanup : Pass<"cleanup", "func::FuncOp"> {
  let summary = "General IR cleanup e.g., canonicalization, CSE etc.";
  let constructor = "mlir::astl::createCleanupPass()";
}

def Transform : Pass<"transform", "ModuleOp"> {
  let summary = "Runs transformation schedules and then drops them.";
  let constructor = "mlir::astl::createTransformPass()";
}

def LocalDialectsLowering : Pass<"lower-local-dialects", "ModuleOp"> {
  let summary = "Lower all local dialects (XSMM, check etc.).";
  let constructor = "mlir::astl::createLocalDialectsLoweringPass()";
}

def Postprocessing : Pass<"postprocess", "func::FuncOp"> {
  let summary = "IR postprocessing pass";
  let description = [{
    Apply various postprocessing passes such parallel loop fusion,
    buffer deallocation, general cleanup etc.
  }];
  let constructor = "mlir::astl::createPostprocessingPass()";
}

def AstlMapping : Pass<"astl-mapping", "ModuleOp"> {
  let summary = "Map operations to be ASTL compatible";
  let description = [{
    Apply collection of ASTL rewriting passes to map eligble operations
    into equivalent ASTL-compatible forms.
  }];
  let constructor = "mlir::astl::createAstlMappingPass()";
}

def AstlConversion : Pass<"astl-conversion", "func::FuncOp"> {
  let summary = "Convert operations to ASTL";
  let description = [{
    Convert all eligble operations into ASTL operations.
  }];
  let constructor = "mlir::astl::createAstlConversionPass()";
}

def AstlLowering : Pass<"astl-lowering", "func::FuncOp"> {
  let summary = "Lower ASTL operations";
  let description = [{
    Lower all ASTL operations into combination of operations from
    standard and local dialects.
  }];
  let options= [
    Option<"astlToLoops", "astl-to-loops",
           "bool", /*default=*/"0",
           "By default ASTL ops are lowered to XSMM. Lower ASTL to loops instead.">,
  ];
  let constructor = "mlir::astl::createAstlLoweringPass()";
}

def ConvertForAllToParallelOp : Pass<"convert-forall-to-parallel", 
                                     "func::FuncOp"> {
  let summary = "Convert scf.forall to scf.parallel";
  let description = [{
    Rewrite an scf.forall to scf.parallel after bufferization.
  }];
  let constructor = "mlir::astl::createConvertForAllToParallelOpPass()";
}

def GpuPipeline : Pass<"gpu-pipeline", "ModuleOp"> {
  let summary = "Lower all eligible operations into GPU compatible IR";
  let constructor = "mlir::astl::createGpuPipelinePass()";
  let options = [
    Option<"gpuBackend", "gpu", "std::string",
            /*default=*/"\"cuda\"",
           "Target GPU backend for lowering (cuda).">,
  ];
}

def GpuConversion : Pass<"gpu-conversion", "ModuleOp"> {
  let summary = "Convert operations to GPU";
  let description = [{
    Convert all eligble operations into generic GPU operations.
  }];
  let constructor = "mlir::astl::createGpuConversionPass()";
}

def GpuToCuda : Pass<"gpu-to-cuda", "ModuleOp"> {
  let summary = "Lower generic GPU operations to CUDA backend";
  let constructor = "mlir::astl::createGpuToCudaPass()";
  let options = [
    Option<"gpuTriple", "triple", "std::string",
            /*default=*/"\"nvptx64-nvidia-cuda\"",
           "GPU target triple.">,
    Option<"gpuChip", "chip", "std::string",
            /*default=*/"\"sm_35\"",
           "GPU target architecture.">,
    Option<"gpuFeatures", "features", "std::string",
            /*default=*/"\"+ptx60\"",
           "GPU target features.">,
  ];
}

def GpuToVulkan : Pass<"gpu-to-vulkan", "ModuleOp"> {
  let summary = "Lower generic GPU operations to Vulkan backend";
  let constructor = "mlir::astl::createGpuToVulkanPass()";
}

def LinalgDeGeneralize : Pass<"linalg-degeneralize-generic-ops", "func::FuncOp"> {
  let summary = "Convert generic ops into named ops";
  let constructor = "mlir::linalg::createLinalgDeGeneralizationPass()";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def DefaultPipeline : Pass<"default-pipeline", "ModuleOp"> {
  let summary = "The default compiler lowering pipeline";
  let description = [{
    A collection of passes that lower everything to MLIR LLVM IR.
  }];
  let constructor = "mlir::astl::createDefaultPipelinePass()";
  let options = [
    Option<"gpuBackend", "gpu", "std::string",
            /*default=*/"\"\"",
           "Optional target GPU backend.">,
  ];
}

def GPUToSPIRV : Pass<"gpu-to-spirv", "ModuleOp"> {
  let summary = "Convert GPU dialect to SPIR-V dialect";
  let description = [{
    This pass converts supported GPU device ops to SPIR-V ops. It does not
    handle GPU host ops.

    A `gpu.func` op can have parameters to pass in resources. But in SPIR-V
    entry functions cannot take parameters; they use descriptors to access
    resources. By default, parameters to a `gpu.func` op will be converted to
    global variables. These global variables will be assigned sequential binding
    numbers following their order in the original `gpu.func` op, starting from
    0, in set 0. One can attach `spirv.interface_var_abi` to those parameters
    to control the set and binding if wanted.
  }];
  let constructor = "mlir::astl::createConvertGPUToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"use64bitIndex", "use-64bit-index",
           "bool", /*default=*/"false",
           "Use 64-bit integers to convert index types">
  ];
}

def SetSPIRVCapabilities : Pass<"set-spirv-capabilities", "ModuleOp"> {
  let summary = "Set SPIR-V capabilities.";
  let constructor = "mlir::astl::createSetSPIRVCapabilitiesPass()";
  let options = [
    Option<"clientAPI", "client-api", "std::string",
            /*default=*/"\"vulkan\"",
           "The client API to use for capabilities">,
  ];
}

def SetSPIRVAbiAttribute : Pass<"set-spirv-abi-attr", "gpu::GPUModuleOp"> {
  let summary = "Set SPIR-V ABI attribute.";
  let constructor = "mlir::astl::createSetSPIRVAbiAttributePass()";
  let options = [
    Option<"clientAPI", "client-api", "std::string",
            /*default=*/"\"vulkan\"",
           "The client API to use for ABI attribute">,
  ];
}

def GpuVulkanAbi : Pass<"gpu-vulkan-abi", "ModuleOp"> {
  let summary = "Rewrite GPU kernels to comply with Vulkan ABI.";
  let description = [{
    This pass rewrites GPU kernels and the GPU function launches
    to be compatible with Vulkan calling convention.

    The rewrite is focused only on ensuring GPU kernel prototype
    compatibility with Vulkan ABI.Thus, it is assumed that the kernel
    operations have been already preserved in a separate SPIR-V module.
    The original GPU kernel logic is thrown away to make interface
    adaptation easier.
  }];
  let constructor = "mlir::astl::createGpuVulkanAbiPass()";
  let options = [
    Option<"use64bitIndex", "use-64bit-index",
           "bool", /*default=*/"false",
           "Use 64-bit integers to convert index types">
  ];
}

def DecomposeAggregatedOps : Pass<"decompose-aggregated-ops", "func::FuncOp"> {
  let summary = "Decompose aggregated operations.";
  let description = [{
    Decompose operations that implement the `AggregatedOpInterface`.
  }];
  let constructor = "mlir::astl::createDecomposeAggregatedOpsPass()";
}

def LinalgToGpu : Pass<"linalg-to-gpu", "func::FuncOp"> {
  let summary = "Convert linalg ops to be GPU compatible.";
  let description = [{
    Lower linalg to ops optimized for computation on GPU.
  }];
  let constructor = "mlir::astl::createLinalgToGpuPass()";
  let dependentDialects = ["linalg::LinalgDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect",
                           "gpu::GPUDialect",
                           "arith::ArithDialect"];
}

#endif // ASTL_DIALECT_ASTL_PASSES
