//===- AstlPasses.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTL_PASSES_H
#define ASTL_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace func {
class FuncOp;
class FuncDialect;
} // namespace func

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace math {
class MathDialect;
} // namespace math

namespace arith {
class ArithDialect;
} // namespace arith

namespace vector {
class VectorDialect;
} // namespace vector

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace scf {
class SCFDialect;
} // namespace scf

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace memref {
class MemRefDialect;
} // namespace memref

namespace xsmm {
class XsmmDialect;
} // namespace xsmm

namespace vnni {
class VNNIDialect;
} // namespace vnni

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace math {
class MathDialect;
} // namespace math

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace gpu {
class GPUModuleOp;
class GPUDialect;
} // namespace gpu

namespace spirv {
class SPIRVDialect;
} // namespace spirv

namespace astl {
class AstlDialect;

// The pass options are provided with default argument values to avoid
// API duplications and combinatorial explosion of flags.
// The values should be kept consistent with the default values of the pass
// declarations present in the corresponding TableGen file.
std::unique_ptr<OperationPass<func::FuncOp>> createConvertLinalgToAstlPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertLinalgToAstlPass(bool, bool, ArrayRef<int64_t> tiles = {});
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertAstlToLoopsPass(bool parallel = false);
std::unique_ptr<OperationPass<ModuleOp>> createConvertXsmmToFuncPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertCheckToLoopsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertVNNIToAstlPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertAstlToXsmmPass();
std::unique_ptr<OperationPass<ModuleOp>>
createTransformDialectInterpreterPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertPerfToLoopsPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertPerfToFuncPass();
std::unique_ptr<OperationPass<func::FuncOp>> createCombineAstlPass();
std::unique_ptr<OperationPass<ModuleOp>> createTransformDropSchedulePass();
std::unique_ptr<OperationPass<func::FuncOp>> createPackVNNIPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createPackMatmulPass(ArrayRef<int64_t> blockingFactors = {});
std::unique_ptr<OperationPass<func::FuncOp>>
createPackConv2DNchwFchwPass(ArrayRef<int64_t> blockingFactors = {});
std::unique_ptr<OperationPass<func::FuncOp>>
createPackConv2DNhwcHwcfPass(ArrayRef<int64_t> blockingFactors = {});
std::unique_ptr<OperationPass<func::FuncOp>>
createTileConsumerAndFuseProducersPass(ArrayRef<int64_t> tileSizes = {},
                                       int maxDepth = 5,
                                       bool startFromLastFusableConsumer = true,
                                       bool useForAll = true);
std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteConvToMatmulOrBrgemmPass();
std::unique_ptr<OperationPass<ModuleOp>>
createDefaultAstlPass(bool astlLoops = false, bool linalgLoops = false,
                     bool linalgToXsmm = false);
std::unique_ptr<OperationPass<func::FuncOp>>
createGeneralizeTensorPackAndUnPackPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPropagatePackUnPackPass();
std::unique_ptr<OperationPass<ModuleOp>> createConstantFoldPackPass();
std::unique_ptr<OperationPass<func::FuncOp>> createElementWiseFusionPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvInitSimplifyPass();
std::unique_ptr<OperationPass<ModuleOp>> createBufferizePass();
std::unique_ptr<OperationPass<func::FuncOp>> createCleanupPass();
std::unique_ptr<OperationPass<ModuleOp>> createTransformPass();
std::unique_ptr<OperationPass<ModuleOp>> createLocalDialectsLoweringPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPostprocessingPass();
std::unique_ptr<OperationPass<ModuleOp>> createAstlMappingPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAstlConversionPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createAstlLoweringPass(bool loops = false);
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertForAllToParallelOpPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createSimplifyAndCanonicalizePackPass();
std::unique_ptr<OperationPass<ModuleOp>>
createGpuPipelinePass(StringRef gpuBackend = "cuda");
std::unique_ptr<OperationPass<ModuleOp>> createGpuConversionPass();
std::unique_ptr<OperationPass<ModuleOp>>
createGpuToCudaPass(StringRef gpuTriple = "nvptx64-nvidia-cuda",
                    StringRef gpuChip = "sm_35",
                    StringRef gpuFeatures = "+ptx60");
std::unique_ptr<OperationPass<ModuleOp>> createGpuToVulkanPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteBatchMatmulToMatmulPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertMemRefToAstlPass();
std::unique_ptr<OperationPass<ModuleOp>>
createDefaultPipelinePass(StringRef gpuBackend = "");
std::unique_ptr<OperationPass<ModuleOp>>
createConvertGPUToSPIRVPass(bool mapMemorySpace = true);
std::unique_ptr<OperationPass<ModuleOp>>
createSetSPIRVCapabilitiesPass(StringRef api = "vulkan");
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createSetSPIRVAbiAttributePass(StringRef api = "vulkan");
std::unique_ptr<OperationPass<ModuleOp>>
createGpuVulkanAbiPass(bool use64bitIndex = false);
std::unique_ptr<OperationPass<func::FuncOp>> createConvertLinalgToXsmmPass();
std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeAggregatedOpsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgToGpuPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertPackOptimization();
// Testing passes.
void registerTestStructuralMatchers();
void registerTestForToForAllRewrite();

} // namespace astl
namespace linalg {
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgDeGeneralizationPass();
} // namespace linalg
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "ASTL/Passes.h.inc"

#endif // ASTL_PASSES_H
