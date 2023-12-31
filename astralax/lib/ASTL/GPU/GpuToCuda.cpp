//===- GpuToCuda.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "ASTL/Dialect/Astl/AstlDialect.h"
#include "ASTL/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::astl;

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

namespace {

// Lower generic GPU ops to CUDA backend.
struct GpuToCuda : public GpuToCudaBase<GpuToCuda>,
                   UtilityPassBase<gpu::GPUModuleOp> {
  GpuToCuda() = default;
  GpuToCuda(StringRef gpuTriple, StringRef gpuChip, StringRef gpuFeatures) {
    this->gpuTriple = gpuTriple.str();
    this->gpuChip = gpuChip.str();
    this->gpuFeatures = gpuFeatures.str();
  };

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<NVVM::NVVMDialect>();
    registry.insert<nvgpu::NVGPUDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    // Process GPU modules sequentially to avoid CUDA errors that
    // appear during parallel compilation.
    auto walkRes = module->walk([&](gpu::GPUModuleOp gpuModule) {
      if (failed(runPipeline(pm, gpuModule)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkRes.wasInterrupted())
      return signalPassFailure();
  }

private:
  void constructPipeline() override {
    pm.clear();

#ifdef ASTL_CUDA_ENABLE
    // Preprocess and lower standard ops.
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(arith::createArithExpandOpsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createConvertSCFToCFPass());

    // Create CUDA kernels.
    pm.addPass(createStripDebugInfoPass());
    pm.addPass(createLowerGpuOpsToNVVMOpsPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createGpuSerializeToCubinPass(gpuTriple, gpuChip, gpuFeatures));

    // Cleanup IR.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
#endif // ASTL_CUDA_ENABLE
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::astl::createGpuToCudaPass(StringRef gpuTriple, StringRef gpuChip,
                               StringRef gpuFeatures) {
  return std::make_unique<GpuToCuda>(gpuTriple, gpuChip, gpuFeatures);
}
