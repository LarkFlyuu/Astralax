//===- GpuConversion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "ASTL/Dialect/Astl/AstlDialect.h"
#include "ASTL/Dialect/Transform/LinalgXTransformOps.h"
#include "ASTL/PassUtils.h"

#include <optional>

using namespace mlir;
using namespace mlir::astl;

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

namespace {

// Map and lower operations to generic GPU ops.
struct GpuConversion : public GpuConversionBase<GpuConversion>,
                       UtilityPassBase<ModuleOp> {
  GpuConversion() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<astl::AstlDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }

private:
  void constructPipeline() override {
    pm.clear();

    // Lower ASTL ops to GPU-compatible format.
    pm.addNestedPass<func::FuncOp>(createConvertAstlToLoopsPass(true));

    // First lower linalg using custom patterns then fall back to
    // the default lowering for any remaining ops.
    pm.addNestedPass<func::FuncOp>(linalg::createLinalgDeGeneralizationPass());
    pm.addNestedPass<func::FuncOp>(createLinalgToGpuPass());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());

    // Map loops into GPU kernels.
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());

    pm.addNestedPass<func::FuncOp>(createCleanupPass());

    // Create GPU kernels.
    pm.addPass(createGpuKernelOutliningPass());

    // Generic cleanup.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::astl::createGpuConversionPass() {
  return std::make_unique<GpuConversion>();
}
