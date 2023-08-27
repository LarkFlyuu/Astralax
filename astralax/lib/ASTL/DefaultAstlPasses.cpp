//===- DefaultAstlPasses.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "ASTL/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "ASTL/Dialect/Check/CheckDialect.h"
#include "ASTL/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "ASTL/Dialect/Perf/PerfDialect.h"
#include "ASTL/Dialect/Astl/BufferizableOpInterfaceImpl.h"
#include "ASTL/Dialect/Astl/AstlDialect.h"
#include "ASTL/Dialect/Transform/LinalgXTransformOps.h"
#include "ASTL/Dialect/Xsmm/XsmmDialect.h"
#include "ASTL/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::astl;

// Extra command line options to control the default ASTL utility passes.
llvm::cl::opt<bool>
    defPackMatmul("def-pack-matmul",
                  llvm::cl::desc("Default pipeline - pack matmul"),
                  llvm::cl::init(true));

llvm::cl::opt<bool> defPipePack("def-pack",
                                llvm::cl::desc("Default pipeline - packing"),
                                llvm::cl::init(true));

llvm::cl::opt<bool>
    disableDefPipe("disable-def-pipe",
                   llvm::cl::desc("Disable default pipeline execution"),
                   llvm::cl::init(false));

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

namespace {

// A general cleanup pass that performs general IR normalization and
// generic optimizations without any lowering or any logical changes.
// Commonly applied after other major passes.
struct CleanupPass : public CleanupBase<CleanupPass>,
                     UtilityPassBase<func::FuncOp> {
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

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

// Apply any present transforms and remove transform blocks afterwards.
struct TransformPass : public TransformBase<TransformPass>,
                       UtilityPassBase<ModuleOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
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

    // Run all transforms and clean them up afterwards.
    pm.addPass(createTransformDialectInterpreterPass());
    pm.addPass(createTransformDropSchedulePass());
  }
};

// Lower all local dialects (XSMM, check etc.) to standard dialects
// and function calls.
struct LocalDialectsLoweringPass
    : public LocalDialectsLoweringBase<LocalDialectsLoweringPass>,
      UtilityPassBase<ModuleOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                func::FuncDialect,
                memref::MemRefDialect,
                check::CheckDialect,
                perf::PerfDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                xsmm::XsmmDialect,
                LLVM::LLVMDialect>();
    // clang-format on
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

    pm.addNestedPass<func::FuncOp>(createConvertCheckToLoopsPass());
    pm.addNestedPass<func::FuncOp>(createConvertPerfToLoopsPass());

    // Note that LICM should be performed before any function calls are
    // generated
    // to ensure that ops which map directly to functions also get moved outside
    // of loops, if possible. This approach assumes that the function calls do
    // not have any side effects and can be safely moved outside of loop body.
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    // Run cleanup after LICM to allow CSE to eliminate common operations now
    // that they are hoisted out of loops.
    pm.addNestedPass<func::FuncOp>(createCleanupPass());

    pm.addPass(createConvertXsmmToFuncPass());
    pm.addPass(createConvertPerfToFuncPass());
  }
};

// Apply various postprocessing passes such as LICM, parallel loop fusion,
// buffer deallocation, general cleanup etc.
struct PostprocessingPass : public PostprocessingBase<PostprocessingPass>,
                            UtilityPassBase<func::FuncOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<bufferization::BufferizationDialect,
                memref::MemRefDialect,
                scf::SCFDialect>();
    // clang-format on
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
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

    // Postprocess buffers.
    pm.addPass(bufferization::createBufferHoistingPass());

    // Run general cleanup to normalize IR.
    pm.addPass(createCleanupPass());
  }
};

// Apply collection of high-level passes that map operations to
// ASTL-compatible forms.
struct AstlMappingPass : public AstlMappingBase<AstlMappingPass>,
                        UtilityPassBase<ModuleOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<linalg::LinalgDialect,
                memref::MemRefDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                astl::AstlDialect>();
    // clang-format on
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module))) {
      llvm::dbgs() << "Failed astl mapping\n";
      return signalPassFailure();
    }
  }

private:
  void constructPipeline() override {
    pm.clear();

    // Preprocess convolutions.
    pm.addPass(createConvInitSimplifyPass());
    pm.addPass(createCleanupPass());
    if (defPipePack) {
      pm.addPass(createPackConv2DNhwcHwcfPass({32, 32}));
      pm.addPass(createPackConv2DNchwFchwPass({32, 32}));
      pm.addPass(createRewriteConvToMatmulOrBrgemmPass());

      // Convert ops to packed layouts.
      if (defPackMatmul)
        pm.addPass(createPackMatmulPass({32, 32, 32}));
      pm.addPass(createPackVNNIPass());
    }

    // Postprocess packing.
    // Run only canonicalizer at this stage as full cleanup (mostly CSE) can
    // mess up tensor producer-consumer chains used for analysis in the
    // following passes.
    pm.addPass(createPropagatePackUnPackPass());
    pm.addPass(createConstantFoldPackPass());
    pm.addPass(createSimplifyAndCanonicalizePackPass());

    // Looks like we want to agressively remove tensor.empty before fusion.
    // See: `test/Passes/tile-and-fuse-with-cse.mlir`.
    pm.addPass(createCleanupPass());
    pm.addPass(createTileConsumerAndFuseProducersPass());
    pm.addPass(createCleanupPass());
    pm.addPass(createConvertPackOptimization());

    // Generalize tensor.pack and tensor.unpack.
    pm.addPass(createGeneralizeTensorPackAndUnPackPass());

    // Final clenaup after all the mapping.
    pm.addPass(createCleanupPass());
  }
};

// Convert all matching operations to ASTL.
struct AstlConversionPass : public AstlConversionBase<AstlConversionPass>,
                           UtilityPassBase<func::FuncOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<linalg::LinalgDialect,
                scf::SCFDialect,
                astl::AstlDialect>();
    // clang-format on
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

    // Convert all higher level dialects to ASTL.
    pm.addPass(createConvertLinalgToAstlPass());
    pm.addPass(createCombineAstlPass());
  }
};

// Lower ASTL to into combination of standard and local dialects.
struct AstlLoweringPass : public AstlLoweringBase<AstlLoweringPass>,
                         UtilityPassBase<func::FuncOp> {
  AstlLoweringPass() : AstlLoweringPass(false){};
  AstlLoweringPass(bool astlToLoops) { this->astlToLoops = astlToLoops; };

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<xsmm::XsmmDialect,
                scf::SCFDialect,
                memref::MemRefDialect,
                astl::AstlDialect>();
    // clang-format on
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

    // Lower all ASTL ops.
    if (astlToLoops)
      pm.addPass(createConvertAstlToLoopsPass());
    else {
      // Memref to astl conversion patterns.
      pm.addPass(createConvertMemRefToAstlPass());
      // astl to Xsmm conversion patterns.
      pm.addPass(createConvertAstlToXsmmPass());
    }
  }
};

// The default pipeline for ASTL.
struct DefaultAstlPasses : public DefaultAstlPassesBase<DefaultAstlPasses>,
                          UtilityPassBase<ModuleOp> {
  DefaultAstlPasses() : DefaultAstlPasses(false, false, false){};
  DefaultAstlPasses(bool astlToLoops, bool linalgToLoops, bool linalgToXsmm) {
    this->astlToLoops = astlToLoops;
    this->linalgToLoops = linalgToLoops;
    this->linalgToXsmm = linalgToXsmm;
  };

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom ASTL dialects.
    registry.insert<astl::AstlDialect>();
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
    bufferization::registerAllocationOpInterfaceExternalModels(registry);
    linalgx::registerTransformDialectExtension(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
    astl::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default ASTL passes may contain any
    // combination of other passes.
    registerAllDialects(registry);
  }

  void runOnOperation() override {
    if (!disableDefPipe) {
      auto module = getOperation();

      // Initialize the pipeline if needed.
      // Otherwise, just run the cached one.
      if (pm.empty())
        constructPipeline();

      if (failed(runPipeline(pm, module)))
        return signalPassFailure();
    }
  }

private:
  void constructPipeline() override {
    pm.clear();

    // Default pipeline does not support transforms yet
    pm.addPass(createTransformDropSchedulePass());

    if (linalgToLoops) {
      // Lower linalg directly to loops.
      // Skip all ASTL transformations.
      // Generalize tensor.pack and tensor.unpack.
      pm.addPass(createGeneralizeTensorPackAndUnPackPass());
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOpsPass());
      pm.addPass(createBufferizePass());
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
      pm.addNestedPass<func::FuncOp>(createCleanupPass());
    } else if (linalgToXsmm) {
      // tile and fuse.
      pm.addPass(createTileConsumerAndFuseProducersPass());
      pm.addPass(createCleanupPass());

      // tensor->memref.
      pm.addPass(createBufferizePass());

      // linalg to xsmm.
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToXsmmPass());
      pm.addNestedPass<func::FuncOp>(createCleanupPass());

    } else {
      // Convert linalg.batch_matmul to linalg.matmul.
      pm.addPass(createRewriteBatchMatmulToMatmulPass());

      // Applies a set of passes at the linalg level to fuse and pack.
      pm.addPass(createAstlMappingPass());
      pm.addNestedPass<func::FuncOp>(createCleanupPass());

      // Decompose Aggregated operations. These ops currently do not
      // bufferize. Once this is possible we can move this pass after
      // bufferization.
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOpsPass());

      // Lower linalg operations to ASTL.
      pm.addNestedPass<func::FuncOp>(createAstlConversionPass());
      pm.addNestedPass<func::FuncOp>(createCleanupPass());

      // Bufferize: tensor->memref.
      pm.addPass(createBufferizePass());

      // Lower all ASTL operations.
      pm.addNestedPass<func::FuncOp>(createAstlLoweringPass(astlToLoops));
      pm.addNestedPass<func::FuncOp>(createCleanupPass());
    }

    // Convert forAll to parallel loops should run after bufferization
    // as scf.parallel does not handle tensor.
    pm.addPass(createConvertForAllToParallelOpPass());

    // Covert all local ASTL-related dialects.
    pm.addPass(createLocalDialectsLoweringPass());

    // Clean up after the default pipeline.
    pm.addNestedPass<func::FuncOp>(createPostprocessingPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::astl::createCleanupPass() {
  return std::make_unique<CleanupPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::astl::createTransformPass() {
  return std::make_unique<TransformPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::astl::createLocalDialectsLoweringPass() {
  return std::make_unique<LocalDialectsLoweringPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::astl::createPostprocessingPass() {
  return std::make_unique<PostprocessingPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::astl::createAstlMappingPass() {
  return std::make_unique<AstlMappingPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::astl::createAstlConversionPass() {
  return std::make_unique<AstlConversionPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::astl::createAstlLoweringPass(bool loops) {
  return std::make_unique<AstlLoweringPass>(loops);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::astl::createDefaultAstlPass(bool astlLoops, bool linalgLoops,
                                bool linalgToXsmm) {
  return std::make_unique<DefaultAstlPasses>(astlLoops, linalgLoops,
                                            linalgToXsmm);
}
