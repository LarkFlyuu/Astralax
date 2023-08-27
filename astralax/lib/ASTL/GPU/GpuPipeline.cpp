//===- GpuPipeline.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

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

#include <optional>

using namespace mlir;
using namespace mlir::astl;

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

namespace {

enum class GpuType {
  Cuda,
  Vulkan,
};

GpuType parseGpuOption(StringRef gpuStr) {
  auto type = llvm::StringSwitch<std::optional<GpuType>>(gpuStr)
                  .CaseLower("cuda", GpuType::Cuda)
                  .CaseLower("vulkan", GpuType::Vulkan)
                  .Default(std::nullopt);
  assert(type && "Unsupported GPU backend");

  return *type;
}

// GPU pipeline - map and lower operations to enable execution on a GPU.
struct GpuPipeline : public GpuPipelineBase<GpuPipeline>,
                     UtilityPassBase<ModuleOp> {
  GpuPipeline() = default;
  GpuPipeline(StringRef gpuBackend) { this->gpuBackend = gpuBackend.str(); }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<astl::AstlDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<NVVM::NVVMDialect>();
    registry.insert<nvgpu::NVGPUDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<spirv::SPIRVDialect>();
    bufferization::registerAllocationOpInterfaceExternalModels(registry);
    linalgx::registerTransformDialectExtension(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
    astl::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects to make the pipeline more robust with respect
    // to accepted input IR by preventing cryptic runtime crashes due to missing
    // dialect registrations.
    registerAllDialects(registry);
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

    // Typically there max number of threads per block is 1024.
    // For 2D tiles, use max 32 threads per dimension.
    constexpr int maxNumThreadsPerDim = 32;

    // Tile to split the kernel into threads and blocks
    pm.addPass(createCleanupPass());
    pm.addPass(createTileConsumerAndFuseProducersPass(
        /*tileSizes=*/{maxNumThreadsPerDim, maxNumThreadsPerDim}));
    pm.addPass(createCleanupPass());

    // Preprocess and bufferize as further conversion requires memref
    // abstraction.
    pm.addPass(createGeneralizeTensorPackAndUnPackPass());
    pm.addPass(createBufferizePass());
    pm.addPass(createConvertForAllToParallelOpPass());
    pm.addNestedPass<func::FuncOp>(createCleanupPass());

    // Convert to generic GPU ops.
    pm.addPass(createGpuConversionPass());

    // Lower GPU ops to the chosen GPU backend.
    switch (parseGpuOption(this->gpuBackend)) {
    case GpuType::Cuda:
      pm.addPass(createGpuToCudaPass());
      break;
    case GpuType::Vulkan:
      pm.addPass(createGpuToVulkanPass());
      break;
    }

    // Clean up after the GPU pipeline.
    // Use upstream passes directly instead of the cleanup pass as the GPU
    // kernel is at the LLVM dialect level which is not compatible with the
    // custom ASTL passes.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::astl::createGpuPipelinePass(StringRef gpuBackend) {
  return std::make_unique<GpuPipeline>(gpuBackend);
}
