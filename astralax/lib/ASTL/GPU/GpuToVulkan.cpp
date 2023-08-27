//===- GpuToVulkan.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
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

// Lower generic GPU ops to Vulkan backend.
struct GpuToVulkan : public GpuToVulkanBase<GpuToVulkan>,
                     UtilityPassBase<ModuleOp> {
  GpuToVulkan() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<spirv::SPIRVDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
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

#ifdef ASTL_VULKAN_ENABLE
    // Preprocess
    // Subviews are not supported by SPIRV ops
    pm.addPass(memref::createFoldMemRefAliasOpsPass());

    // Create SPIRV kernels.
    pm.addPass(astl::createSetSPIRVCapabilitiesPass());
    pm.addPass(astl::createSetSPIRVAbiAttributePass());
    pm.addPass(astl::createConvertGPUToSPIRVPass());
    pm.addNestedPass<spirv::ModuleOp>(
        spirv::createSPIRVLowerABIAttributesPass());
    pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());

    // Adapt GPU kernel to be compliant with Vulkan ABI.
    pm.addPass(astl::createGpuVulkanAbiPass());

    // Create Vulkan dispatch.
    pm.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
    pm.addNestedPass<func::FuncOp>(LLVM::createRequestCWrappersPass());

    // Cleanup IR.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
#endif // ASTL_VULKAN_ENABLE
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::astl::createGpuToVulkanPass() {
  return std::make_unique<GpuToVulkan>();
}
