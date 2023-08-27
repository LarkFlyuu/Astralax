//===- astl-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "ASTL/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "ASTL/Dialect/Check/CheckDialect.h"
#include "ASTL/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "ASTL/Dialect/Perf/PerfDialect.h"
#include "ASTL/Dialect/Astl/BufferizableOpInterfaceImpl.h"
#include "ASTL/Dialect/Astl/AstlDialect.h"
#include "ASTL/Dialect/Transform/LinalgXTransformOps.h"
#include "ASTL/Dialect/Xsmm/XsmmDialect.h"
#include "ASTL/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  registerAstlCompilerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::astl::AstlDialect>();
  registry.insert<mlir::xsmm::XsmmDialect>();
  registry.insert<mlir::check::CheckDialect>();
  registry.insert<mlir::perf::PerfDialect>();
  mlir::linalgx::registerTransformDialectExtension(registry);
  mlir::check::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::perf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::astl::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::astl::registerTestStructuralMatchers();
  mlir::astl::registerTestForToForAllRewrite();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated.
  registerAllDialects(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  registerAllToLLVMIRTranslations(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ASTL optimizer driver\n", registry));
}
