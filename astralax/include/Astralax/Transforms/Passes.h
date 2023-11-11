#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Astralax/Dialect/AstlOps.h"

using namespace mlir;
namespace astl {

std::unique_ptr<OperationPass<ModuleOp>> createQuantPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES

#include "Astralax/Transforms/Passes.h.inc"

} // namespace astl