//===- CombineAstl.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Dialect/Astl/AstlOps.h"
#include "ASTL/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

namespace {

struct CombineBrgemmAddAndRelu : public OpRewritePattern<astl::ReluOp> {
  using OpRewritePattern<astl::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(astl::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    if (!reluOp.hasTensorSemantics())
      return failure();
    Value operandRelu = reluOp.getInputs()[0];
    auto maybeAdd = operandRelu.getDefiningOp<astl::AddOp>();
    if (!maybeAdd)
      return failure();
    SmallVector<Value> brgemmOperands;
    Value addOperand;
    bool hasBrgemmProducer = false;
    for (Value operand : maybeAdd.getInputs()) {
      if (auto brgemmOp = operand.getDefiningOp<astl::BrgemmOp>()) {
        brgemmOperands = brgemmOp.getInputs();
        hasBrgemmProducer = true;
        continue;
      }
      addOperand = operand;
    }
    if (!hasBrgemmProducer)
      return failure();
    auto ctx = rewriter.getContext();
    auto unaryType =
        astl::FusedUnaryOpKindAttr::get(ctx, astl::FusedUnaryOpKind::RELU);
    auto binaryType =
        astl::FusedBinaryOpKindAttr::get(ctx, astl::FusedBinaryOpKind::ADD);
    rewriter.replaceOpWithNewOp<astl::FusedBrgemmOp>(
        reluOp, brgemmOperands, brgemmOperands.back().getType(), addOperand,
        unaryType, binaryType);
    return success();
  }
};

void populatePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineBrgemmAddAndRelu>(patterns.getContext());
}

struct CombineAstlOps : public CombineAstlOpsBase<CombineAstlOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::astl::createCombineAstlPass() {
  return std::make_unique<CombineAstlOps>();
}
