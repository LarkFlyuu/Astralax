//===- LinalgConvertToAstl.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Dialect/Astl/AstlOps.h"
#include "ASTL/Dialect/Astl/AstlUtils.h"
#include "ASTL/MatcherUtils.h"
#include "ASTL/Passes.h"
#include "ASTL/TransformUtils.h"
#include "ASTL/Transforms.h"
#include "ASTL/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

#define DEBUG_TYPE "linalg-convert-to-astl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// Convert a linalg.generic to a astl operation.
struct ConvertGenericOpToAstl : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult rewriteToAstlOp(linalg::GenericOp linalgOp,
                               PatternRewriter &rewriter) const {
    SmallVector<Value> operands;
    if (structured_match::utils::isTwoDZeroOp(linalgOp, &operands)) {
      assert(operands.size() == 1 && "astl.zero expects one operand");
      rewriter.replaceOpWithNewOp<astl::ZeroOp>(linalgOp, operands[0],
                                               operands[0].getType());
      return success();
    }

    if (structured_match::utils::isTwoDIdentityOp(linalgOp, &operands)) {
      assert(operands.size() == 2 && "astl.identity expects two operands");
      rewriter.replaceOpWithNewOp<astl::IdentityOp>(linalgOp, operands[0],
                                                   operands[1].getType());
      return success();
    }

    if (structured_match::utils::isTwoDReluOp(linalgOp, &operands)) {
      assert(operands.size() == 2 && "astl.relu expects two operands");
      rewriter.replaceOpWithNewOp<astl::ReluOp>(linalgOp, operands[0],
                                               operands[1].getType());
      return success();
    }

    if (structured_match::utils::isTwoDAddOp(linalgOp, &operands)) {
      assert(operands.size() == 3 && "astl.add expects three operands");
      rewriter.replaceOpWithNewOp<astl::AddOp>(
          linalgOp, ValueRange{operands[0], operands[1]},
          operands[2].getType());
      return success();
    }

    if (structured_match::utils::isTwoDBiasReluOp(linalgOp, &operands)) {
      assert(operands.size() == 3 && "astl.add+astl.relu expects three operands");
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(linalgOp);
      auto add = rewriter.create<astl::AddOp>(
          linalgOp.getLoc(), ValueRange{operands[0], operands[1]},
          operands[2].getType());
      rewriter.replaceOpWithNewOp<astl::ReluOp>(linalgOp, add.getResult(0),
                                               operands[2].getType());
      return success();
    }

    return rewriter.notifyMatchFailure(
        linalgOp, "failed to match to a known astl operation");
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Expect tensor type when mapping to astl");
    }
    if (linalgOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Expect static shape when mapping to astl");
    }
    return rewriteToAstlOp(linalgOp, rewriter);
  }
};

// Convert a linalg.batch_reduce_matmul to a astl.brgemm.
struct ConvertBrgemmToAstl
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brMatmulOp,
                                PatternRewriter &rewriter) const override {
    if (!brMatmulOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          brMatmulOp, "Expect tensor type when mapping to astl");
    }
    if (brMatmulOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          brMatmulOp, "Expect static shape when mapping to astl");
    }
    SmallVector<Value> inputs = brMatmulOp.getDpsInputOperands();
    inputs.push_back(brMatmulOp.getDpsInitOperands()[0]->get());
    SmallVector<Value> outputs = brMatmulOp.getDpsInitOperands();
    rewriter.replaceOpWithNewOp<astl::BrgemmOp>(brMatmulOp, inputs,
                                               outputs[0].getType());
    return success();
  }
};

// Convert a linalg.matmul to a astl.matmul.
struct ConvertMatmulToAstl : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect tensor type when mapping to astl");
    }
    if (matmulOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect static shape when mapping to astl");
    }
    SmallVector<Value> inputs = matmulOp.getDpsInputOperands();
    inputs.push_back(matmulOp.getDpsInitOperands()[0]->get());
    SmallVector<Value> outputs = matmulOp.getDpsInitOperands();
    rewriter.replaceOpWithNewOp<astl::GemmOp>(matmulOp, inputs,
                                             outputs[0].getType());
    return success();
  }
};

// Convert a linalg.fill to a astl.zero.
struct ConvertFillToAstl : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!fillOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          fillOp, "Expect tensor type when mapping to astl");
    }
    if (fillOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          fillOp, "Expect static shape when mapping to astl");
    }

    auto inputs = fillOp.getInputs();
    if (!utils::isZeroTensor(inputs[0]))
      return rewriter.notifyMatchFailure(fillOp, "Unsupported fill type");

    auto output = fillOp.getOutputs()[0];
    auto outputRank = output.getType().cast<ShapedType>().getRank();
    if (outputRank != 2)
      return rewriter.notifyMatchFailure(fillOp, "Expect output rank 2");

    rewriter.replaceOpWithNewOp<astl::ZeroOp>(fillOp, output, output.getType());
    return success();
  }
};

struct ConvertLinalgToAstl : public ConvertLinalgToAstlBase<ConvertLinalgToAstl> {
  ConvertLinalgToAstl() = default;
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    astl::populateConvertLinalgToAstlPatterns(patterns);
    memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace

void mlir::astl::populateConvertLinalgToAstlPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertGenericOpToAstl,
               ConvertBrgemmToAstl,
               ConvertMatmulToAstl,
               ConvertFillToAstl>(patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::astl::createConvertLinalgToAstlPass() {
  return std::make_unique<ConvertLinalgToAstl>();
}
