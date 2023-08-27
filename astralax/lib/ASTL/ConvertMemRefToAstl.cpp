//===- LinalgMemRefToAstl.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Dialect/Astl/AstlOps.h"
#include "ASTL/Dialect/Astl/AstlTraits.h"
#include "ASTL/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

namespace {

// Convert a memref.copy to a astl.identity.
struct ConvertMemRefCopyToAstl : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto source = copyOp.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    if (sourceType.getRank() != 2 ||
        failed(OpTrait::astl::verifyUnitStrideInnerLoop(
            copyOp, /*emitDiagnostic=*/false))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<astl::IdentityOp>(copyOp, copyOp.getSource(),
                                                 copyOp.getTarget());
    return success();
  }
};

struct ConvertMemRefToAstl : public ConvertMemRefToAstlBase<ConvertMemRefToAstl> {
  ConvertMemRefToAstl() = default;
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMemRefCopyToAstl>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::astl::createConvertMemRefToAstlPass() {
  return std::make_unique<ConvertMemRefToAstl>();
}
