//===- AstlUtils.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Dialect/Astl/AstlUtils.h"
#include "ASTL/Dialect/Astl/AstlOps.h"
#include "ASTL/IR/StructuredOpMatcher.h"
#include "ASTL/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace astl {
namespace utils {

// TODO: Remove this once convolutions stop using it
bool isMarkedWithAstl(linalg::LinalgOp linalgOp, const std::string &target) {
  return isa<linalg::GenericOp>(linalgOp) &&
         linalgOp.getLibraryCallName() == target;
}

// Return true if the linalg.generic an be mapped to a astl.brgemm in VNNI
// format.
bool isAstlVnniOp(linalg::GenericOp linalgOp, SmallVectorImpl<Value> *operands) {
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr r1, p4, p5, r2, r3;
  bindDims(linalgOp.getContext(), r1, r2, p4, p5, r3);
  auto blockingFactor =
      vnni::utils::getVnniBlockingFactor(linalgOp->getOperands()[0].getType());
  if (!blockingFactor)
    return false;
  SmallVector<AffineMap> mapList;
  mapList = infer(
      {{r1, p4, r3}, {r1, r3.floorDiv(*blockingFactor), p5, r2}, {p4, p5}});

  using namespace structured_match;
  // clang-format off
  auto matmulMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(EqualsTo(2)))
          .operation(NumRegions(EqualsTo(1)))
          .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                            mlir::utils::IteratorType::parallel,
                            mlir::utils::IteratorType::parallel,
                            mlir::utils::IteratorType::reduction,
                            mlir::utils::IteratorType::reduction})
          .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
          .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
          .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
          .region(MatchOne(0),
                  WithOpChain<arith::MulFOp, arith::AddFOp>(operands));
  // clang-format on
  return matmulMatcher.match(linalgOp);
}

LogicalResult splitAndReplaceFusedOp(astl::FusedBrgemmOp fusedBrgemmOp,
                                     PatternRewriter &rewriter) {
  if (!fusedBrgemmOp.hasBufferSemantics())
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(fusedBrgemmOp);

  Location loc = fusedBrgemmOp.getLoc();

  // Split the fused op into individual operations.
  auto ins = fusedBrgemmOp.getInputs();
  auto out = fusedBrgemmOp.getOutput();
  rewriter.create<astl::BrgemmOp>(loc, ValueRange{ins[0], ins[1], ins[2]}, out);

  switch (fusedBrgemmOp.getBinaryKind()) {
  case astl::FusedBinaryOpKind::ADD:
    rewriter.create<astl::AddOp>(loc, ValueRange{ins[3], out}, out);
    break;
  case astl::FusedBinaryOpKind::NONE:
    break;
  }

  switch (fusedBrgemmOp.getUnaryKind()) {
  case astl::FusedUnaryOpKind::RELU:
    rewriter.create<astl::ReluOp>(loc, out, out);
    break;
  case astl::FusedUnaryOpKind::NONE:
    break;
  }

  rewriter.eraseOp(fusedBrgemmOp);
  return success();
}

} // namespace utils
} // namespace astl
} // namespace mlir
