//===- ConvertAstlToXsmm.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTL/Dialect/Astl/AstlOps.h"
#include "ASTL/Dialect/Astl/AstlUtils.h"
#include "ASTL/Dialect/Xsmm/XsmmEnum.h"
#include "ASTL/Dialect/Xsmm/XsmmOps.h"
#include "ASTL/Dialect/Xsmm/XsmmUtils.h"
#include "ASTL/Passes.h"
#include "ASTL/Transforms.h"
#include "ASTL/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "ASTL/Passes.h.inc"

#define DEBUG_TYPE "convert-astl-to-xsmm"

namespace {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static FailureOr<int64_t> getLeadingDim(Type type, size_t pos = 0) {
  // Not shaped type, the leading dimension is the single scalar.
  if (!isa<ShapedType>(type))
    return 1;
  MemRefType memref = type.cast<MemRefType>();
  // For 1d memref we cannot use the stride as leading dimension, but the
  // leading dimension is the dimension itself.
  if (memref.getRank() == 1)
    return memref.getShape()[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return failure();
  // fail if the strides are non-constant
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides[pos];
}

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
static void
computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                       ArrayRef<int64_t> lowerRankShape,
                       SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      assert(false && "bCast semantics for identity op broken");
  }
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

template <typename OpTy>
static FailureOr<DenseI64ArrayAttr>
getSizesAndLeadingDimsForGemmLikeOp(RewriterBase &rewriter, OpTy opTy) {
  assert(opTy.hasBufferSemantics() && "expects buffer semantics");

  bool isBrgemm = isa<astl::BrgemmOp>(opTy.getOperation()) ||
                  isa<astl::FusedBrgemmOp>(opTy.getOperation());

  auto memrefC = opTy.getOutputType();
  auto memrefA = opTy.getMemRefInputType(0);
  auto memrefB = opTy.getMemRefInputType(1);

  int64_t m = memrefC.getShape()[0];
  int64_t n = memrefC.getShape()[1];
  int64_t k = (isBrgemm) ? memrefA.getShape()[2] : memrefA.getShape()[1];

  auto ldaDim =
      (isBrgemm) ? getLeadingDim(memrefA, /*pos=*/1) : getLeadingDim(memrefA);
  if (failed(ldaDim)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot compute lda\n");
    return failure();
  }
  int64_t lda = *ldaDim;

  auto ldbDim =
      (isBrgemm) ? getLeadingDim(memrefB, /*pos=*/1) : getLeadingDim(memrefB);
  if (failed(ldbDim)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot compute ldb\n");
    return failure();
  }
  int64_t ldb = (vnni::utils::isInVnniLayout(memrefB))
                    ? *ldbDim / *vnni::utils::getVnniBlockingFactor(memrefB)
                    : *ldbDim;

  auto ldcDim = getLeadingDim(memrefC);
  if (failed(ldcDim)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot compute ldc\n");
    return failure();
  }
  int64_t ldc = *ldcDim;

  // If we are dealing with a BRGEMM we need to pass two extra dimensions:
  // - strideA and strideB that represent the stride between different GEMM
  // in BRGEMM.
  if (isBrgemm) {
    int64_t strideA = lda * m;
    int64_t strideB = ldb * k;
    return DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});
  }
  return DenseI64ArrayAttr::get(rewriter.getContext(),
                                ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
}

template <typename OpTy>
static ArrayAttr getGemmFlags(RewriterBase &rewriter, OpTy opTy) {
  auto memrefB = opTy.getMemRefInputType(1);
  xsmm::GemmFlagsAttr gemmFlag =
      (vnni::utils::isInVnniLayout(memrefB))
          ? xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                     xsmm::GemmFlags::VNNI_B)
          : xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                     xsmm::GemmFlags::NONE);
  return rewriter.getArrayAttr(gemmFlag);
}

struct ConvertAstlGemmOp : public OpRewritePattern<astl::GemmOp> {
  using OpRewritePattern<astl::GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(astl::GemmOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "xsmm expects buffer semantics");
    }

    Location loc = matmulOp.getLoc();
    auto dims = getSizesAndLeadingDimsForGemmLikeOp(rewriter, matmulOp);
    if (failed(dims)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Cannot compute leading dims or sizes");
    }

    auto dtype = xsmm::utils::getDataType(rewriter, matmulOp.getOutputType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, *dims, getGemmFlags(rewriter, matmulOp), dtype);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(matmulOp->getOperands().begin(),
                          matmulOp->getOperands().end());
    // Drop the aliasing output operand.
    invokeOperands.pop_back();
    rewriter.replaceOpWithNewOp<xsmm::GemmOp>(matmulOp, dtype, invokeOperands);
    return success();
  }
};

struct ConvertAstlBrgemmOp : public OpRewritePattern<astl::BrgemmOp> {
  using OpRewritePattern<astl::BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(astl::BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "xsmm expects buffer semantics");
    }

    Location loc = brgemmOp.getLoc();
    auto dims = getSizesAndLeadingDimsForGemmLikeOp(rewriter, brgemmOp);
    if (failed(dims)) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Cannot compute leading dims or sizes");
    }
    auto memrefB = brgemmOp.getMemRefInputType(1);
    int64_t batchSize = memrefB.getShape()[0];

    auto dtype = xsmm::utils::getDataType(rewriter, brgemmOp.getOutputType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, *dims, getGemmFlags(rewriter, brgemmOp), dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    // Drop the aliasing output operand.
    invokeOperands.pop_back();
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::BrgemmOp>(brgemmOp, dtype,
                                                invokeOperands);
    return success();
  }
};

// Forward decl.
static xsmm::BinaryFlags getBinaryBCast(MemRefType operandType,
                                        MemRefType outputType,
                                        size_t operandNumber);

struct ConvertAstlFusedBrgemmOp : public OpRewritePattern<astl::FusedBrgemmOp> {
  using OpRewritePattern<astl::FusedBrgemmOp>::OpRewritePattern;

  ArrayAttr getUnaryFlags(RewriterBase &rewriter,
                          astl::FusedBrgemmOp brgemmOp) const {
    return rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
  }

  ArrayAttr getBinaryFlags(RewriterBase &rewriter,
                           astl::FusedBrgemmOp brgemmOp) const {
    auto binaryInputType =
        brgemmOp.getBiasOperand().getType().cast<MemRefType>();
    auto outputType = brgemmOp.getOutputType();
    return rewriter.getArrayAttr(xsmm::BinaryFlagsAttr::get(
        rewriter.getContext(),
        getBinaryBCast(binaryInputType, outputType, /*operandNumber=*/0)));
  }

  xsmm::BinaryKindAttr getBinaryKind(RewriterBase &rewriter,
                                     astl::FusedBrgemmOp brgemmOp) const {
    auto kind = brgemmOp.getBinaryKind();
    auto ctx = rewriter.getContext();
    if (kind == astl::FusedBinaryOpKind::NONE)
      return xsmm::BinaryKindAttr::get(ctx, xsmm::BinaryKind::NONE);
    if (kind == astl::FusedBinaryOpKind::ADD)
      return xsmm::BinaryKindAttr::get(ctx, xsmm::BinaryKind::ADD);
    assert(false && "invalid binary kind");
  }

  xsmm::UnaryKindAttr getUnaryKind(RewriterBase &rewriter,
                                   astl::FusedBrgemmOp brgemmOp) const {
    auto kind = brgemmOp.getUnaryKind();
    auto ctx = rewriter.getContext();
    if (kind == astl::FusedUnaryOpKind::NONE)
      return xsmm::UnaryKindAttr::get(ctx, xsmm::UnaryKind::NONE);
    if (kind == astl::FusedUnaryOpKind::RELU)
      return xsmm::UnaryKindAttr::get(ctx, xsmm::UnaryKind::RELU);
    assert(false && "invalid unary kind");
  }

  LogicalResult matchAndRewrite(astl::FusedBrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "xsmm expects buffer semantics");
    }

    Location loc = brgemmOp.getLoc();

    // Current limitation in LIBXSMM.
    // See: https://github.com/libxsmm/libxsmm/issues/766
    // Split into separate operations if bcast_col_in0 is not present when add
    // is fused.
    // TODO: remove the split once LIBXSMM is fixed.
    auto isBiasAdd = brgemmOp.getBinaryKind() == astl::FusedBinaryOpKind::ADD;
    auto binaryFlag = getBinaryFlags(rewriter, brgemmOp)[0]
                          .cast<xsmm::BinaryFlagsAttr>()
                          .getValue();
    auto isBitSet = static_cast<uint64_t>(binaryFlag) &
                    static_cast<uint64_t>(xsmm::BinaryFlags::BCAST_COL_IN_0);
    if (isBiasAdd && !isBitSet)
      return astl::utils::splitAndReplaceFusedOp(brgemmOp, rewriter);

    auto dims = getSizesAndLeadingDimsForGemmLikeOp(rewriter, brgemmOp);
    if (failed(dims)) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Cannot compute leading dims or sizes");
    }
    auto memrefB = brgemmOp.getMemRefInputType(1);
    int64_t batchSize = memrefB.getShape()[0];

    auto dtype = xsmm::utils::getDataType(rewriter, brgemmOp.getOutputType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    Value dispatched = rewriter.create<xsmm::FusedBrgemmDispatchOp>(
        loc, integer64, *dims, getBinaryKind(rewriter, brgemmOp),
        getUnaryKind(rewriter, brgemmOp), getGemmFlags(rewriter, brgemmOp),
        getUnaryFlags(rewriter, brgemmOp), getBinaryFlags(rewriter, brgemmOp),
        dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    // Drop the aliasing output operand.
    invokeOperands.pop_back();
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::FusedBrgemmOp>(brgemmOp, dtype,
                                                     invokeOperands);
    return success();
  }
};

// ======================================== Unary/Binary Ops Lowering

template <class OpKind, class OpFlags, class KindAttr, class FlagsAttr,
          class DispatchOp, class Op>
static LogicalResult lowerASTLtoXSMM(astl::AstlOp op, PatternRewriter &rewriter,
                                    Type elmTy, OpKind kind, OpFlags flags,
                                    ArrayRef<int64_t> dims) {
  auto *ctx = op.getContext();
  auto loc = op.getLoc();

  KindAttr kindAttr = KindAttr::get(ctx, kind);
  DenseI64ArrayAttr dimsAttr =
      DenseI64ArrayAttr::get(rewriter.getContext(), dims);
  auto flagsAttr = FlagsAttr::get(ctx, flags);
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  xsmm::DataTypeAttr dtype =
      xsmm::utils::getDataType(rewriter, op.getOutputType());

  Value dispatched =
      rewriter.create<DispatchOp>(loc, integer64, kindAttr, dimsAttr,
                                  rewriter.getArrayAttr(flagsAttr), dtype);

  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(op.getInputs().begin(), op.getInputs().end());
  invokeOperands.push_back(op.getOutput());

  rewriter.replaceOpWithNewOp<Op>(op, dtype, kindAttr, invokeOperands);
  return success();
}

static xsmm::UnaryFlags getUnaryFlags(astl::AstlOp astlOp) {
  Type inputType = astlOp.getInputs()[0].getType();

  // There are multiple ways to define a scalar.  f32, memref<1x1xf32> or
  // memref<f32>. Handle f32, and memref<1x1xf32>. memref<f32> is not allowed
  // in astl at the moment.
  if (!inputType.isa<ShapedType>())
    return xsmm::UnaryFlags::BCAST_SCALAR;
  ArrayRef<int64_t> shapeInput = inputType.cast<ShapedType>().getShape();
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(shapeInput, isOne))
    return xsmm::UnaryFlags::BCAST_SCALAR;

  Type outputType = astlOp.getOutputType();
  ArrayRef<int64_t> shapeOutput = outputType.cast<ShapedType>().getShape();
  assert(shapeOutput.size() >= shapeInput.size() &&
         "output rank must be >= input rank");
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
  assert(shapeOutput.size() == bShapeInput.size());
  shapeInput = bShapeInput;

  if (shapeInput[1] == 1 && shapeOutput[1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  if (shapeInput[0] == shapeOutput[0] && shapeInput[1] == shapeOutput[1])
    return xsmm::UnaryFlags::NONE;

  assert(false && "failed to get bCast for astl op");
}

static LogicalResult lowerUnaryASTLtoXSMM(PatternRewriter &rewriter,
                                         Operation *op, xsmm::UnaryKind kind) {
  auto astlOp = cast<astl::AstlOp>(op);
  if (!astlOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(astlOp, "xsmm expects a memref type");

  MemRefType outputMemRef = astlOp.getOutputType();
  int64_t m = outputMemRef.getShape()[0];
  int64_t n = outputMemRef.getShape()[1];
  auto ldo = getLeadingDim(outputMemRef);
  if (failed(ldo))
    return rewriter.notifyMatchFailure(astlOp, "cannot compute ldo");
  auto ldi = getLeadingDim(astlOp.getInputs()[0].getType());
  if (failed(ldi))
    return rewriter.notifyMatchFailure(astlOp, "cannot compute ldi");
  xsmm::UnaryFlags flags = getUnaryFlags(astlOp);
  return lowerASTLtoXSMM<xsmm::UnaryKind, xsmm::UnaryFlags, xsmm::UnaryKindAttr,
                        xsmm::UnaryFlagsAttr, xsmm::UnaryDispatchOp,
                        xsmm::UnaryOp>(astlOp, rewriter,
                                       outputMemRef.getElementType(), kind,
                                       flags, {m, n, *ldi, *ldo});
}

static LogicalResult lowerBinaryASTLtoXSMM(Operation *op,
                                          PatternRewriter &rewriter, Type elmTy,
                                          xsmm::BinaryKind kind,
                                          xsmm::BinaryFlags flags,
                                          ArrayRef<int64_t> dims) {
  assert(isa<astl::AstlOp>(op));
  auto astlOp = cast<astl::AstlOp>(op);
  if (!astlOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(astlOp, "xsmm expects a memref type");
  return lowerASTLtoXSMM<xsmm::BinaryKind, xsmm::BinaryFlags,
                        xsmm::BinaryKindAttr, xsmm::BinaryFlagsAttr,
                        xsmm::BinaryDispatchOp, xsmm::BinaryOp>(
      astlOp, rewriter, elmTy, kind, flags, dims);
}

struct ConvertAstlIdentityOp : public OpRewritePattern<astl::IdentityOp> {
  using OpRewritePattern<astl::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(astl::IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryASTLtoXSMM(rewriter, identityOp, xsmm::UnaryKind::IDENTITY);
  }
};

struct ConvertAstlReluOp : public OpRewritePattern<astl::ReluOp> {
  using OpRewritePattern<astl::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(astl::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryASTLtoXSMM(rewriter, reluOp, xsmm::UnaryKind::RELU);
  }
};

struct ConvertAstlZeroOp : public OpRewritePattern<astl::ZeroOp> {
  using OpRewritePattern<astl::ZeroOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(astl::ZeroOp zeroOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryASTLtoXSMM(rewriter, zeroOp, xsmm::UnaryKind::ZERO);
  }
};

// Given the operand type and the output type return the broadcast
// to use in the XSMM call.
static xsmm::BinaryFlags getBinaryBCast(MemRefType operandType,
                                        MemRefType outputType,
                                        size_t operandNumber) {

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };

  auto shapeOutput = outputType.getShape();
  auto shapeOperand = operandType.getShape();
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);

  auto getBCastEnum = [](BCastType bCastType,
                         std::optional<unsigned> operand) -> xsmm::BinaryFlags {
    switch (bCastType) {
    case BCastType::NONE:
      return xsmm::BinaryFlags::NONE;
    case BCastType::SCALAR:
      assert(operand != std::nullopt && "Require operand idx");
      assert(*operand == 1 || *operand == 0 && "Expect idx to be 1 or 0");
      if (*operand == 0)
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
    case BCastType::ROW:
      assert(operand != std::nullopt && "Require operand idx");
      assert(*operand == 1 || *operand == 0 && "Expect idx to be 1 or 0");
      if (*operand == 0)
        return xsmm::BinaryFlags::BCAST_ROW_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_ROW_IN_1;
    case BCastType::COL:
      assert(operand != std::nullopt && "Require operand idx");
      assert(*operand == 1 || *operand == 0 && "Expect idx to be 1 or 0");
      if (*operand == 0)
        return xsmm::BinaryFlags::BCAST_COL_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_COL_IN_1;
    }
    assert(false && "unrechable");
  };

  // Multiple way to define a scalar. Check if the memref
  // is a scalar here.
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(bOperandShape, isOne))
    return getBCastEnum(BCastType::SCALAR, operandNumber);

  if (bOperandShape[1] == 1 && shapeOutput[1] > 1)
    return getBCastEnum(BCastType::ROW, operandNumber);
  if (bOperandShape[0] == 1 && shapeOutput[0] > 1)
    return getBCastEnum(BCastType::COL, operandNumber);
  if (bOperandShape == shapeOutput)
    return getBCastEnum(BCastType::NONE, operandNumber);

  assert(false && "failed to get bCast for astl.add");
}

struct ConvertAstlAddOp : public OpRewritePattern<astl::AddOp> {
  using OpRewritePattern<astl::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(astl::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (!addOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(addOp, "xsmm expects a memref type");

    MemRefType outputMemRef = addOp.getOutputType();
    assert(outputMemRef.getRank() == 2 && "expect rank 2 for ASTL ops");

    int64_t m = outputMemRef.getShape()[0];
    int64_t n = outputMemRef.getShape()[1];

    auto lhsMemRef = addOp.getInputs()[0].getType().cast<MemRefType>();
    auto rhsMemRef = addOp.getInputs()[1].getType().cast<MemRefType>();

    auto ldiLhsDim = getLeadingDim(lhsMemRef);
    if (failed(ldiLhsDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldi on lhs");
    int64_t ldiLhs = *ldiLhsDim;

    auto ldiRhsDim = getLeadingDim(rhsMemRef);
    if (failed(ldiRhsDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldi on rhs");
    int64_t ldiRhs = *ldiRhsDim;

    auto ldoDim = getLeadingDim(outputMemRef);
    if (failed(ldoDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldo");
    int64_t ldo = *ldoDim;

    xsmm::BinaryFlags bCastOnLhs = getBinaryBCast(lhsMemRef, outputMemRef, 0);
    xsmm::BinaryFlags bCastOnRhs = getBinaryBCast(rhsMemRef, outputMemRef, 1);

    LLVM_DEBUG(llvm::dbgs() << stringifyBinaryFlags(bCastOnLhs) << "\n");
    LLVM_DEBUG(llvm::dbgs() << stringifyBinaryFlags(bCastOnRhs) << "\n");

    xsmm::BinaryFlags bCast =
        (bCastOnLhs != xsmm::BinaryFlags::NONE) ? bCastOnLhs : bCastOnRhs;

    return lowerBinaryASTLtoXSMM(addOp, rewriter, outputMemRef.getElementType(),
                                xsmm::BinaryKind::ADD, bCast,
                                {m, n, ldiLhs, ldiRhs, ldo});
  }
};

struct ConvertAstlToXsmm : public ConvertAstlToXsmmBase<ConvertAstlToXsmm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    astl::populateAstlToXsmmPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

void mlir::astl::populateAstlToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertAstlIdentityOp, ConvertAstlReluOp, ConvertAstlZeroOp,
               ConvertAstlAddOp, ConvertAstlGemmOp, ConvertAstlBrgemmOp,
               ConvertAstlFusedBrgemmOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::astl::createConvertAstlToXsmmPass() {
  return std::make_unique<ConvertAstlToXsmm>();
}
