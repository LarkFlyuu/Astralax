#include "Operator.h"

using namespace mlir;

bool Operator::matchConv(AstlOp *op, ConvParam &param) {
  auto convOp = llvm::dyn_cast_or_null<linalg::Conv2DOp>(op);
  if (convOp == nullptr) return false;
  return true;
}

bool Operator::matchWinograd(AstlOp *op, ConvParam &param) {
  /* kernel=3x3, stride=1, padding=1 */
  auto convOp = llvm::dyn_cast_or_null<linalg::Conv2DOp>(op);
  if (convOp == nullptr) return false;
  return true;
}

bool Operator::matchDepthwise(AstlOp *op, ConvParam &param) {
  auto convOp = llvm::dyn_cast_or_null<linalg::Conv2DOp>(op);
  if (convOp == nullptr) return false;
  return true;
}

bool Operator::matchMatMul(AstlOp *op, CommonParam &param) {
  auto matmulOp = llvm::dyn_cast_or_null<linalg::MatmulOp>(op);
  if (matmulOp == nullptr) return false;
  return true;
}

bool Operator::matchColwiseMatmul(AstlOp *op, CommonParam &param) {
  auto matmulOp = llvm::dyn_cast_or_null<linalg::MatmulOp>(op);
  if (matmulOp == nullptr) return false;
  return true;
}

bool Operator::matchFC(AstlOp *op, CommonParam &param) {
  auto matmulOp = llvm::dyn_cast_or_null<linalg::MatmulOp>(op);
  if (matmulOp == nullptr) return false;
  return true;
}

AstlOp* Operator::getDefiningOp(AstlOp *op, uint32_t index) {
  MLIR_ASSERT(op->getNumOperands() > index,
              "index overflow: " + std::to_string(index));
  return op->getOperand(index).getDefiningOp();
}

std::vector<AstlOp*> Operator::getUserOps(AstlOp *op) {
  auto users = op->getResult(0).getUsers();
  return std::vector<AstlOp*>(users.begin(), users.end());
}

