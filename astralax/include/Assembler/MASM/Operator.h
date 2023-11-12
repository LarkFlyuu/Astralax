#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Astralax/Dialect/AstlOps.h"
#include "common.h"
#include "Assembler/masm.h"
#include <iostream>
#include <string>

namespace astl {
namespace masm {

typedef struct {
  Tensors inputs;
  Tensor output;
  MasmOpList fused;
} CommonParam;

typedef struct {
  int kernel;
  int stride;
  int padding;
  Tensor output;
  Tensors inputs;
  MasmOpList fused;
} ConvParam;

using AstlOp = mlir::Operation;

class Operator {
public:
  virtual ~Operator() = default;

  virtual bool match(AstlOp *op) = 0;
  virtual std::string getSharding(std::string ofmSharding) = 0;
  virtual std::string getTiling(std::string ofmTiling) = 0;

  /* pattern match functions */
  bool matchConv(AstlOp *op, ConvParam &param);
  bool matchWinograd(AstlOp *op, ConvParam &param);
  bool matchDepthwise(AstlOp *op, ConvParam &param);
  bool matchMatMul(AstlOp *op, CommonParam &param);
  bool matchColwiseMatmul(AstlOp *op, CommonParam &param);
  bool matchFC(AstlOp *op, CommonParam &param);

  AstlOp* getDefiningOp(AstlOp *op, uint32_t index);
  std::vector<AstlOp*> getUserOps(AstlOp *op);

};

}  // namespace masm
}  // namespace astl