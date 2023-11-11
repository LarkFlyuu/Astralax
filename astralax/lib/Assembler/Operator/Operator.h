#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Astralax/Dialect/AstlOps.h"
#include <iostream>
#include <string>

#define MLIR_ASSERT(x, y)                                              \
  {                                                                    \
    const std::string &funcName = y;                                   \
    if (!(x)) {                                                        \
      llvm::report_fatal_error(llvm::StringRef("Error: " + funcName)); \
    }                                                                  \
  }

typedef enum {
  Quant = 0,
  DeQuant = 1,
  Max = 2,
  Lut = 3,
} AstlOpName;

typedef struct {
  std::string dtype;
  std::vector<uint32_t> shape;
} TensorParam;

typedef struct {
  std::vector<TensorParam> inputs;
  TensorParam output;
  std::vector<AstlOpName> fused;
} CommonParam;

typedef struct {
  int kernel;
  int stride;
  int padding;
  TensorParam output;
  std::vector<TensorParam> inputs;
  std::vector<AstlOpName> fused;
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

