#include "Operator.h"

class Conv3x3s1p1_i8xi10ti8 : public Operator {
public:
  
  bool match(AstlOp *op) override {
    ConvParam param;
    param.kernel = 3;
    param.stride = 1;
    param.padding = 1;
    param.output = TensorParam();
    param.inputs = {};
    param.fused = {AstlOpName::Max, AstlOpName::Quant};
    return matchConv(op, param);
  }

  std::string getSharding(std::string ofmSharding) override {
    return ofmSharding;
  }

  std::string getTiling(std::string ofmTiling) override {
    return ofmTiling;
  }

};

extern "C" Operator* create() {
  return new Conv3x3s1p1_i8xi10ti8();
}
