#include "Assembler/MASM/Operator.h"

namespace astl {
namespace masm {

class Conv3x3s1p1_i8xi10ti8 : public Operator {
public:
  
  bool match(AstlOp *op) override {
    ConvParam param;
    param.kernel = 3;
    param.stride = 1;
    param.padding = 1;
    param.output = Tensor();
    param.inputs = {};
    param.fused = {};
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

}  // namespace masm
}  // namespace astl