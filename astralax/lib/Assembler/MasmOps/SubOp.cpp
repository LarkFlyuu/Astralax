#include "Assembler/MASM/Operator.h"

namespace astl {
namespace masm {

class SubOperator : public Operator {  
public:
  bool match(mlir::Operation *op) override {
    return false;
  }

  std::string getSharding(std::string ofmSharding) override {
    return ofmSharding;
  }

  std::string getTiling(std::string ofmTiling) override {
    return ofmTiling;
  }
};

extern "C" Operator* create() {
  return new SubOperator(); 
}

}  // namespace masm
}  // namespace astl