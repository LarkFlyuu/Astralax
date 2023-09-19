#include "Operator.h"

class AddOperator : public Operator {
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
  return new AddOperator(); 
}
