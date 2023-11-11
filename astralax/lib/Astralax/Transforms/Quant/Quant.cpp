#include "Astralax/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

namespace astl {

class QuantPass : public QuantBase<QuantPass> {
public:
  QuantPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createQuantPass() {
  return std::make_unique<QuantPass>();
}

} // namespace astl