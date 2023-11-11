#include "Astralax/Conversion/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;

namespace astl {

class InitPass : public InitBase<InitPass> {
public:
  InitPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createInitPass() {
  return std::make_unique<InitPass>();
}

} // namespace astl