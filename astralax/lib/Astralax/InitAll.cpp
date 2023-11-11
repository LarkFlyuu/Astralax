#include "Astralax/Transforms/Passes.h"
#include "Astralax/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace astl {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::func::FuncDialect, AstlDialect>();
}

void registerAllPasses() {
  registerAstlTransformsPasses();
  registerAstlConversionPasses();
}
} // namespace astl