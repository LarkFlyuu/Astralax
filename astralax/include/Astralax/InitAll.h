#pragma once
#include "mlir/IR/Dialect.h"
#include "Astralax/Dialect/AstlOps.h"

namespace astl {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace astl