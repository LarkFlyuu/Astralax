#include "Astralax/Dialect/AstlOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace astl;

#include "Astralax/Dialect/AstlOpsDialect.cpp.inc"
void AstlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Astralax/Dialect/AstlOps.cpp.inc"
>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Astralax/Dialect/AstlAttr.cpp.inc"
>();
}

#ifndef ASTL_DIALECT_COMMON_ENUMS
#define ASTL_DIALECT_COMMON_ENUMS
#include "Astralax/Dialect/AstlEnums.cpp.inc"
#endif

#define GET_ATTRDEF_CLASSES
#include "Astralax/Dialect/AstlAttr.cpp.inc"
#define GET_OP_CLASSES
#include "Astralax/Dialect/AstlOps.cpp.inc"
