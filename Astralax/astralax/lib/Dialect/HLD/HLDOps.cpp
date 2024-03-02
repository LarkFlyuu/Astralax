#include "Dialect/HLD/HLDOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace astl;
using namespace astl::hld;

#include "Dialect/HLD/HldOpsDialect.cpp.inc"
void HldDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/HLD/HldOps.cpp.inc"
>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/HLD/HldAttr.cpp.inc"
>();
}

#ifndef HLD_DIALECT_COMMON_ENUMS
#define HLD_DIALECT_COMMON_ENUMS
#include "Dialect/HLD/HldEnums.cpp.inc"
#endif

#define GET_ATTRDEF_CLASSES
#include "Dialect/HLD/HldAttr.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/HLD/HldOps.cpp.inc"
