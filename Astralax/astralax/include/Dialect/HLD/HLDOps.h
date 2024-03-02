#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

#include "Dialect/HLD/HldEnums.h.inc"
#include "Dialect/HLD/HldOpsDialect.h.inc"

#include "Dialect/HLD/Interfaces/ShapeInterface.h.inc"

#include "Traits/CommonTrait.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/HLD/HldAttr.h.inc"

#define GET_OP_CLASSES
#include "Dialect/HLD/HldOps.h.inc"
