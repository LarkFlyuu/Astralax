#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"

#include "Astralax/Dialect/AstlEnums.h.inc"
#include "Astralax/Dialect/AstlOpsDialect.h.inc"

#include "Astralax/Traits/CommonTrait.h"

#define GET_ATTRDEF_CLASSES
#include "Astralax/Dialect/AstlAttr.h.inc"

#define GET_OP_CLASSES
#include "Astralax/Dialect/AstlOps.h.inc"
