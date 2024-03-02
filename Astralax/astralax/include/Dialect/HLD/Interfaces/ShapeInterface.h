#pragma once
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace hld {

// only one output, and output shape is the same with the first input shape
void common_shape_inference(mlir::Operation *op);

// binary operation with the implicit broadcast
// void broadcast_shape_inference(mlir::Operation *op);

// reshape broadcast tensor if shape dim is not same with the expected tensor
// void broadcast_tensor_reshape(const mlir::Value &expect, mlir::Value input);

}  // namespace hld
}  // namespace mlir