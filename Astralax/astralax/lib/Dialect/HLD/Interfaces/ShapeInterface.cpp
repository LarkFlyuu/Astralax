#include "Dialect/HLD/HLDOps.h"
#include "Dialect/HLD/Interfaces/ShapeInterface.h"

namespace astl::hld {

void common_shape_inference(mlir::Operation *op) {}

void AbsOp::shape_inference() {
  common_shape_inference(getOperation());
}

void AddOp::shape_inference() {
  common_shape_inference(getOperation());
}

void AveragePoolOp::shape_inference() {
  common_shape_inference(getOperation());
}

void BatchNormalizationOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ConcatOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ConvOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ConvTransposeOp::shape_inference() {
  common_shape_inference(getOperation());
}

void DepthToSpaceOp::shape_inference() {
  common_shape_inference(getOperation());
}

void DivOp::shape_inference() {
  common_shape_inference(getOperation());
}

void GemmOp::shape_inference() {
  common_shape_inference(getOperation());
}

void GlobalAveragePoolOp::shape_inference() {
  common_shape_inference(getOperation());
}

void GlobalMaxPoolOp::shape_inference() {
  common_shape_inference(getOperation());
}

void IdentityOp::shape_inference() {
  common_shape_inference(getOperation());
}

void LRNOp::shape_inference() {
  common_shape_inference(getOperation());
}

void LSTMOp::shape_inference() {
  common_shape_inference(getOperation());
}

void MatMulOp::shape_inference() {
  common_shape_inference(getOperation());
}

void MaxOp::shape_inference() {
  common_shape_inference(getOperation());
}

void MaxPoolOp::shape_inference() {
  common_shape_inference(getOperation());
}

void MinOp::shape_inference() {
  common_shape_inference(getOperation());
}

void MulOp::shape_inference() {
  common_shape_inference(getOperation());
}

void RNNOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ReciprocalOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ReduceMaxOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ReduceMeanOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ReluOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ReshapeOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ResizeOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SigmoidOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SliceOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SoftmaxOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SpaceToDepthOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SplitOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SqueezeOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SubOp::shape_inference() {
  common_shape_inference(getOperation());
}

void SumOp::shape_inference() {
  common_shape_inference(getOperation());
}

void TopKOp::shape_inference() {
  common_shape_inference(getOperation());
}

void TransposeOp::shape_inference() {
  common_shape_inference(getOperation());
}

void UnsqueezeOp::shape_inference() {
  common_shape_inference(getOperation());
}

void UpsampleOp::shape_inference() {
  common_shape_inference(getOperation());
}

}  // namespace astl::hld
