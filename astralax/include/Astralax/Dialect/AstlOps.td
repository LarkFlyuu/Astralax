//===--------------------------------
// Define ASTL Base Dialect
//===--------------------------------
#ifndef ASTL_BASE_DIALECT_OPS
#define ASTL_BASE_DIALECT_OPS
include "mlir/IR/OpBase.td"
include "AstlEnums.td"
include "AstlTypes.td"
include "AstlAttrs.td"

def CommonTrait : NativeOpTrait<"CommonTrait">;
defvar CommonTraits = [NoTerminator, CommonTrait];

def Astl_Dialect : Dialect {
  let name = "astl";
  let summary = "Astralax base dialect";
  let description = [{
    this dialect is used to define base the 
    Enum, attrs, types for Astralax compiler
  }];
  let cppNamespace = "::astl";
  let useDefaultAttributePrinterParser = 1;
}

//===--------------------------------
// Define Astl Base Op
//===--------------------------------

class Astl_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Astl_Dialect, mnemonic, !listconcat(traits, CommonTraits, [CommonTrait])> {
  let summary = "Astralax Base Op";
  // let regions = (region VariadicRegion<SizedRegion<1>>:$body);
}

def Astl_OpTypeAttr : EnumAttr<Astl_Dialect, Ax_OpType, "OpTypeAttr">;
def Astl_LayoutAttr : EnumAttr<Astl_Dialect, Ax_Layout, "LayoutAttr">;
def Astl_DataTypeAttr : EnumAttr<Astl_Dialect, Ax_DataType, "DataTypeAttr">;

def Astl_TensorAttr : Ax_Attr<Astl_Dialect, "Tensor", []> {
  let summary = "Tensor Attr";
  let parameters = (ins
    OptionalParameter<"LayoutAttr">:$layout,
    OptionalParameter<"::mlir::StringAttr">:$name,
    OptionalParameter<"::mlir::DenseElementsAttr">:$data,
    OptionalParameter<"::mlir::ArrayAttr">:$mesh, // device parallel
    OptionalParameter<"::mlir::ArrayAttr">:$sharding, // pipeline parallel
    OptionalParameter<"::mlir::BoolAttr">:$isSharded
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::MLIRContext *":$context), [{
      return Base::get(context);
    }]>
  ];
}

// ===------------------------
// Define Astl Base Ops
// ===------------------------

def Astl_NoneOp : Astl_BaseOp<"None"> {
  let summary = "none op";
  let description = [{
    this op is used to define none type
  }];
  let results = (outs NoneType);
}
def Astl_InputOp : Astl_BaseOp<"Input"> {
  let summary = "input op";
  let description = [{
    this op is used to define input type
  }];
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<Astl_LayoutAttr>:$layout
  );
  let results = (outs ATensor:$output);
}
def Astl_ConstantOp : Astl_BaseOp<"Constant"> {
  let summary = "constant op";
  let results = (outs ATensor:$output);
}

// ===------------------------
// Define ONNX Operators
// ===------------------------

def Astl_AbsOp : Astl_BaseOp<"Abs"> {
  let summary = "abs op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_AcosOp : Astl_BaseOp<"Acos"> {
  let summary = "acos op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_AcoshOp : Astl_BaseOp<"Acosh"> {
  let summary = "acosh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_AddOp : Astl_BaseOp<"Add"> {
  let summary = "add op";
  let arguments = (ins ATensor:$lhs, ATensorOrNone:$rhs, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Astl_AndOp : Astl_BaseOp<"And"> {
  let summary = "and op";
  let arguments = (ins ATensor:$lhs, ATensor:$rhs);
  let results = (outs ATensor:$output);
}
def Astl_ArgMaxOp : Astl_BaseOp<"ArgMax"> {
  let summary = "argmax op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ArgMinOp : Astl_BaseOp<"ArgMin"> {
  let summary = "argmin op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_AsinOp : Astl_BaseOp<"Asin"> {
  let summary = "asin op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ArgSinhOp : Astl_BaseOp<"ArgSinh"> {
  let summary = "argsinh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_AtanOp : Astl_BaseOp<"Atan"> {
  let summary = "atan op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_AtanhOp : Astl_BaseOp<"Atanh"> {
  let summary = "atanh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_AveragePoolOp : Astl_BaseOp<"AveragePool"> {
  let summary = "average pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_BatchNormalizationOp : Astl_BaseOp<"BatchNormalization"> {
  let summary = "batch normalization op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_CastOp : Astl_BaseOp<"Cast"> {
  let summary = "cast op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_CeilOp : Astl_BaseOp<"Ceil"> {
  let summary = "ceil op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ClipOp : Astl_BaseOp<"Clip"> {
  let summary = "clip op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ConcatOp : Astl_BaseOp<"Concat"> {
  let summary = "concat op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ConvOp : Astl_BaseOp<"Conv"> {
  let summary = "conv op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ConvTransposeOp : Astl_BaseOp<"ConvTranspose"> {
  let summary = "conv transpose op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_CosOp : Astl_BaseOp<"Cos"> {
  let summary = "cos op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_CoshOp : Astl_BaseOp<"Cosh"> {
  let summary = "cosh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_DepthToSpaceOp : Astl_BaseOp<"DepthToSpace"> {
  let summary = "depth to space op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_DivOp : Astl_BaseOp<"Div"> {
  let summary = "div op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$divided, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Astl_DropoutOp : Astl_BaseOp<"Dropout"> {
  let summary = "dropout op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_EluOp : Astl_BaseOp<"Elu"> {
  let summary = "elu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_EqualOp : Astl_BaseOp<"Equal"> {
  let summary = "equal op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ErfOp : Astl_BaseOp<"Erf"> {
  let summary = "erf op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ExpOp : Astl_BaseOp<"Exp"> {
  let summary = "exp op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ExpandOp : Astl_BaseOp<"Expand"> {
  let summary = "expand op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_FlattenOp : Astl_BaseOp<"Flatten"> {
  let summary = "flatten op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_FloorOp : Astl_BaseOp<"Floor"> {
  let summary = "floor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_GatherOp : Astl_BaseOp<"Gather"> {
  let summary = "gather op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_GemmOp : Astl_BaseOp<"Gemm"> {
  let summary = "gemm op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_GRUOp : Astl_BaseOp<"GRU"> {
  let summary = "gru op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_GlobalAveragePoolOp : Astl_BaseOp<"GlobalAveragePool"> {
  let summary = "global average pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_GlobalMaxPoolOp : Astl_BaseOp<"GlobalMaxPool"> {
  let summary = "global max pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_GlobalLpPoolOp : Astl_BaseOp<"GlobalLpPool"> {
  let summary = "global lp pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_GreaterOp : Astl_BaseOp<"Greater"> {
  let summary = "greater op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_HardSigmoidOp : Astl_BaseOp<"HardSigmoid"> {
  let summary = "hard sigmoid op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_IdentityOp : Astl_BaseOp<"Identity"> {
  let summary = "identity op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_IfOp : Astl_BaseOp<"If"> {
  let summary = "if op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_InstanceNormalizationOp : Astl_BaseOp<"InstanceNormalization"> {
  let summary = "instance normalization op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_LRNOp : Astl_BaseOp<"LRN"> {
  let summary = "lrn op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_LSTMOp : Astl_BaseOp<"LSTM"> {
  let summary = "lstm op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_LeakyReluOp : Astl_BaseOp<"LeakyRelu"> {
  let summary = "leaky relu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_LessOp : Astl_BaseOp<"Less"> {
  let summary = "less op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_LogOp : Astl_BaseOp<"Log"> {
  let summary = "log op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_LogSoftmaxOp : Astl_BaseOp<"LogSoftmax"> {
  let summary = "log softmax op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_LoopOp : Astl_BaseOp<"Loop"> {
  let summary = "loop op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_MatMulOp : Astl_BaseOp<"MatMul"> {
  let summary = "matmul op";
  let arguments = (ins ATensor:$lhs, ATensor:$rhs);
  let results = (outs ATensor:$output);
}
def Astl_MaxOp : Astl_BaseOp<"Max"> {
  let summary = "max op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$comp, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Astl_MaxPoolOp : Astl_BaseOp<"MaxPool"> {
  let summary = "max pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_MaxRoiPoolOp : Astl_BaseOp<"MaxRoiPool"> {
  let summary = "max roi pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_MeanOp : Astl_BaseOp<"Mean"> {
  let summary = "mean op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$comp, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Astl_MinOp : Astl_BaseOp<"Min"> {
  let summary = "min op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$comp, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Astl_MulOp : Astl_BaseOp<"Mul"> {
  let summary = "mul op";
  let arguments = (ins ATensor:$lhs, ATensorOrNone:$rhs, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Astl_MultinomialOp : Astl_BaseOp<"Multinomial"> {
  let summary = "multinomial op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_NegOp : Astl_BaseOp<"Neg"> {
  let summary = "neg op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_NotOp : Astl_BaseOp<"Not"> {
  let summary = "not op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_OrOp : Astl_BaseOp<"Or"> {
  let summary = "or op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_PReluOp : Astl_BaseOp<"PRelu"> {
  let summary = "prelu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_PadOp : Astl_BaseOp<"Pad"> {
  let summary = "pad op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_PowOp : Astl_BaseOp<"Pow"> {
  let summary = "pow op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_QLinearConvOp : Astl_BaseOp<"QLinearConv"> {
  let summary = "q linear conv op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_QuantizeLinearOp : Astl_BaseOp<"QuantizeLinear"> {
  let summary = "quantize linear op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_RNNOp : Astl_BaseOp<"RNN"> {
  let summary = "rnn op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_RandomNormalOp : Astl_BaseOp<"RandomNormal"> {
  let summary = "random normal op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_RandomNormalLikeOp : Astl_BaseOp<"RandomNormalLike"> {
  let summary = "random normal like op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_RandomUniformOp : Astl_BaseOp<"RandomUniform"> {
  let summary = "random uniform op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_RandomUniformLikeOp : Astl_BaseOp<"RandomUniformLike"> {
  let summary = "random uniform like op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReciprocalOp : Astl_BaseOp<"Reciprocal"> {
  let summary = "reciprocal op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceL1Op : Astl_BaseOp<"ReduceL1"> {
  let summary = "reduce l1 op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceL2Op : Astl_BaseOp<"ReduceL2"> {
  let summary = "reduce l2 op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceLogSumOp : Astl_BaseOp<"ReduceLogSum"> {
  let summary = "reduce log sum op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceLogSumExpOp : Astl_BaseOp<"ReduceLogSumExp"> {
  let summary = "reduce log sum exp op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceMaxOp : Astl_BaseOp<"ReduceMax"> {
  let summary = "reduce max op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceMeanOp : Astl_BaseOp<"ReduceMean"> {
  let summary = "reduce mean op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceMinOp : Astl_BaseOp<"ReduceMin"> {
  let summary = "reduce min op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceProdOp : Astl_BaseOp<"ReduceProd"> {
  let summary = "reduce prod op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceSumOp : Astl_BaseOp<"ReduceSum"> {
  let summary = "reduce sum op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReduceSumSquareOp : Astl_BaseOp<"ReduceSumSquare"> {
  let summary = "reduce sum square op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReluOp : Astl_BaseOp<"Relu"> {
  let summary = "relu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReshapeOp : Astl_BaseOp<"Reshape"> {
  let summary = "reshape op";
  let arguments = (ins ATensor:$input, I64ArrayAttr:$dims);
  let results = (outs ATensor:$output);
}
def Astl_ResizeOp : Astl_BaseOp<"Resize"> {
  let summary = "resize op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ReverseSequenceOp : Astl_BaseOp<"ReverseSequence"> {
  let summary = "reverse sequence op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_RoiAlignOp : Astl_BaseOp<"RoiAlign"> {
  let summary = "roi align op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_RoundOp : Astl_BaseOp<"Round"> {
  let summary = "round op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SVMClassifierOp : Astl_BaseOp<"SVMClassifier"> {
  let summary = "svm classifier op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SVMRegressorOp : Astl_BaseOp<"SVMRegressor"> {
  let summary = "svm regressor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ScatterOp : Astl_BaseOp<"Scatter"> {
  let summary = "scatter op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ScatterElementsOp : Astl_BaseOp<"ScatterElements"> {
  let summary = "scatter elements op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ScatterNDOp : Astl_BaseOp<"ScatterND"> {
  let summary = "scatter nd op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SeluOp : Astl_BaseOp<"Selu"> {
  let summary = "selu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ShapeOp : Astl_BaseOp<"Shape"> {
  let summary = "shape op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ShrinkOp : Astl_BaseOp<"Shrink"> {
  let summary = "shrink op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SigmoidOp : Astl_BaseOp<"Sigmoid"> {
  let summary = "sigmoid op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SignOp : Astl_BaseOp<"Sign"> {
  let summary = "sign op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SinOp : Astl_BaseOp<"Sin"> {
  let summary = "sin op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SinhOp : Astl_BaseOp<"Sinh"> {
  let summary = "sinh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SizeOp : Astl_BaseOp<"Size"> {
  let summary = "size op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SliceOp : Astl_BaseOp<"Slice"> {
  let summary = "slice op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SoftmaxOp : Astl_BaseOp<"Softmax"> {
  let summary = "softmax op";
  let arguments = (ins ATensor:$input, I64Attr:$axis);
  let results = (outs ATensor:$output);
}
def Astl_SoftplusOp : Astl_BaseOp<"Softplus"> {
  let summary = "softplus op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SoftsignOp : Astl_BaseOp<"Softsign"> {
  let summary = "softsign op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SpaceToDepthOp : Astl_BaseOp<"SpaceToDepth"> {
  let summary = "space to depth op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SplitOp : Astl_BaseOp<"Split"> {
  let summary = "split op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SqrtOp : Astl_BaseOp<"Sqrt"> {
  let summary = "sqrt op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SqueezeOp : Astl_BaseOp<"Squeeze"> {
  let summary = "squeeze op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_StringNormalizerOp : Astl_BaseOp<"StringNormalizer"> {
  let summary = "string normalizer op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SubOp : Astl_BaseOp<"Sub"> {
  let summary = "sub op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_SumOp : Astl_BaseOp<"Sum"> {
  let summary = "sum op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_TanOp : Astl_BaseOp<"Tan"> {
  let summary = "tan op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_TanhOp : Astl_BaseOp<"Tanh"> {
  let summary = "tanh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_TfIdfVectorizerOp : Astl_BaseOp<"TfIdfVectorizer"> {
  let summary = "tf idf vectorizer op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_ThresholdedReluOp : Astl_BaseOp<"ThresholdedRelu"> {
  let summary = "thresholded relu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_TileOp : Astl_BaseOp<"Tile"> {
  let summary = "tile op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_TopKOp : Astl_BaseOp<"TopK"> {
  let summary = "top k op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_TransposeOp : Astl_BaseOp<"Transpose"> {
  let summary = "transpose op";
  let arguments = (ins ATensor:$input, I64ArrayAttr:$perm);
  let results = (outs ATensor:$output);
}
def Astl_TreeEnsembleClassifierOp : Astl_BaseOp<"TreeEnsembleClassifier"> {
  let summary = "tree ensemble classifier op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_TreeEnsembleRegressorOp : Astl_BaseOp<"TreeEnsembleRegressor"> {
  let summary = "tree ensemble regressor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_UnsqueezeOp : Astl_BaseOp<"Unsqueeze"> {
  let summary = "unsqueeze op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_UpsampleOp : Astl_BaseOp<"Upsample"> {
  let summary = "upsample op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_WhereOp : Astl_BaseOp<"Where"> {
  let summary = "where op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Astl_XorOp : Astl_BaseOp<"Xor"> {
  let summary = "xor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}

#endif
