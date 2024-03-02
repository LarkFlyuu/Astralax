//===--------------------------------
// Define ASTL HLD Base Dialect
//===--------------------------------
#ifndef ASTL_HLD_BASE_DIALECT_OPS
#define ASTL_HLD_BASE_DIALECT_OPS
include "mlir/IR/OpBase.td"
include "Dialect/Enums.td"
include "Dialect/Types.td"
include "Dialect/Attrs.td"
include "Interfaces/ShapeInterface.td"

def CommonTrait : NativeOpTrait<"CommonTrait">;
defvar CommonTraits = [NoTerminator, CommonTrait];
defvar CommonIntfcs = [DeclareOpInterfaceMethods<ShapeInterface>];

def Hld_Dialect : Dialect {
  let name = "hld";
  let summary = "Astralax high level dialect";
  let description = [{
    this dialect is used to define base the 
    Enum, attrs, types for Astralax compiler
  }];
  let cppNamespace = "::astl::hld";
  let useDefaultAttributePrinterParser = 1;
}

//===--------------------------------
// Define Astl Base Op
//===--------------------------------

class Hld_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Hld_Dialect, mnemonic, !listconcat(traits, CommonTraits)> {
  let summary = "Astralax Base Op";
}

def Hld_OpTypeAttr : EnumAttr<Hld_Dialect, Ax_OpType, "OpTypeAttr"> {
  let cppNamespace = "::astl::hld";
}
def Hld_LayoutAttr : EnumAttr<Hld_Dialect, Ax_Layout, "LayoutAttr"> {
  let cppNamespace = "::astl::hld";
}
def Hld_DataTypeAttr : EnumAttr<Hld_Dialect, Ax_DataType, "DataTypeAttr"> {
  let cppNamespace = "::astl::hld";
}

def Hld_TensorAttr : Ax_Attr<Hld_Dialect, "Tensor", []> {
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

def Hld_NoneOp : Hld_BaseOp<"None"> {
  let summary = "none op";
  let description = [{
    this op is used to define none type
  }];
  let results = (outs NoneType);
}
def Hld_InputOp : Hld_BaseOp<"Input"> {
  let summary = "input op";
  let description = [{
    this op is used to define input type
  }];
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<Hld_LayoutAttr>:$layout
  );
  let results = (outs ATensor:$output);
}
def Hld_ConstantOp : Hld_BaseOp<"Constant"> {
  let summary = "constant op";
  let results = (outs ATensor:$output);
}

// ===------------------------
// Define ONNX Operators
// ===------------------------

def Hld_AbsOp : Hld_BaseOp<"Abs", CommonIntfcs> {
  let summary = "abs op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_AcosOp : Hld_BaseOp<"Acos"> {
  let summary = "acos op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_AcoshOp : Hld_BaseOp<"Acosh"> {
  let summary = "acosh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_AddOp : Hld_BaseOp<"Add", CommonIntfcs> {
  let summary = "add op";
  let arguments = (ins ATensor:$lhs, ATensorOrNone:$rhs, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Hld_AndOp : Hld_BaseOp<"And"> {
  let summary = "and op";
  let arguments = (ins ATensor:$lhs, ATensor:$rhs);
  let results = (outs ATensor:$output);
}
def Hld_ArgMaxOp : Hld_BaseOp<"ArgMax"> {
  let summary = "argmax op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ArgMinOp : Hld_BaseOp<"ArgMin"> {
  let summary = "argmin op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_AsinOp : Hld_BaseOp<"Asin"> {
  let summary = "asin op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ArgSinhOp : Hld_BaseOp<"ArgSinh"> {
  let summary = "argsinh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_AtanOp : Hld_BaseOp<"Atan"> {
  let summary = "atan op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_AtanhOp : Hld_BaseOp<"Atanh"> {
  let summary = "atanh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_AveragePoolOp : Hld_BaseOp<"AveragePool", CommonIntfcs> {
  let summary = "average pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_BatchNormalizationOp : Hld_BaseOp<"BatchNormalization", CommonIntfcs> {
  let summary = "batch normalization op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_CastOp : Hld_BaseOp<"Cast"> {
  let summary = "cast op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_CeilOp : Hld_BaseOp<"Ceil"> {
  let summary = "ceil op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ClipOp : Hld_BaseOp<"Clip"> {
  let summary = "clip op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ConcatOp : Hld_BaseOp<"Concat", CommonIntfcs> {
  let summary = "concat op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ConvOp : Hld_BaseOp<"Conv", CommonIntfcs> {
  let summary = "conv op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ConvTransposeOp : Hld_BaseOp<"ConvTranspose", CommonIntfcs> {
  let summary = "conv transpose op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_CosOp : Hld_BaseOp<"Cos"> {
  let summary = "cos op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_CoshOp : Hld_BaseOp<"Cosh"> {
  let summary = "cosh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_DepthToSpaceOp : Hld_BaseOp<"DepthToSpace", CommonIntfcs> {
  let summary = "depth to space op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_DivOp : Hld_BaseOp<"Div", CommonIntfcs> {
  let summary = "div op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$divided, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Hld_DropoutOp : Hld_BaseOp<"Dropout"> {
  let summary = "dropout op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_EluOp : Hld_BaseOp<"Elu"> {
  let summary = "elu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_EqualOp : Hld_BaseOp<"Equal"> {
  let summary = "equal op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ErfOp : Hld_BaseOp<"Erf"> {
  let summary = "erf op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ExpOp : Hld_BaseOp<"Exp"> {
  let summary = "exp op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ExpandOp : Hld_BaseOp<"Expand"> {
  let summary = "expand op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_FlattenOp : Hld_BaseOp<"Flatten"> {
  let summary = "flatten op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_FloorOp : Hld_BaseOp<"Floor"> {
  let summary = "floor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_GatherOp : Hld_BaseOp<"Gather"> {
  let summary = "gather op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_GemmOp : Hld_BaseOp<"Gemm", CommonIntfcs> {
  let summary = "gemm op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_GRUOp : Hld_BaseOp<"GRU", CommonIntfcs> {
  let summary = "gru op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_GlobalAveragePoolOp : Hld_BaseOp<"GlobalAveragePool", CommonIntfcs> {
  let summary = "global average pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_GlobalMaxPoolOp : Hld_BaseOp<"GlobalMaxPool", CommonIntfcs> {
  let summary = "global max pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_GlobalLpPoolOp : Hld_BaseOp<"GlobalLpPool"> {
  let summary = "global lp pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_GreaterOp : Hld_BaseOp<"Greater"> {
  let summary = "greater op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_HardSigmoidOp : Hld_BaseOp<"HardSigmoid"> {
  let summary = "hard sigmoid op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_IdentityOp : Hld_BaseOp<"Identity", CommonIntfcs> {
  let summary = "identity op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_IfOp : Hld_BaseOp<"If"> {
  let summary = "if op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_InstanceNormalizationOp : Hld_BaseOp<"InstanceNormalization"> {
  let summary = "instance normalization op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_LRNOp : Hld_BaseOp<"LRN", CommonIntfcs> {
  let summary = "lrn op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_LSTMOp : Hld_BaseOp<"LSTM", CommonIntfcs> {
  let summary = "lstm op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_LeakyReluOp : Hld_BaseOp<"LeakyRelu"> {
  let summary = "leaky relu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_LessOp : Hld_BaseOp<"Less"> {
  let summary = "less op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_LogOp : Hld_BaseOp<"Log"> {
  let summary = "log op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_LogSoftmaxOp : Hld_BaseOp<"LogSoftmax"> {
  let summary = "log softmax op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_LoopOp : Hld_BaseOp<"Loop"> {
  let summary = "loop op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_MatMulOp : Hld_BaseOp<"MatMul", CommonIntfcs> {
  let summary = "matmul op";
  let arguments = (ins ATensor:$lhs, ATensor:$rhs);
  let results = (outs ATensor:$output);
}
def Hld_MaxOp : Hld_BaseOp<"Max", CommonIntfcs> {
  let summary = "max op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$comp, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Hld_MaxPoolOp : Hld_BaseOp<"MaxPool", CommonIntfcs> {
  let summary = "max pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_MaxRoiPoolOp : Hld_BaseOp<"MaxRoiPool"> {
  let summary = "max roi pool op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_MeanOp : Hld_BaseOp<"Mean"> {
  let summary = "mean op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$comp, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Hld_MinOp : Hld_BaseOp<"Min", CommonIntfcs> {
  let summary = "min op";
  let arguments = (ins ATensor:$input, ATensorOrNone:$comp, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Hld_MulOp : Hld_BaseOp<"Mul", CommonIntfcs> {
  let summary = "mul op";
  let arguments = (ins ATensor:$lhs, ATensorOrNone:$rhs, OptionalAttr<F32Attr>:$scalar);
  let results = (outs ATensor:$output);
}
def Hld_MultinomialOp : Hld_BaseOp<"Multinomial"> {
  let summary = "multinomial op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_NegOp : Hld_BaseOp<"Neg"> {
  let summary = "neg op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_NotOp : Hld_BaseOp<"Not"> {
  let summary = "not op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_OrOp : Hld_BaseOp<"Or"> {
  let summary = "or op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_PReluOp : Hld_BaseOp<"PRelu"> {
  let summary = "prelu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_PadOp : Hld_BaseOp<"Pad"> {
  let summary = "pad op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_PowOp : Hld_BaseOp<"Pow"> {
  let summary = "pow op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_QLinearConvOp : Hld_BaseOp<"QLinearConv"> {
  let summary = "q linear conv op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_QuantizeLinearOp : Hld_BaseOp<"QuantizeLinear"> {
  let summary = "quantize linear op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_RNNOp : Hld_BaseOp<"RNN", CommonIntfcs> {
  let summary = "rnn op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_RandomNormalOp : Hld_BaseOp<"RandomNormal"> {
  let summary = "random normal op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_RandomNormalLikeOp : Hld_BaseOp<"RandomNormalLike"> {
  let summary = "random normal like op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_RandomUniformOp : Hld_BaseOp<"RandomUniform"> {
  let summary = "random uniform op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_RandomUniformLikeOp : Hld_BaseOp<"RandomUniformLike"> {
  let summary = "random uniform like op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReciprocalOp : Hld_BaseOp<"Reciprocal", CommonIntfcs> {
  let summary = "reciprocal op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceL1Op : Hld_BaseOp<"ReduceL1"> {
  let summary = "reduce l1 op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceL2Op : Hld_BaseOp<"ReduceL2"> {
  let summary = "reduce l2 op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceLogSumOp : Hld_BaseOp<"ReduceLogSum"> {
  let summary = "reduce log sum op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceLogSumExpOp : Hld_BaseOp<"ReduceLogSumExp"> {
  let summary = "reduce log sum exp op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceMaxOp : Hld_BaseOp<"ReduceMax", CommonIntfcs> {
  let summary = "reduce max op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceMeanOp : Hld_BaseOp<"ReduceMean", CommonIntfcs> {
  let summary = "reduce mean op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceMinOp : Hld_BaseOp<"ReduceMin"> {
  let summary = "reduce min op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceProdOp : Hld_BaseOp<"ReduceProd"> {
  let summary = "reduce prod op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceSumOp : Hld_BaseOp<"ReduceSum"> {
  let summary = "reduce sum op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReduceSumSquareOp : Hld_BaseOp<"ReduceSumSquare"> {
  let summary = "reduce sum square op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReluOp : Hld_BaseOp<"Relu", CommonIntfcs> {
  let summary = "relu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReshapeOp : Hld_BaseOp<"Reshape", CommonIntfcs> {
  let summary = "reshape op";
  let arguments = (ins ATensor:$input, I64ArrayAttr:$dims);
  let results = (outs ATensor:$output);
}
def Hld_ResizeOp : Hld_BaseOp<"Resize", CommonIntfcs> {
  let summary = "resize op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ReverseSequenceOp : Hld_BaseOp<"ReverseSequence"> {
  let summary = "reverse sequence op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_RoiAlignOp : Hld_BaseOp<"RoiAlign"> {
  let summary = "roi align op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_RoundOp : Hld_BaseOp<"Round"> {
  let summary = "round op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SVMClassifierOp : Hld_BaseOp<"SVMClassifier"> {
  let summary = "svm classifier op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SVMRegressorOp : Hld_BaseOp<"SVMRegressor"> {
  let summary = "svm regressor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ScatterOp : Hld_BaseOp<"Scatter"> {
  let summary = "scatter op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ScatterElementsOp : Hld_BaseOp<"ScatterElements"> {
  let summary = "scatter elements op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ScatterNDOp : Hld_BaseOp<"ScatterND"> {
  let summary = "scatter nd op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SeluOp : Hld_BaseOp<"Selu"> {
  let summary = "selu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ShapeOp : Hld_BaseOp<"Shape"> {
  let summary = "shape op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ShrinkOp : Hld_BaseOp<"Shrink"> {
  let summary = "shrink op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SigmoidOp : Hld_BaseOp<"Sigmoid", CommonIntfcs> {
  let summary = "sigmoid op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SignOp : Hld_BaseOp<"Sign"> {
  let summary = "sign op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SinOp : Hld_BaseOp<"Sin"> {
  let summary = "sin op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SinhOp : Hld_BaseOp<"Sinh"> {
  let summary = "sinh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SizeOp : Hld_BaseOp<"Size"> {
  let summary = "size op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SliceOp : Hld_BaseOp<"Slice", CommonIntfcs> {
  let summary = "slice op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SoftmaxOp : Hld_BaseOp<"Softmax", CommonIntfcs> {
  let summary = "softmax op";
  let arguments = (ins ATensor:$input, I64Attr:$axis);
  let results = (outs ATensor:$output);
}
def Hld_SoftplusOp : Hld_BaseOp<"Softplus"> {
  let summary = "softplus op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SoftsignOp : Hld_BaseOp<"Softsign"> {
  let summary = "softsign op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SpaceToDepthOp : Hld_BaseOp<"SpaceToDepth", CommonIntfcs> {
  let summary = "space to depth op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SplitOp : Hld_BaseOp<"Split", CommonIntfcs> {
  let summary = "split op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SqrtOp : Hld_BaseOp<"Sqrt"> {
  let summary = "sqrt op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SqueezeOp : Hld_BaseOp<"Squeeze", CommonIntfcs> {
  let summary = "squeeze op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_StringNormalizerOp : Hld_BaseOp<"StringNormalizer"> {
  let summary = "string normalizer op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SubOp : Hld_BaseOp<"Sub", CommonIntfcs> {
  let summary = "sub op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_SumOp : Hld_BaseOp<"Sum", CommonIntfcs> {
  let summary = "sum op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_TanOp : Hld_BaseOp<"Tan"> {
  let summary = "tan op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_TanhOp : Hld_BaseOp<"Tanh"> {
  let summary = "tanh op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_TfIdfVectorizerOp : Hld_BaseOp<"TfIdfVectorizer"> {
  let summary = "tf idf vectorizer op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_ThresholdedReluOp : Hld_BaseOp<"ThresholdedRelu"> {
  let summary = "thresholded relu op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_TileOp : Hld_BaseOp<"Tile"> {
  let summary = "tile op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_TopKOp : Hld_BaseOp<"TopK", CommonIntfcs> {
  let summary = "top k op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_TransposeOp : Hld_BaseOp<"Transpose", CommonIntfcs> {
  let summary = "transpose op";
  let arguments = (ins ATensor:$input, I64ArrayAttr:$perm);
  let results = (outs ATensor:$output);
}
def Hld_TreeEnsembleClassifierOp : Hld_BaseOp<"TreeEnsembleClassifier"> {
  let summary = "tree ensemble classifier op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_TreeEnsembleRegressorOp : Hld_BaseOp<"TreeEnsembleRegressor"> {
  let summary = "tree ensemble regressor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_UnsqueezeOp : Hld_BaseOp<"Unsqueeze", CommonIntfcs> {
  let summary = "unsqueeze op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_UpsampleOp : Hld_BaseOp<"Upsample", CommonIntfcs> {
  let summary = "upsample op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_WhereOp : Hld_BaseOp<"Where"> {
  let summary = "where op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}
def Hld_XorOp : Hld_BaseOp<"Xor"> {
  let summary = "xor op";
  let arguments = (ins ATensor:$input);
  let results = (outs ATensor:$output);
}

#endif
