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

def Astl_LayoutAttr : EnumAttr<Astl_Dialect, Ax_Layout, "LayoutAttr">;
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
    OptionalAttr<Ax_Layout>:$layout
  );
  let results = (outs ATensor:$output);
}

def Astl_ConstantOp : Astl_BaseOp<"Constant"> {
  let summary = "constant op";
  let results = (outs ATensor:$output);
}
def Astl_ConcatOp : Astl_BaseOp<"Concat"> {
  let summary = "concat op";
  let arguments = (ins
    Variadic<ATensor>:$inputs,
    OptionalAttr<I8Attr>:$axis
  );
  let results = (outs ATensor:$output);
}

def Astl_SliceOp : Astl_BaseOp<"Slice"> {
  let summary = "slice op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$begin,
    OptionalAttr<I64Attr>:$end,
    OptionalAttr<I64Attr>:$stride
  );
  let results = (outs ATensor:$output);
}

def Astl_DesliceOp : Astl_BaseOp<"Deslice"> {
  let summary = "deslice op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$begin,
    OptionalAttr<I64Attr>:$end,
    OptionalAttr<I64Attr>:$stride
  );
  let results = (outs ATensor:$output);
}

def Astl_ReshapeOp : Astl_BaseOp<"Reshape"> {
  let summary = "reshape op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$shape
  );
  let results = (outs ATensor:$output);
}

def Astl_TransposeOp : Astl_BaseOp<"Transpose"> {
  let summary = "transpose op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$perm
  );
  let results = (outs ATensor:$output);
}

def Astl_SqueezeOp : Astl_BaseOp<"Squeeze"> {
  let summary = "squeeze op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$axis
  );
  let results = (outs ATensor:$output);
}

def Astl_UnsqueezeOp : Astl_BaseOp<"Unsqueeze"> {
  let summary = "unsqueeze op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$axis
  );
  let results = (outs ATensor:$output);
}

def Astl_FlattenOp : Astl_BaseOp<"Flatten"> {
  let summary = "flatten op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$axis
  );
  let results = (outs ATensor:$output);
}

def Astl_PixelShuffleOp : Astl_BaseOp<"PixelShuffle"> {
  let summary = "pixel shuffle op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$upscaleFactor
  );
  let results = (outs ATensor:$output);
}

def Astl_ExpandOp : Astl_BaseOp<"Expand"> {
  let summary = "expand op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$shape
  );
  let results = (outs ATensor:$output);
}

def Astl_Spatial2ChannelOp : Astl_BaseOp<"Spatial2Channel"> {
  let summary = "spatial2channel op";
  let arguments = (ins
    ATensor:$input,
    OptionalAttr<I64Attr>:$kernelSize,
    OptionalAttr<I64Attr>:$stride,
    OptionalAttr<I64Attr>:$padding,
    OptionalAttr<I64Attr>:$dilation,
    OptionalAttr<I64Attr>:$groups
  );
  let results = (outs ATensor:$output);
}

#endif
