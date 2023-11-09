//===--------------------------------
// Define ASTL Base Dialect
//===--------------------------------
#ifndef ASTL_BASE_DIALECT_OPS
#define ASTL_BASE_DIALECT_OPS

include "mlir/IR/OpBase.td"
include "AstlEnums.td"
include "AstlTypes.td"
include "AstlAttrs.td"

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
    Op<Astl_Dialect, mnemonic, !listconcat(traits, [NoTerminator])> {
  let summary = "Astralax Base Op";
  // let regions = (region VariadicRegion<SizedRegion<1>>:$body);
}

def Astl_LayerNormOp : Astl_BaseOp<"LayerNorm"> {
  let summary = "layernorm op";
  let arguments = (ins ATensor:$ifm,
                   ATensor:$bias,
                   ATensorOrNone:$post_lut_params,
                   ATensorOrNone:$quant_mean_params,
                   ATensorOrNone:$quant_divisor_params,
                   ATensorOrNone:$quant_out_params,
                   ATensorOrNone:$quant_sign_params,
                   OptionalAttr<I64Attr>:$divisor,
                   OptionalAttr<I64Attr>:$remain_bits);
  let results = (outs ATensor:$output);
}

#endif
