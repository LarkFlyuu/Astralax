//===--------------------------------
// Define common Enums
//===--------------------------------
#ifndef ASTL_DIALECT_ENUMS_TD
#define ASTL_DIALECT_ENUMS_TD
include "mlir/IR/EnumAttr.td"

def Ax_Status : I32EnumAttr<"Status",
  "status code for func return status",[
    I32EnumAttrCase<"Success", 0>,
    I32EnumAttrCase<"Error", 1>,
    I32EnumAttrCase<"InvalidArgs", 2>
  ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::astl";
}

def Ax_Layout : I32EnumAttr<"Layout",
  "layout of the tensor",[
    I32EnumAttrCase<"NCHW", 0>,
    I32EnumAttrCase<"NHWC", 1>,
    I32EnumAttrCase<"Matrix", 2>,
    I32EnumAttrCase<"Vector", 3>,
    I32EnumAttrCase<"Scalar", 4>,
    I32EnumAttrCase<"C128", 5>,
  ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::astl";
}

#endif
