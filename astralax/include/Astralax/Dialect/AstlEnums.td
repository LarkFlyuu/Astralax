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

#endif
