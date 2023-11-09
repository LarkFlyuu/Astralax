//===--------------------------------
// Define common types
//===--------------------------------
#ifndef ASTL_MLIR_ASTL_TYPES_TD
#define ASTL_MLIR_ASTL_TYPES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

def ATensor : AnyTypeOf<[AnyTensor]>;
def ATensorOrNone : AnyTypeOf<[ATensor, NoneType]>;

class Ax_Type<Dialect dialect, string name, string attrMnemonic, list<Trait> traits = []>
  : TypeDef<dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // ASTL_MLIR_ASTL_TYPES_TD