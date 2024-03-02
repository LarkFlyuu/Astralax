//===--------------------------------
// Define common Attrs
//===--------------------------------
#ifndef ASTL_DIALECT_ASTL_ATTRS_TD
#define ASTL_DIALECT_ASTL_ATTRS_TD

include "Dialect/Enums.td"

class Ax_Attr<Dialect dialect, string name, list<Trait> traits = []>
  : AttrDef<dialect, name, traits> {
  let mnemonic = name;
}

class Ax_ArrayAttr<Dialect dialect, string arrayName, string eltName, list<Trait> traits = []> :
  ArrayOfAttr<dialect, arrayName, arrayName, eltName, traits>;

#endif
