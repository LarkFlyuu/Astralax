add_subdirectory(Interfaces)

add_mlir_dialect(HldOps hld)

set(LLVM_TARGET_DEFINITIONS HldOps.td)
mlir_tablegen(HldEnums.h.inc -gen-enum-decls)
mlir_tablegen(HldEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(HldAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(HldAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRHldAttrIncGen)
