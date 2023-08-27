add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AstlCompiler)
add_public_tablegen_target(ASTLCompilerPassIncGen)
add_mlir_doc(Passes AstlCompilerPasses ./ -gen-pass-doc)
