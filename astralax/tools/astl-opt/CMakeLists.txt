get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRToLLVMIRTranslationRegistration
        MLIROptLib
        MLIRASTL
        astl_c_runner_utils
        )
add_llvm_executable(astl-opt astl-opt.cpp)

llvm_update_compile_flags(astl-opt)
target_link_libraries(astl-opt PRIVATE ${LIBS})
install(TARGETS astl-opt RUNTIME DESTINATION bin)
