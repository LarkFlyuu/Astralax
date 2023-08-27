get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRExecutionEngine
        MLIRIR
        MLIRJitRunner
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRToLLVMIRTranslationRegistration
        MLIRParser
        MLIRTargetLLVMIRExport
        MLIRSupport
        MLIROptLib
        MLIRASTL
        astl_c_runner_utils
        )

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(astl-run
  MLIRBench.cpp
  astl-run.cpp)

llvm_update_compile_flags(astl-run)

target_link_libraries(astl-run PRIVATE ${LIBS})

if (ASTL_GPU MATCHES "cuda")
  set(ASTL_GPU_LINK_FLAGS
      ${ASTL_GPU_LINK_FLAGS}
      -lmlir_cuda_runtime
    )
endif()

if (ASTL_GPU MATCHES "vulkan")
  set(ASTL_GPU_LINK_FLAGS
      ${ASTL_GPU_LINK_FLAGS}
      -lvulkan-runtime-wrappers
    )
endif()

message(STATUS "ASTL libraries at: ${CMAKE_BINARY_DIR}/lib")
message(STATUS "MLIR libraries at: ${LLVM_LIBRARY_DIR}")
# Add ASTL/MLIR library path so we don't need --shared-libs on every call
target_link_options(astl-run PRIVATE
  -Wl,--no-as-needed
  -L${CMAKE_BINARY_DIR}/lib
  -lastl_c_runner_utils
  -L${LLVM_LIBRARY_DIR}
  -lmlir_c_runner_utils
  -lmlir_runner_utils
  -lmlir_async_runtime
  ${ASTL_GPU_LINK_FLAGS}
  -lomp
  -Wl,--as-needed
)

install(TARGETS astl-run RUNTIME DESTINATION bin)
