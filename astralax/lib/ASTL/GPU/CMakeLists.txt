add_mlir_library(ASTLGPU
  Utils.cpp
  GpuPipeline.cpp
  GpuConversion.cpp
  GpuToCuda.cpp
  GpuToVulkan.cpp
  GPUToSPIRVPass.cpp
  SetSPIRVCapabilities.cpp
  SetSPIRVAbiAttribute.cpp
  GpuVulkanAbi.cpp
  LinalgToGpu.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ASTL

  LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRSCFToGPU
    MLIRSCFToSPIRV
    MLIRMathToSPIRV
    MLIRControlFlowToSPIRV
    MLIRMemRefTransforms
)

if (ASTL_GPU MATCHES "cuda")
  target_compile_definitions(obj.ASTLGPU
    PRIVATE
    ASTL_CUDA_ENABLE=1
  )

  target_link_libraries(ASTLGPU
    PUBLIC
    MLIRGPUToNVVMTransforms
  )
endif()

if (ASTL_GPU MATCHES "vulkan")
  target_compile_definitions(obj.ASTLGPU
    PRIVATE
    ASTL_VULKAN_ENABLE=1
  )

  target_link_libraries(ASTLGPU
    PUBLIC
    MLIRSPIRVDialect
    MLIRSPIRVTransforms
    MLIRGPUToVulkanTransforms
    MLIRGPUToSPIRV
  )
endif()
