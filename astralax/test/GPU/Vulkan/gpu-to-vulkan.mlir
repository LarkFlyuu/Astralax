// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-to-vulkan -split-input-file | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_add(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %0 = gpu.block_id x
      %1 = memref.load %arg0[%0] : memref<8xf32>
      %2 = memref.load %arg1[%0] : memref<8xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<8xf32>
      gpu.return
    }
  }

  func.func @entry() {
    %arg0 = memref.alloc() : memref<8xf32>
    %arg1 = memref.alloc() : memref<8xf32>
    %arg2 = memref.alloc() : memref<8xf32>
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = arith.constant 2 : i32
    %value0 = arith.constant 0.0 : f32
    %value1 = arith.constant 1.1 : f32
    %value2 = arith.constant 2.2 : f32
    %arg3 = memref.cast %arg0 : memref<8xf32> to memref<?xf32>
    %arg4 = memref.cast %arg1 : memref<8xf32> to memref<?xf32>
    %arg5 = memref.cast %arg2 : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%arg3, %value1) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%arg4, %value2) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%arg5, %value0) : (memref<?xf32>, f32) -> ()

    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@kernel_add
        blocks in (%cst8, %cst1, %cst1) threads in (%cst1, %cst1, %cst1)
        args(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>)
    %arg6 = memref.cast %arg5 : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%arg6) : (memref<*xf32>) -> ()

    memref.dealloc %arg0 : memref<8xf32>
    memref.dealloc %arg1 : memref<8xf32>
    memref.dealloc %arg2 : memref<8xf32>

    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}

// CHECK: module attributes {gpu.container_module, spirv.target_env =
// CHECK-LABEL: func.func @entry
// CHECK:         call @vulkanLaunch({{.*}}) {spirv_blob = "
// CHECK:       }
// CHECK: func.func private @vulkanLaunch