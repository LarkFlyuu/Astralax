# Astralax

git clone https://github.com/LarkFlyuu/Astralax.git

mkdir build && cd build

cmake -G Ninja .. \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DMLIR_DIR=/data/share/llvm-project/build/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=/data/share/llvm-project/build/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DUSE_OpenMP=False

cmake --build .
