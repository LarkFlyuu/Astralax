# Astralax

## Build LLVM

### clone
```sh
git clone https://github.com/llvm/llvm-project.git
```

### create build dir
```sh
mkdir llvm-project/build
cd llvm-project/build

git checkout 576b184d6e3b633f51b908b61ebd281d2ecbf66f

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang-16 \
   -DCMAKE_CXX_COMPILER=clang++-16 \
   -DLLVM_USE_LINKER=lld

ninja
```

## Build Astralax

```sh
# Clone
git clone https://github.com/LarkFlyuu/Astralax.git

mkdir build && cd build

# Configure Build
cmake -G Ninja .. \
   -DCMAKE_BUILD_TYPE=Debug \
   -DMLIR_DIR=/data/llvm-project/build/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=/data/llvm-project/build/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang-16 \
   -DCMAKE_CXX_COMPILER=clang++-16

# Build
cmake --build .

```
