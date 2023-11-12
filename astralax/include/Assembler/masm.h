#ifndef __ASTL_MASM_H__
#define __ASTL_MASM_H__
#include <string>
#include <vector>
#include "common.h"
#include "Astralax/Dialect/AstlOps.h"

namespace astl {
namespace masm {

typedef struct {
  std::string name;
  DataType dtype;
  Layout layout;
  Dims dims;
} Tensor;
using Tensors = std::vector<Tensor>;

typedef struct {
  OpType op;
  Tensor output;
  Tensors inputs;
  OpAttr attr;
} MasmOp;
using MasmOpList = std::vector<MasmOp>;

}
}

#endif // __ASTL_MASM_H__