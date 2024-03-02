#ifndef ASTL_COMMON_H
#define ASTL_COMMON_H
#include <vector>
#include <unordered_map>

#define LOGE(...) printf(__VA_ARGS__)

#define ASSERT_THROW(x, y)                                             \
  {                                                                    \
    const std::string &funcName = y;                                   \
    if (!(x)) {                                                        \
      llvm::report_fatal_error(llvm::StringRef("Error: " + funcName)); \
    }                                                                  \
  }

using Dims = std::vector<uint32_t>;

typedef struct {
  std::vector<uint8_t> kernel_size;
  std::vector<uint8_t> stride;
  std::vector<uint8_t> padding;
  std::vector<uint8_t> dilation;
  std::unordered_map<std::string, int64_t> extra_attrs;
} OpAttr;

#endif // ASTL_COMMON_H