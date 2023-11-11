#ifndef ASTL_COMMON_TRAIT_H
#define ASTL_COMMON_TRAIT_H

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class CommonTrait : public TraitBase<ConcreteType, CommonTrait> {
 public:
  std::string getNodeName() {
    auto *concrete = static_cast<ConcreteType *>(this);
    auto loc = concrete->getLoc().template dyn_cast<::mlir::NameLoc>();
    if (loc == nullptr) return "unknown";
    return loc.getName().str();
  }

  size_t getOpIndex() {
    auto *concrete = static_cast<ConcreteType *>(this);
    auto *parent = concrete->getParentOp();
    if (parent == nullptr) return 0;
    auto *region = concrete->getParentRegion();
    if (region == nullptr) return 0;
    auto &ops = region->getOps();
    for (size_t i = 0; i < ops.size(); ++i) {
      if (ops[i] == parent) return i;
    }
    return 0;
  }
};

}  // namespace OpTrait
}  // namespace mlir

#endif
