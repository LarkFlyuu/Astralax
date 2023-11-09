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
};

}  // namespace OpTrait
}  // namespace mlir

#endif
