
#ifndef ASTRALAX_ONNX_CONVERTER_H
#define ASTRALAX_ONNX_CONVERTER_H
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include "ONNX/onnx.pb.h"

#include "Dialect/HLD/HLDOps.h"
#include "Utils/MLIRHelper.h"
#include "common.h"

using namespace mlir;
namespace astl{
namespace hld {

struct BuilderInfo {
  std::shared_ptr<OpBuilder> builder;
  std::unordered_map<std::string, Value> TensorValue;
  std::unordered_map<std::string, onnx::TensorProto> Constants;
  Value noneValue;
  RankedTensorType outputType;
};
typedef struct BuilderInfo BuilderInfo;

class ONNXConverter {
public:
  ONNXConverter(ModuleOp& moduleOp) : 
    moduleOp(moduleOp), builder(moduleOp.getContext()) {
    builderInfo.builder = std::make_shared<OpBuilder>(builder);
  }
  virtual ~ONNXConverter() = default;
  void onnx2mlir(const std::string& srcOnnxFile);
 
private:
  ModuleOp moduleOp;
  OpBuilder builder;
  func::FuncOp mainFunc;
  Value noneValue;

  BuilderInfo builderInfo;

  typedef void (*addNodeFunc)(const onnx::NodeProto& node, BuilderInfo& builderInfo);
  
  void loadOnnxModel(const std::string& srcOnnxFile, onnx::ModelProto& model);
  void loadConstants(const onnx::GraphProto& graph);
  void loadConstant(const onnx::NodeProto& node);
  void loadValueInfos(const onnx::GraphProto& graph);
  void loadGraphNodes(const onnx::GraphProto& graph);
  std::vector<Value> loadGraphInput(const onnx::GraphProto& graph);
  std::vector<Value> loadGraphOutput(const onnx::GraphProto& graph);

  RankedTensorType getTensorType(const onnx::TypeProto& type);
  RankedTensorType getTensorType(const onnx::TensorProto& tensor);

  std::unordered_map<std::string, onnx::ValueInfoProto> ValueInfoMap;
  std::unordered_map<std::string, onnx::TensorProto> Constants;
  std::unordered_map<std::string, std::pair<astl::OpType, addNodeFunc>> AddNodeFuncs;
  void RegisterAddNodeFunc();
  
};

}  // namespace hld
}  // namespace astl

#endif  // ASTRALAX_ONNX_CONVERTER_H