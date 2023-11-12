
#ifndef __ONNX_CONVERTER_H__
#define __ONNX_CONVERTER_H__
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include "ONNX/onnx.pb.h"

#include "Astralax/Dialect/AstlOps.h"
#include "Astralax/Helper/Helper.h"
#include "common.h"

using namespace mlir;
namespace astl {
class ONNXConverter {
public:
  ONNXConverter(ModuleOp& moduleOp) : 
    moduleOp(moduleOp), builder(moduleOp.getContext()) {}
  virtual ~ONNXConverter() = default;
  
  void onnx2mlir(const std::string& srcOnnxFile);

  typedef void (*addNodeFunc)(const onnx::NodeProto& node, OpBuilder& builder,
                              std::unordered_map<std::string, Value>& TensorValue,
                              RankedTensorType& outputType);
 
private:
  ModuleOp moduleOp;
  func::FuncOp mainFunc;
  OpBuilder builder;
  
  void loadOnnxModel(const std::string& srcOnnxFile, onnx::ModelProto& model);
  void loadConstants(const onnx::GraphProto& graph);
  void loadValueInfos(const onnx::GraphProto& graph);
  void loadGraphNodes(const onnx::GraphProto& graph);
  std::vector<Value> loadGraphInput(const onnx::GraphProto& graph);
  std::vector<Value> loadGraphOutput(const onnx::GraphProto& graph);

  RankedTensorType getTensorType(const onnx::TypeProto& type);
  RankedTensorType getTensorType(const onnx::TensorProto& tensor);

  std::unordered_map<std::string, onnx::ValueInfoProto> ValueInfoMap;
  std::unordered_map<std::string, Value> TensorValue;
  std::unordered_map<std::string, std::pair<OpType, addNodeFunc>> AddNodeFuncs;
  void RegisterAddNodeFunc();
  
};

}  // namespace astl

#endif  // __ONNX_CONVERTER_H__