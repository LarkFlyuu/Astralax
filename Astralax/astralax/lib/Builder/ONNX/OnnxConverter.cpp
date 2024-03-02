#include "OnnxConverter.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

using namespace mlir;
namespace astl {
namespace hld {
static const std::unordered_map<std::string, astl::OpType> OnnxOpTypeMap = {
  {"Div", astl::OpType::Div},
  {"Add", astl::OpType::Add},
  {"Mul", astl::OpType::Mul},
  {"Max", astl::OpType::Max},
  {"Min", astl::OpType::Min},
  {"Mean", astl::OpType::Mean},
};

static void addEltwiseNode(const onnx::NodeProto& node, OpBuilder& builder,
                           std::unordered_map<std::string, Value>& TensorValue,
                           std::unordered_map<std::string, onnx::TensorProto> Constants,
                           RankedTensorType& outputType, Value& noneValue) {
  ASSERT_THROW(node.input_size() == 2, "invalid div operands");

  const std::string& inputName = node.input(0);
  auto input = TensorValue[inputName];

  std::vector<Value> operands = {input};
  std::vector<NamedAttribute> attrs;
  const std::string& eltName = node.input(1);
  if (TensorValue.find(eltName) != TensorValue.end()) {
    operands.emplace_back(TensorValue.at(eltName));
  } else {
    operands.emplace_back(noneValue);
    auto tensor = Constants[eltName];
    auto raw_data = tensor.raw_data();
    std::vector<float_t> scalar(raw_data.size() / sizeof(float_t));
    std::memcpy(scalar.data(), raw_data.data(), raw_data.size());
    attrs.push_back({builder.getStringAttr("scalar"), builder.getF32FloatAttr(scalar[0])});
  }

  const std::string& name = node.name();
  const std::string& opTypeName = node.op_type();
  auto opType = OnnxOpTypeMap.at(opTypeName);

  Value result;
  switch (opType) {
    case OpType::Div:
      result = builder.create<DivOp>(getLoc(builder.getContext(), name),
        outputType, operands, attrs).getResult();
      break;
    case OpType::Add:
      result = builder.create<AddOp>(getLoc(builder.getContext(), name),
        outputType, operands, attrs).getResult();
      break;
    case OpType::Max:
      result = builder.create<MaxOp>(getLoc(builder.getContext(), name),
        outputType, operands, attrs).getResult();
      break;
    case OpType::Min:
      result = builder.create<MinOp>(getLoc(builder.getContext(), name),
        outputType, operands, attrs).getResult();
      break;
    case OpType::Mean:
      result = builder.create<MeanOp>(getLoc(builder.getContext(), name),
        outputType, operands, attrs).getResult();
      break;
    default:
      llvm::report_fatal_error(llvm::StringRef("invalid op type: " + opTypeName));
  }
  TensorValue[node.output(0)] = result;
}

static void addConvNode(const onnx::NodeProto& node, OpBuilder& builder,
                        std::unordered_map<std::string, Value>& TensorValue,
                        std::unordered_map<std::string, onnx::TensorProto> Constants,
                        RankedTensorType& outputType, Value& noneValue) {
//   std::vector<int64_t> kernel_size = get_node_attr_ai(onnx_node, "kernel_shape");
//   if (kernel_size.size() == 1) {
//     LOGE("node:%s conv 1d is not support now.\n", node->name.c_str());
//     return XPRT_ERROR;
//   }
//   struct Conv2DParam conv_param;
//   node->op_param = conv_param.Copy();
//   Conv2DParam* param_ptr = dynamic_cast<Conv2DParam*>(node->op_param.get());
//   param_ptr->kernel_size = kernel_size;
//   param_ptr->dilation = get_node_attr_ai(onnx_node, "dilations");
//   param_ptr->stride = get_node_attr_ai(onnx_node, "strides");
//   param_ptr->groups = get_node_attr_i(onnx_node, "group", 1);
//   std::vector<int64_t> padding = get_node_attr_ai(onnx_node, "pads");
//   XPRT_ASSERT(padding.size() == 4)
//   XPRT_ASSERT(padding[0] == padding[2])
//   XPRT_ASSERT(padding[1] == padding[3])
//   param_ptr->padding.push_back(padding[0]);  // h
//   param_ptr->padding.push_back(padding[1]);  // w

//   std::string auto_pad = get_node_attr_s(onnx_node, "auto_pad");
//   if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
//     LOGE("node %s ,pad mode %s not support.\n", node->name.c_str(), auto_pad.c_str());
//   }

//   param_ptr->bias_term = onnx_node.input_size() == 3 ? 1 : 0;
//   /* get weight of conv node*/
//   Tensor* weight = node->input_tensor[1];
//   /* onnx hide the output channel in weight .. */
//   param_ptr->out_channels = weight->dims[0];
//   param_ptr->in_channels = weight->dims[1] * param_ptr->groups;
//   param_ptr->winograd = is_winograd(param_ptr->kernel_size, param_ptr->stride, param_ptr->dilation, param_ptr->groups);

//   return XPRT_OK;
}

static void addConvLSTMNode(const onnx::NodeProto& node, OpBuilder& builder,
                            std::unordered_map<std::string, Value>& TensorValue,
                            std::unordered_map<std::string, onnx::TensorProto> Constants,
                            RankedTensorType& outputType, Value& noneValue) {
//   struct ConvLstmParam convlstm_param;
//   node->op_param = convlstm_param.Copy();
//   ConvLstmParam* param_ptr = dynamic_cast<ConvLstmParam*>(node->op_param.get());
//   param_ptr->hidden_c = get_node_attr_i(onnx_node, "hidden_c", 1);
//   param_ptr->in_c = get_node_attr_i(onnx_node, "in_c", 1);
//   param_ptr->repeat = get_node_attr_i(onnx_node, "repeat", 1);
//   for (int i = 0; i < 2; i++) {
//     param_ptr->padding.push_back(get_node_attr_i(onnx_node, "pad", 1));
//     param_ptr->stride.push_back(get_node_attr_i(onnx_node, "stride", 1));
//     param_ptr->kernel_size.push_back(get_node_attr_i(onnx_node, "kernel", 1));
//   }
//   param_ptr->bias_term = onnx_node.input_size() == 3 ? 1 : 0;
//   std::vector<int64_t> dilation = {1,1};
//   param_ptr->winograd = is_winograd(param_ptr->kernel_size, param_ptr->stride, dilation, 1);
//   /* get weight of conv node*/
//   Tensor* weight = node->input_tensor[2];
//   /* onnx hide the output channel in weight .. */
//   param_ptr->out_channels = weight->dims[0];
//   param_ptr->in_channels = weight->dims[1];

//   return XPRT_OK;
}

static void addPadNode(const onnx::NodeProto& node, OpBuilder& builder,
                         std::unordered_map<std::string, Value>& TensorValue,
                         std::unordered_map<std::string, onnx::TensorProto> Constants,
                         RankedTensorType& outputType, Value& noneValue) {
//   struct PadParam pad_param;
//   node->op_param = pad_param.Copy();
//   PadParam* param_ptr = dynamic_cast<PadParam*>(node->op_param.get());

//   for (int k = 0; k < onnx_node.attribute_size(); k++) {
//     const onnx::AttributeProto& attr = onnx_node.attribute(k);
//     if (attr.name() == "mode") {
//       if (attr.s() == "constant") {
//         param_ptr->mode = "constant";
//       } else if (attr.s() == "edge") {
//         param_ptr->mode = "replicate";
//       } else {
//         param_ptr->mode = "refelct";
//       }
//     }
//     if (attr.name() == "pads") {
//       param_ptr->pads.push_back(attr.ints(3));  // left
//       param_ptr->pads.push_back(attr.ints(7));  // right
//       param_ptr->pads.push_back(attr.ints(2));  // top
//       param_ptr->pads.push_back(attr.ints(6));  // bottom
//       param_ptr->pads.push_back(attr.ints(1));  // front
//       param_ptr->pads.push_back(attr.ints(5));  // behind
//     }
//     if (attr.name() == "value") {
//       param_ptr->value = attr.f();
//     }
//   }
//   if (param_ptr->pads.size() == 0) {
//     std::vector<int64_t> add_constant = get_node_attr_from_input_ai(attr_const_tensors[onnx_node.input(1)]);
//     if (add_constant.size() == 0) {
//       if (node->input_tensor.size() > 1) {
//         int64_t* pad_ptr = node->input_tensor[1]->data_ptr<int64_t>();
//         param_ptr->pads = {pad_ptr[3], pad_ptr[7], pad_ptr[2], pad_ptr[6], pad_ptr[1], pad_ptr[5]};
//         graph.remove_tensor(node->input_tensor[1]);
//         node->input_tensor.pop_back();
//       }
//     } else {
//       param_ptr->pads = {add_constant[3], add_constant[7], add_constant[2],
//                          add_constant[6], add_constant[1], add_constant[5]};
//     }
//   }
//   if (param_ptr->pads.size() == 0) {
//     LOGE("pads.size() == 0");
//   }
//   return XPRT_OK;
}

static void addReluNode(const onnx::NodeProto& node, OpBuilder& builder,
                        std::unordered_map<std::string, Value>& TensorValue,
                        std::unordered_map<std::string, onnx::TensorProto> Constants,
                        RankedTensorType& outputType, Value& noneValue) {
//   struct ReluParam relu_param;
//   node->op_param = relu_param.Copy();
//   ReluParam* param_ptr = dynamic_cast<ReluParam*>(node->op_param.get());
//   param_ptr->negative_slope = 0.f;
//   return XPRT_OK;
}

static void addSliceNode(const onnx::NodeProto& node, OpBuilder& builder,
                         std::unordered_map<std::string, Value>& TensorValue,
                         std::unordered_map<std::string, onnx::TensorProto> Constants,
                         RankedTensorType& outputType, Value& noneValue) {
//   struct SliceParam slice_param;
//   node->op_param = slice_param.Copy();
//   SliceParam* param_ptr = dynamic_cast<SliceParam*>(node->op_param.get());

//   if (onnx_node.input_size() == 1) {
//     param_ptr->axes = get_node_attr_ai(onnx_node, "axes");
//     param_ptr->ends = get_node_attr_ai(onnx_node, "ends");
//     param_ptr->starts = get_node_attr_ai(onnx_node, "starts");
//     if (param_ptr->axes.size() == 0) {
//       int value = 0;
//       for (int i = 0; i < param_ptr->starts.size(); i++) {
//         param_ptr->axes.push_back(value);
//         value += 1;
//       }
//     }
//     std::vector<int64_t> steps(param_ptr->starts.size(), 1);
//     param_ptr->steps = steps;
//   } else if (onnx_node.input_size() == 3) {
//     param_ptr->starts = load_constant_tensor_ai(onnx_node.input(1), attr_const_tensors);
//     param_ptr->ends = load_constant_tensor_ai(onnx_node.input(2), attr_const_tensors);
//     int value = 0;
//     for (int i = 0; i < param_ptr->starts.size(); i++) {
//       param_ptr->axes.push_back(value);
//       value += 1;
//     }
//     std::vector<int64_t> steps(param_ptr->starts.size(), 1);
//     param_ptr->steps = steps;
//   } else if (onnx_node.input_size() == 5) {
//     param_ptr->starts = load_constant_tensor_ai(onnx_node.input(1), attr_const_tensors);
//     param_ptr->ends = load_constant_tensor_ai(onnx_node.input(2), attr_const_tensors);
//     param_ptr->axes = load_constant_tensor_ai(onnx_node.input(3), attr_const_tensors);
//     param_ptr->steps = load_constant_tensor_ai(onnx_node.input(4), attr_const_tensors);
//   } else {
//     LOGE("NotImplementation!\n");
//   }
//   return XPRT_OK;
}

static void addTransposeNode(const onnx::NodeProto& node, OpBuilder& builder,
                             std::unordered_map<std::string, Value>& TensorValue,
                             std::unordered_map<std::string, onnx::TensorProto> Constants,
                             RankedTensorType& outputType, Value& noneValue) {
  // ASSERT_THROW(node.input_size() == 1, "invalid transpose operands");
  
  // const std::string& lhsName = node.input(0);

  // auto input = TensorValue[lhsName];

  // ASSERT_THROW(node.attribute_size() == 1, "transpose perm not found");
  // const onnx::AttributeProto& attr = node.attribute(0);
  // std::vector<int64_t> perm(attr.ints_size());
  // for (int i = 0; i < attr.ints_size(); i++)
  //   perm[i] = attr.ints(i);

  // const std::string& name = node.name();
  // auto op = builder.create<astl::TransposeOp>(getLoc(builder.getContext(), name),
  //   outputType, input, builder.getI64ArrayAttr(perm));
  
  // const std::string& outputName = node.output(0);
  // TensorValue[outputName] = op.getResult();
}

static void addReshapeNode(const onnx::NodeProto& node, OpBuilder& builder,
                           std::unordered_map<std::string, Value>& TensorValue,
                           std::unordered_map<std::string, onnx::TensorProto> Constants,
                           RankedTensorType& outputType, Value& noneValue) {
  // ASSERT_THROW(node.input_size() == 2, "invalid reshape operands");
  
  // const std::string& inputName = node.input(0);
  // auto input = TensorValue[inputName];
  // if (input == nullptr) return; // todo: assert it's not null

  // const std::string& dimsName = node.input(1);
  // auto tensor = Constants[dimsName];

  // auto raw_data = tensor.raw_data();
  // std::vector<int64_t> dims(raw_data.size() / sizeof(int64_t));
  // std::memcpy(dims.data(), raw_data.data(), raw_data.size());

  // const std::string& name = node.name();
  // auto reshapeOp = builder.create<astl::ReshapeOp>(
  //   getLoc(builder.getContext(), name), outputType, input,
  //   builder.getI64ArrayAttr(dims));
  
  // const std::string& outputName = node.output(0);
  // TensorValue[outputName] = reshapeOp.getResult();
}

static void addUpsampleNode(const onnx::NodeProto& node, OpBuilder& builder,
                            std::unordered_map<std::string, Value>& TensorValue,
                            std::unordered_map<std::string, onnx::TensorProto> Constants,
                            RankedTensorType& outputType, Value& noneValue) {
//   struct UpsampleParam upsample_param;
//   node->op_param = upsample_param.Copy();
//   UpsampleParam* param_ptr = dynamic_cast<UpsampleParam*>(node->op_param.get());
//   if (onnx_node.input_size() == 1) {
//     param_ptr->scales = get_node_attr_af(onnx_node, "scales");
//   } else if (onnx_node.input_size() == 2 && node->input_tensor.size() == 1) {
//     param_ptr->scales = get_node_attr_from_input_af(attr_const_tensors[onnx_node.input(1)]);
//   } else if (onnx_node.input_size() == 2 && node->input_tensor.size() == 2) {
//     float* input1 = node->input_tensor[1]->data_ptr<float>();
//     for (int i = 0; i < node->input_tensor[1]->num_ele(); i++) {
//       param_ptr->scales.push_back(input1[i]);
//     }
//     node->input_tensor.pop_back();
//   } else if (onnx_node.input_size() == 3 && node->input_tensor.size() == 3) {
//     float* input1 = node->input_tensor[2]->data_ptr<float>();
//     for (int i = 0; i < node->input_tensor[2]->num_ele(); i++) {
//       param_ptr->scales.push_back(input1[i]);
//     }
//     node->input_tensor.pop_back();
//     node->input_tensor.pop_back();
//   } else if (onnx_node.input_size() == 3 && node->input_tensor.size() == 2) {
//     float* input1 = node->input_tensor[1]->data_ptr<float>();
//     for (int i = 0; i < node->input_tensor[1]->num_ele(); i++) {
//       param_ptr->scales.push_back(input1[i]);
//     }
//     node->input_tensor.pop_back();
//   } else if (node->input_tensor.size() > 3) {
//     if (node->input_tensor[2]->num_ele() != 0) {
//       float* input1 = node->input_tensor[2]->data_ptr<float>();
//       for (int i = 0; i < node->input_tensor[2]->num_ele(); i++) {
//         param_ptr->scales.push_back(input1[i]);
//       }
//     } else {
//       int64_t* input1 = node->input_tensor[3]->data_ptr<int64_t>();
//       for (int i = 0; i < node->input_tensor[3]->num_ele(); i++) {
//         param_ptr->sizes.push_back(input1[i]);
//       }
//     }
//     node->input_tensor.pop_back();
//     node->input_tensor.pop_back();
//     node->input_tensor.pop_back();
//   } else {
//     LOGE("Resize: No scales are read!\n");
//   }

//   param_ptr->mode = get_node_attr_s(onnx_node, "mode");
//   std::string transformation_mode = get_node_attr_s(onnx_node, "coordinate_transformation_mode");
//   if (transformation_mode == "align_corners") {
//     param_ptr->align_corners = true;
//   } else {
//     param_ptr->align_corners = false;
//   }

//   return XPRT_OK;
}

static void addConcatNode(const onnx::NodeProto& node, OpBuilder& builder,
                          std::unordered_map<std::string, Value>& TensorValue,
                          std::unordered_map<std::string, onnx::TensorProto> Constants,
                          RankedTensorType& outputType, Value& noneValue) {
//   struct ConcatParam concat_param;
//   node->op_param = concat_param.Copy();
//   ConcatParam* param_ptr = dynamic_cast<ConcatParam*>(node->op_param.get());
//   const onnx::AttributeProto& attr = onnx_node.attribute(0);
//   const auto& attributeName = attr.name();
//   if (attributeName == "axis") {
//     param_ptr->axis = attr.i();
//   }
//   return XPRT_OK;
}

static void addSplitNode(const onnx::NodeProto& node, OpBuilder& builder,
                         std::unordered_map<std::string, Value>& TensorValue,
                         std::unordered_map<std::string, onnx::TensorProto> Constants,
                         RankedTensorType& outputType, Value& noneValue) {
//   struct SplitParam split_param;
//   node->op_param = split_param.Copy();
//   SplitParam* param_ptr = dynamic_cast<SplitParam*>(node->op_param.get());
//   const onnx::AttributeProto& attr = onnx_node.attribute(0);
//   const auto& attributeName = attr.name();
//   return XPRT_OK;
}

static void addUnsqueezeNode(const onnx::NodeProto& node, OpBuilder& builder,
                             std::unordered_map<std::string, Value>& TensorValue,
                             std::unordered_map<std::string, onnx::TensorProto> Constants,
                             RankedTensorType& outputType, Value& noneValue) {
//   struct UnsqueezeParam unsqueeze_param;
//   node->op_param = unsqueeze_param.Copy();
//   UnsqueezeParam* param_ptr = dynamic_cast<UnsqueezeParam*>(node->op_param.get());
//   const onnx::AttributeProto& attr = onnx_node.attribute(0);
//   const auto& attributeName = attr.name();
//   if (attributeName == "axes") {
//     param_ptr->axes = get_node_attr_ai(onnx_node, "axes");
//   }
//   if (param_ptr->axes.size() == 0) {
//     LOGE("can not read axes!\n");
//   }
//   return XPRT_OK;
}

static void addGatherNode(const onnx::NodeProto& node, OpBuilder& builder,
                          std::unordered_map<std::string, Value>& TensorValue,
                          std::unordered_map<std::string, onnx::TensorProto> Constants,
                          RankedTensorType& outputType, Value& noneValue) {
//   struct GatherParam gather_param;
//   node->op_param = gather_param.Copy();
//   GatherParam* param_ptr = dynamic_cast<GatherParam*>(node->op_param.get());
//   const onnx::AttributeProto& attr = onnx_node.attribute(0);
//   const auto& attributeName = attr.name();
//   if (attributeName == "axis") {
//     param_ptr->axis = attr.i();
//   }
//   if (onnx_node.input_size() == 2 && node->num_input() == 2) {
//     int64_t* data_ptr = node->input_tensor[1]->data_ptr<int64_t>();
//     param_ptr->indices = data_ptr[0];
//     node->input_tensor.pop_back();
//     // graph.remove_tensor(node->input_tensor[1]);
//   }
//   return XPRT_OK;
}

static void addGlobalAvgPoolNode(const onnx::NodeProto& node, OpBuilder& builder,
                                 std::unordered_map<std::string, Value>& TensorValue,
                                 std::unordered_map<std::string, onnx::TensorProto> Constants,
                                 RankedTensorType& outputType, Value& noneValue) {
//   struct AdaptiveAvgPool2DParam global_avgpool_param;
//   node->op_param = global_avgpool_param.Copy();
//   AdaptiveAvgPool2DParam* param_ptr = dynamic_cast<AdaptiveAvgPool2DParam*>(node->op_param.get());
//   // onnx->globalAveragePooling -> nn.AdaptiveAvgPool2D output_size = {1, 1}
//   param_ptr->output_size = {1, 1};
//   return XPRT_OK;
}

static void addMaxPoolNode(const onnx::NodeProto& node, OpBuilder& builder,
                           std::unordered_map<std::string, Value>& TensorValue,
                           std::unordered_map<std::string, onnx::TensorProto> Constants,
                           RankedTensorType& outputType, Value& noneValue) {
//   struct Pool2DParam maxpool_param;
//   node->op_param = maxpool_param.Copy();
//   Pool2DParam* param_ptr = dynamic_cast<Pool2DParam*>(node->op_param.get());
//   //deprecated
//   //std::string auto_pad = get_node_attr_s(onnx_node, "auto_pad");
//   param_ptr->ceil_mode = get_node_attr_i(onnx_node, "ceil_mode", 0);
//   param_ptr->stride = get_node_attr_ai(onnx_node, "strides");
//   param_ptr->kernel_size = get_node_attr_ai(onnx_node, "kernel_shape");
//   ;
//   //{h0, w0, h1, w1} -> {h, w}
//   std::vector<int64_t> padding = get_node_attr_ai(onnx_node, "pads");
//   XPRT_ASSERT(padding[0] == padding[2]);
//   XPRT_ASSERT(padding[1] == padding[3]);
//   param_ptr->padding.push_back(padding[0]);
//   param_ptr->padding.push_back(padding[1]);
//   //opset 10/11/12
//   std::vector<int64_t> dilation = get_node_attr_ai(onnx_node, "dilations");
//   // opset 1/8
//   if (dilation.empty()) {
//     dilation = {1, 1};
//   }
//   param_ptr->dilation = dilation;

//   return XPRT_OK;
}

static void addGemmNode(const onnx::NodeProto& node, OpBuilder& builder,
                        std::unordered_map<std::string, Value>& TensorValue,
                        std::unordered_map<std::string, onnx::TensorProto> Constants,
                        RankedTensorType& outputType, Value& noneValue) {
//   struct LinearParam linear_param;
//   node->op_param = linear_param.Copy();
//   LinearParam* param_ptr = dynamic_cast<LinearParam*>(node->op_param.get());

//   float alpha = get_node_attr_f(onnx_node, "alpha", 1.f);
//   float beta = get_node_attr_f(onnx_node, "beta", 1.f);
//   int transA = get_node_attr_i(onnx_node, "transA", 0);
//   int transB = get_node_attr_i(onnx_node, "transB", 0);

//   /* torchscript(Linear) -> onnx(Gemm)
//   * Y = alpha * A(input0) * B(input1) + beta * C(input2)
//   * InnerProduct-like Y = alpha * T_B(Weight) * A + beta * C(Bias)
//   * must make sure alpha == 1.f && beta == 1.f && transA == 0 && transB == 1
//   */
//   XPRT_ASSERT(alpha == 1.f);
//   XPRT_ASSERT(beta == 1.f);
//   XPRT_ASSERT(transA == 0);
//   XPRT_ASSERT(transB == 1);
//   param_ptr->transpose_w = true;
//   param_ptr->transpose_i = false;
//   if (onnx_node.input_size() == 3) {
//     param_ptr->bias_term = 1;
//   }
//   // weights(oc, ic)
//   param_ptr->out_features = node->input_tensor[1]->dims[0];
//   param_ptr->in_features = node->input_tensor[1]->dims[1];
//   return XPRT_OK;
}

static void addBatchNormNode(const onnx::NodeProto& node, OpBuilder& builder,
                             std::unordered_map<std::string, Value>& TensorValue,
                             std::unordered_map<std::string, onnx::TensorProto> Constants,
                             RankedTensorType& outputType, Value& noneValue) {
//   struct BatchNorm2DParam batchnorm_param;
//   node->op_param = batchnorm_param.Copy();
//   BatchNorm2DParam* param_ptr = dynamic_cast<BatchNorm2DParam*>(node->op_param.get());
//   param_ptr->eps = get_node_attr_f(onnx_node, "epsilon", 1e-5f);
//   // onnx => scale(input1 not optinal) B(input2 not optional) mean(input3) var(input4)
//   param_ptr->affine = true;
//   return XPRT_OK;
}

static void addFlattenNode(const onnx::NodeProto& node, OpBuilder& builder,
                           std::unordered_map<std::string, Value>& TensorValue,
                           std::unordered_map<std::string, onnx::TensorProto> Constants,
                           RankedTensorType& outputType, Value& noneValue) {
//   struct FlattenParam flatten_param;
//   node->op_param = flatten_param.Copy();
//   FlattenParam* param_ptr = dynamic_cast<FlattenParam*>(node->op_param.get());
//   int axis = get_node_attr_i(onnx_node, "axis", 1);
//   // onnx Flattens the input tensor into a 2D matrix, torch is optional, so
//   //=>end_dim = -1
//   param_ptr->start_dim = axis;
//   param_ptr->end_dim = -1;
//   return XPRT_OK;
}

static void addReduceNode(const onnx::NodeProto& node, OpBuilder& builder,
                          std::unordered_map<std::string, Value>& TensorValue,
                          std::unordered_map<std::string, onnx::TensorProto> Constants,
                          RankedTensorType& outputType, Value& noneValue) {
//   struct ReduceOpParam reduce_param;
//   node->op_param = reduce_param.Copy();
//   ReduceOpParam* param_ptr = dynamic_cast<ReduceOpParam*>(node->op_param.get());
//   param_ptr->axes = get_node_attr_ai(onnx_node, "axes");
//   param_ptr->keepdims = get_node_attr_i(onnx_node, "keepdims", 1);
//   if (node->op_type() == OP_REDUCEMEAN) {
//     param_ptr->type = REDUCE_MEAN;
//   } else if (node->op_type() == OP_REDUCESUM) {
//     param_ptr->type = REDUCE_SUM;
//   } else {
//     LOGE("Not implementation!\n");
//   }
//   XPRT_ASSERT(param_ptr->axes.size() != 0);
//   return XPRT_OK;
}

static void addConvTransposeNode(const onnx::NodeProto& node, OpBuilder& builder,
                                 std::unordered_map<std::string, Value>& TensorValue,
                                 std::unordered_map<std::string, onnx::TensorProto> Constants,
                                 RankedTensorType& outputType, Value& noneValue) {
  
}

static void addMatMulNode(const onnx::NodeProto& node, OpBuilder& builder,
                          std::unordered_map<std::string, Value>& TensorValue,
                          std::unordered_map<std::string, onnx::TensorProto> Constants,
                          RankedTensorType& outputType, Value& noneValue) {
  // ASSERT_THROW(node.input_size() == 2, "invalid matmul operands");
  
  // const std::string& lhsName = node.input(0);
  // const std::string& rhsName = node.input(1);

  // auto lhs = TensorValue[lhsName];
  // auto rhs = TensorValue[rhsName];

  // if (lhs == nullptr || rhs == nullptr) return; // todo: assert it's not null

  // const std::string& name = node.name();
  // auto op = builder.create<astl::MatMulOp>(
  //   getLoc(builder.getContext(), name), outputType, lhs, rhs);
  
  // const std::string& outputName = node.output(0);
  // TensorValue[outputName] = op.getResult();
}

static void addInstNormNode(const onnx::NodeProto& node, OpBuilder& builder,
                            std::unordered_map<std::string, Value>& TensorValue,
                            std::unordered_map<std::string, onnx::TensorProto> Constants,
                            RankedTensorType& outputType, Value& noneValue) {
//   struct InstanceNormParam instancenorm_param;
//   node->op_param = instancenorm_param.Copy();
//   InstanceNormParam* param_ptr = dynamic_cast<InstanceNormParam*>(node->op_param.get());
//   param_ptr->eps = get_node_attr_f(onnx_node, "epsilon", 1e-5f);
//   // onnx => scale(input1 not optinal) B(input2 not optional) mean(input3) var(input4)
//   if (node->input_tensor.size() > 1) {
//     param_ptr->affine = true;
//   }
//   return XPRT_OK;
}

static void addAvgPoolNode(const onnx::NodeProto& node, OpBuilder& builder,
                           std::unordered_map<std::string, Value>& TensorValue,
                           std::unordered_map<std::string, onnx::TensorProto> Constants,
                           RankedTensorType& outputType, Value& noneValue) {
//   struct Pool2DParam avgpool_param;
//   node->op_param = avgpool_param.Copy();
//   Pool2DParam* param_ptr = dynamic_cast<Pool2DParam*>(node->op_param.get());
//   //deprecated
//   //std::string auto_pad = get_node_attr_s(onnx_node, "auto_pad");
//   param_ptr->ceil_mode = get_node_attr_i(onnx_node, "ceil_mode", 0);
//   param_ptr->stride = get_node_attr_ai(onnx_node, "strides");
//   param_ptr->kernel_size = get_node_attr_ai(onnx_node, "kernel_shape");;
//   param_ptr->dilation = {1, 1};
//   //{h0, w0, h1, w1} -> {h, w}
//   std::vector<int64_t> padding = get_node_attr_ai(onnx_node, "pads");
//   XPRT_ASSERT(padding[0] == padding[2]);
//   XPRT_ASSERT(padding[1] == padding[3]);
//   param_ptr->padding.push_back(padding[0]);
//   param_ptr->padding.push_back(padding[1]);

//   return XPRT_OK;
}

static void addActivationLutNode(const onnx::NodeProto& node, OpBuilder& builder,
                                 std::unordered_map<std::string, Value>& TensorValue,
                                 std::unordered_map<std::string, onnx::TensorProto> Constants,
                                 RankedTensorType& outputType, Value& noneValue) {
//   struct LutParam layer_param;
//   node->op_param = layer_param.Copy();
//   return XPRT_OK;
}

static void addSoftmaxNode(const onnx::NodeProto& node, OpBuilder& builder,
                           std::unordered_map<std::string, Value>& TensorValue,
                           std::unordered_map<std::string, onnx::TensorProto> Constants,
                           RankedTensorType& outputType, Value& noneValue) {
  // ASSERT_THROW(node.input_size() == 1, "invalid softmax operands");
  
  // const std::string& inputName = node.input(0);
  // auto input = TensorValue[inputName];
  // if (input == nullptr) return; // todo: assert it's not null

  // auto attr = node.attribute().begin().operator*();
  // uint64_t axis = attr.i();

  // const std::string& name = node.name();
  // auto reshapeOp = builder.create<astl::SoftmaxOp>(
  //   getLoc(builder.getContext(), name), outputType, input,
  //   builder.getI64IntegerAttr(axis));
  
  // const std::string& outputName = node.output(0);
  // TensorValue[outputName] = reshapeOp.getResult();
}
  
static void addClipNode(const onnx::NodeProto& node, OpBuilder& builder,
                        std::unordered_map<std::string, Value>& TensorValue,
                        std::unordered_map<std::string, onnx::TensorProto> Constants,
                        RankedTensorType& outputType, Value& noneValue) {
//   struct ClipParam param;
//   node->op_param = param.Copy();
//   ClipParam* param_ptr = dynamic_cast<ClipParam*>(node->op_param.get());
//   if (onnx_node.attribute_size() != 0) {
//     for (int i = 0; i < onnx_node.attribute_size(); ++i) {
//       const onnx::AttributeProto& attr = onnx_node.attribute(i);
//       const auto& attributeName = attr.name();
//       if (attributeName == "min") {
//         param_ptr->min = attr.f();
//       } else if (attributeName == "max") {
//         param_ptr->max = attr.f();
//       }
//     }
//   } else {
//     if (node->input_tensor.size() == 2) {
//       if (node->input_tensor[1]->name == onnx_node.input(1)) {
//         param_ptr->min = node->input_tensor[1]->data_ptr<float>()[0];
//       } else {
//         param_ptr->max = node->input_tensor[1]->data_ptr<float>()[0];
//       }

//     } else if (node->input_tensor.size() == 3) {
//       param_ptr->min = node->input_tensor[1]->data_ptr<float>()[0];
//       param_ptr->max = node->input_tensor[2]->data_ptr<float>()[0];
//     }
//   }

//   return XPRT_OK;
}

static void addSqueezeNode(const onnx::NodeProto& node, OpBuilder& builder,
                           std::unordered_map<std::string, Value>& TensorValue,
                           std::unordered_map<std::string, onnx::TensorProto> Constants,
                           RankedTensorType& outputType, Value& noneValue) {
//   struct UnsqueezeParam unsqueeze_param;
//   node->op_param = unsqueeze_param.Copy();
//   UnsqueezeParam* param_ptr = dynamic_cast<UnsqueezeParam*>(node->op_param.get());
//   //optional: axes
//   if (onnx_node.attribute_size() != 0) {
//     const onnx::AttributeProto& attr = onnx_node.attribute(0);
//     const auto& attributeName = attr.name();
//     if (attributeName == "axes") {
//       param_ptr->axes = get_node_attr_ai(onnx_node, "axes");
//     }
//   }

//   return XPRT_OK;
}

#define OP_PARSER_REGISTER(Type, Func) std::pair<OpType, addNodeFunc>(Type, Func)
void ONNXConverter::RegisterAddNodeFunc() {
  AddNodeFuncs["Add"] = OP_PARSER_REGISTER(OpType::Add, addEltwiseNode);
  AddNodeFuncs["Max"] = OP_PARSER_REGISTER(OpType::Max, addEltwiseNode);
  AddNodeFuncs["Min"] = OP_PARSER_REGISTER(OpType::Min, addEltwiseNode);
  AddNodeFuncs["Sub"] = OP_PARSER_REGISTER(OpType::Sub, addEltwiseNode);
  AddNodeFuncs["Pow"] = OP_PARSER_REGISTER(OpType::Pow, addEltwiseNode);
  AddNodeFuncs["Sqrt"] = OP_PARSER_REGISTER(OpType::Sqrt, addEltwiseNode);
  AddNodeFuncs["Div"] = OP_PARSER_REGISTER(OpType::Div, addEltwiseNode);
  AddNodeFuncs["Mul"] = OP_PARSER_REGISTER(OpType::Mul, addEltwiseNode);
  AddNodeFuncs["Conv"] = OP_PARSER_REGISTER(OpType::Conv, addConvNode);
  AddNodeFuncs["Pad"] = OP_PARSER_REGISTER(OpType::Pad, addPadNode);
  AddNodeFuncs["Relu"] = OP_PARSER_REGISTER(OpType::Relu, addReluNode);
  AddNodeFuncs["Slice"] = OP_PARSER_REGISTER(OpType::Slice, addSliceNode);
  AddNodeFuncs["Transpose"] = OP_PARSER_REGISTER(OpType::Transpose, addTransposeNode);
  AddNodeFuncs["Reshape"] = OP_PARSER_REGISTER(OpType::Reshape, addReshapeNode);
  AddNodeFuncs["Upsample"] = OP_PARSER_REGISTER(OpType::Upsample, addUpsampleNode);
  AddNodeFuncs["Resize"] = OP_PARSER_REGISTER(OpType::Upsample, addUpsampleNode);
  AddNodeFuncs["Concat"] = OP_PARSER_REGISTER(OpType::Concat, addConcatNode);
  AddNodeFuncs["BatchNormalization"] = OP_PARSER_REGISTER(OpType::BatchNormalization, addBatchNormNode);
  AddNodeFuncs["Flatten"] = OP_PARSER_REGISTER(OpType::Flatten, addFlattenNode);
  AddNodeFuncs["Split"] = OP_PARSER_REGISTER(OpType::Split, addSplitNode);
  AddNodeFuncs["ConvLSTM"] = OP_PARSER_REGISTER(OpType::LSTM, addConvLSTMNode);
  AddNodeFuncs["GlobalAveragePool"] = OP_PARSER_REGISTER(OpType::GlobalAveragePool, addGlobalAvgPoolNode);
  AddNodeFuncs["MaxPool"] = OP_PARSER_REGISTER(OpType::MaxPool, addMaxPoolNode);
  AddNodeFuncs["Gemm"] = OP_PARSER_REGISTER(OpType::Gemm, addGemmNode);
  AddNodeFuncs["Unsqueeze"] = OP_PARSER_REGISTER(OpType::Unsqueeze, addUnsqueezeNode);
  AddNodeFuncs["Gather"] = OP_PARSER_REGISTER(OpType::Gather, addGatherNode);
  AddNodeFuncs["ReduceMean"] = OP_PARSER_REGISTER(OpType::ReduceMean, addReduceNode);
  AddNodeFuncs["ReduceSum"] = OP_PARSER_REGISTER(OpType::ReduceSum, addReduceNode);
  AddNodeFuncs["ConvTranspose"] = OP_PARSER_REGISTER(OpType::ConvTranspose, addConvTransposeNode);
  AddNodeFuncs["MatMul"] = OP_PARSER_REGISTER(OpType::MatMul, addMatMulNode);
  AddNodeFuncs["InstanceNormalization"] = OP_PARSER_REGISTER(OpType::InstanceNormalization, addInstNormNode);
  AddNodeFuncs["AveragePool"] = OP_PARSER_REGISTER(OpType::AveragePool, addAvgPoolNode);
  AddNodeFuncs["Sigmoid"] = OP_PARSER_REGISTER(OpType::Sigmoid, addActivationLutNode);
  AddNodeFuncs["Tanh"] = OP_PARSER_REGISTER(OpType::Tanh, addActivationLutNode);
  AddNodeFuncs["Softmax"] = OP_PARSER_REGISTER(OpType::Softmax, addSoftmaxNode);
  AddNodeFuncs["Clip"] = OP_PARSER_REGISTER(OpType::Clip, addClipNode);
  AddNodeFuncs["Squeeze"] = OP_PARSER_REGISTER(OpType::Squeeze, addSqueezeNode);
}

void ONNXConverter::onnx2mlir(const std::string& srcOnnxFile) {
  /* initizalize module & main function */
  auto unknownLoc = UnknownLoc::get(moduleOp.getContext());
  mainFunc = builder.create<func::FuncOp>(
    unknownLoc, "main", builder.getFunctionType({}, {}));
  moduleOp.push_back(mainFunc);

  auto entryBlock = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  noneValue = builder.create<NoneOp>(unknownLoc, NoneType::get(builder.getContext()));
  
  onnx::ModelProto model;
  loadOnnxModel(srcOnnxFile, model);

  const onnx::GraphProto& graph = model.graph();

  auto inputs = loadGraphInput(graph);
  loadValueInfos(graph);
  loadConstants(graph);

  RegisterAddNodeFunc();
  loadGraphNodes(graph);

  auto outputs = loadGraphOutput(graph);

  llvm::SmallVector<Type> inputTypes{};
  std::for_each(inputs.begin(), inputs.end(), [&](Value val) {
    RankedTensorType type = val.getType().cast<RankedTensorType>();
    inputTypes.push_back(type);
  });
  llvm::SmallVector<Type> outputTypes{};
  std::for_each(outputs.begin(), outputs.end(), [&](Value val) {
    RankedTensorType type = val.getType().cast<RankedTensorType>();
    outputTypes.push_back(type);
  });
  mainFunc.setType(builder.getFunctionType(inputTypes, outputTypes));
  builder.create<func::ReturnOp>(unknownLoc, outputs);
}

void ONNXConverter::loadOnnxModel(const std::string& srcOnnxFile, onnx::ModelProto& model) {
  // std::ifstream ifs(srcOnnxFile, std::ios::in | std::ios::binary);
  // ASSERT_THROW(ifs.is_open(), "cannot open file: " + srcOnnxFile);

  // google::protobuf::io::IstreamInputStream input_stream(&ifs);
  // google::protobuf::io::CodedInputStream coded_input(&input_stream);

  // #if GOOGLE_PROTOBUF_VERSION >= 3011000
  //   coded_input.SetTotalBytesLimit(INT_MAX);
  // #else
  //   coded_input.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
  // #endif

  // bool ret = model.ParseFromCodedStream(&coded_input);

  // ifs.close();

  // ASSERT_THROW(ret, "parse onnx model failed");
}

void ONNXConverter::loadConstants(const onnx::GraphProto& graph) {
  // int tensorNum = graph.initializer_size();
  // for (int i = 0; i < tensorNum; i++) {
  //   const onnx::TensorProto& tensor = graph.initializer(i);
  //   std::string name = tensor.name();

  //   Constants[name] = tensor;
    
  //   RankedTensorType type = getTensorType(tensor);
  //   auto constOp = builder.create<astl::ConstantOp>(getLoc(builder.getContext(), name), type);

  //   TensorValue[name] = constOp.getResult();
  // }
}

void ONNXConverter::loadConstant(const onnx::NodeProto& node) {
//   const std::string& name = node.output(0);
//   const auto& attr = node.attribute().begin().operator*();
//   const auto& tensor = attr.t();
//   Constants[name] = tensor;
}

void ONNXConverter::loadValueInfos(const onnx::GraphProto& graph) {
//   int valueNum = graph.value_info_size();
//   for (int i = 0; i < valueNum; i++) {
//     const onnx::ValueInfoProto& val = graph.value_info(i);
//     const std::string name = val.name();
//     ValueInfoMap[name] = val;
//   }

//   for (int i = 0; i < graph.output_size(); i++) {
//     const onnx::ValueInfoProto& val = graph.output(i);
//     const std::string name = val.name();
//     ValueInfoMap[name] = val;
//   }
}

void ONNXConverter::loadGraphNodes(const onnx::GraphProto& graph) {
  // for (int i = 0; i < graph.node_size(); i++) {
  //   const onnx::NodeProto& node = graph.node(i);

  //   const std::string& op_type = node.op_type();
  //   if (op_type == "Constant") {
  //     loadConstant(node);
  //     continue;
  //   };

  //   ASSERT_THROW(AddNodeFuncs.find(op_type) != AddNodeFuncs.end(), 
  //                "cannot find operator: " + op_type);

  //   printf("- add operator: %s\n", op_type.c_str());

  //   std::string name = node.name();
  //   if (name.empty()) name = std::to_string(i);
    
  //   ASSERT_THROW(node.output_size() == 1, "only support single output");
  //   const std::string& output = node.output(0);
  //   ASSERT_THROW(ValueInfoMap.find(output) != ValueInfoMap.end(),
  //                "cannot find output: " + output);

  //   const onnx::ValueInfoProto& outputInfo = ValueInfoMap[output];
  //   const onnx::TypeProto& type = outputInfo.type();

  //   RankedTensorType outputType = getTensorType(type);

  //   auto& funcType = AddNodeFuncs[op_type];
  //   auto addNodeFunc = funcType.second;
  //   addNodeFunc(node, builder, TensorValue, Constants, outputType, noneValue);
  // }
}

std::vector<Value> ONNXConverter::loadGraphInput(const onnx::GraphProto& graph) {
  std::vector<Value> inputs;
  // for (int i = 0; i < graph.input_size(); i++) {
  //   const onnx::ValueInfoProto& val = graph.input(i);
  //   const std::string name = val.name();
  //   const onnx::TypeProto& type = val.type();

  //   auto inType = getTensorType(type);

  //   // add module arguments
  //   Block *entryBlock = &mainFunc.getBody().back();
  //   auto argVal = entryBlock->addArgument(inType, getLoc(builder.getContext(), name));
  //   inputs.push_back(argVal);

  //   // add tensor to graph
  //   auto inOp = builder.create<astl::InputOp>(
  //     getLoc(builder.getContext(), name), inType, argVal,
  //     LayoutAttr::get(builder.getContext(), Layout::NCHW));
    
  //   TensorValue[name] = inOp.getResult();
  // }
  return inputs;
}

std::vector<Value> ONNXConverter::loadGraphOutput(const onnx::GraphProto& graph) {
  std::vector<Value> outputs;
  // for (int i = 0; i < graph.output_size(); i++) {
  //   const onnx::ValueInfoProto& val = graph.output(i);
  //   const onnx::TypeProto& type = val.type();
  //   const onnx::TypeProto::Tensor& tensorType = type.tensor_type();
  //   const onnx::TensorShapeProto& shape = tensorType.shape();
  //   Dims dims(shape.dim_size());
  //   for (int j = 0; j < shape.dim_size(); j++) {
  //     const onnx::TensorShapeProto::Dimension& dim = shape.dim(j);
  //     if (dim.has_dim_param()) break;
  //     dims[j] = dim.dim_value();
  //   }

  //   // add tensor to graph
  // }
  return outputs;
}

RankedTensorType ONNXConverter::getTensorType(const onnx::TypeProto& type) {
  const onnx::TypeProto::Tensor& tensor_type = type.tensor_type();
  const onnx::TensorShapeProto& shape = tensor_type.shape();
  std::vector<int64_t> dims(shape.dim_size());
  for (int j = 0; j < shape.dim_size(); j++) {
    dims[j] = shape.dim(j).dim_value();
  }
  switch (tensor_type.elem_type()) {
    case 1:
      return RankedTensorType::get(dims, builder.getF32Type()); // float
    case 2:
      return RankedTensorType::get(dims, builder.getIntegerType(8)); // uint8
    case 3:
      return RankedTensorType::get(dims, builder.getI8Type()); // int8
    case 5:
      return RankedTensorType::get(dims, builder.getI16Type()); // int16
    case 6:
      return RankedTensorType::get(dims, builder.getI32Type()); // int32
    case 7:
      return RankedTensorType::get(dims, builder.getI64Type()); // int64
    case 10:
      return RankedTensorType::get(dims, builder.getF16Type()); // fp16
    default:
      llvm::report_fatal_error("Unsupported data type");
  }
}
RankedTensorType ONNXConverter::getTensorType(const onnx::TensorProto& tensor) {
  std::vector<int64_t> dims(tensor.dims_size());
  for (int j = 0; j < tensor.dims_size(); j++) {
    dims[j] = tensor.dims(j);
  }
  switch (tensor.data_type()) {
    case 1:
      return RankedTensorType::get(dims, builder.getF32Type()); // float
    case 2:
      return RankedTensorType::get(dims, builder.getIntegerType(8)); // uint8
    case 3:
      return RankedTensorType::get(dims, builder.getI8Type()); // int8
    case 5:
      return RankedTensorType::get(dims, builder.getI16Type()); // int16
    case 6:
      return RankedTensorType::get(dims, builder.getI32Type()); // int32
    case 7:
      return RankedTensorType::get(dims, builder.getI64Type()); // int64
    case 10:
      return RankedTensorType::get(dims, builder.getF16Type()); // fp16
    default:
      llvm::report_fatal_error("Unsupported data type");
  }
}

}  // namespace hld
} // namespace astl
