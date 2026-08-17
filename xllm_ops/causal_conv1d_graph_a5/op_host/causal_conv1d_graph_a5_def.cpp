/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_def_registry.h"

namespace ops {
class CausalConv1dGraphA5 : public OpDef {
 public:
  explicit CausalConv1dGraphA5(const char* name) : OpDef(name) {
    const std::initializer_list<ge::DataType> data_types = {
        ge::DT_FLOAT16, ge::DT_BF16};
    const std::initializer_list<ge::Format> data_formats = {
        ge::FORMAT_ND, ge::FORMAT_ND};
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType(data_types)
        .Format(data_formats)
        .AutoContiguous();
    this->Input("weight")
        .ParamType(REQUIRED)
        .DataType(data_types)
        .Format(data_formats)
        .AutoContiguous();
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType(data_types)
        .Format(data_formats)
        .AutoContiguous();
    this->Input("convStates")
        .ParamType(REQUIRED)
        .DataType(data_types)
        .Format(data_formats)
        .AutoContiguous();
    // Deliberately not ValueDepend: graph replay updates the contents of this
    // persistent device tensor without rebuilding host tiling.
    this->Input("cacheIndices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .Format(data_formats)
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType(data_types)
        .Format(data_formats)
        .AutoContiguous();
    this->Attr("activationMode").AttrType(OPTIONAL).Int(1);
    this->Attr("padSlotId").AttrType(OPTIONAL).Int(-1);

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend950", config);
  }
};

OP_ADD(CausalConv1dGraphA5);
}  // namespace ops
