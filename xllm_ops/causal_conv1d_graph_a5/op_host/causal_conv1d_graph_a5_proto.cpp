/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_impl_registry.h"

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return GRAPH_SUCCESS;
}

IMPL_OP(CausalConv1dGraphA5)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
