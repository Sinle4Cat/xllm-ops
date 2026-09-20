/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

static ge::graphStatus InferShapeMegaGdnNativeDecode(
    gert::InferShapeContext* context) {
  const gert::Shape* qkv_shape = context->GetInputShape(0);
  const gert::Shape* z_shape = context->GetInputShape(1);
  const gert::Shape* conv_state_shape = context->GetInputShape(5);
  const gert::Shape* ssm_state_shape = context->GetInputShape(8);
  OP_CHECK_NULL_WITH_CONTEXT(context, qkv_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, z_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, conv_state_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, ssm_state_shape);

  const auto* gates = context->GetInputShape(2);
  const auto* weights = context->GetInputShape(4);
  OP_CHECK_NULL_WITH_CONTEXT(context, gates);
  OP_CHECK_NULL_WITH_CONTEXT(context, weights);
  auto* conv = context->GetOutputShape(0);
  conv->SetDimNum(2);
  conv->SetDim(0, gates->GetDim(0)); conv->SetDim(1, weights->GetDim(1));
  *context->GetOutputShape(1) = *conv_state_shape;
  *context->GetOutputShape(2) = *ssm_state_shape;
  auto* output = context->GetOutputShape(3);
  output->SetDimNum(3);
  output->SetDim(0, gates->GetDim(0)); output->SetDim(1, gates->GetDim(1)); output->SetDim(2, 128);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MegaGdnNativeDecode)
    .InferShape(InferShapeMegaGdnNativeDecode);

}  // namespace ops
