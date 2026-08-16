/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_def_registry.h"

namespace {
constexpr size_t kZIndex = 3;
constexpr size_t kOutputIndex = 0;
}  // namespace

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *z_shape = context->GetInputShape(kZIndex);
    if (z_shape == nullptr || z_shape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    *context->GetOutputShape(kOutputIndex) = *z_shape;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MegaGdnPrefillOp).InferShape(InferShape);
}  // namespace ge
