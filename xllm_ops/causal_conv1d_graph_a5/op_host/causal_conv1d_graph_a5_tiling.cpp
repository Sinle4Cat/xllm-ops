/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "causal_conv1d_graph_a5_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int32_t kX = 0;
constexpr int32_t kWeight = 1;
constexpr int32_t kBias = 2;
constexpr int32_t kConvStates = 3;
constexpr int32_t kCacheIndices = 4;
constexpr int32_t kActivationMode = 0;
constexpr int32_t kPadSlotId = 1;
constexpr int64_t kDimAlignment = 16;

struct DimTileChoice {
  int64_t base_dim = 0;
  int64_t base_dim_count = 0;
  int64_t grid_size = 0;
};

DimTileChoice ChooseBaseDim(int64_t batch, int64_t dim, uint32_t core_num) {
  constexpr int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};
  DimTileChoice best_over;
  int64_t best_gap = std::numeric_limits<int64_t>::max();
  DimTileChoice best_under;
  for (int64_t base_dim : candidates) {
    if (dim % base_dim != 0) {
      continue;
    }
    const int64_t count = dim / base_dim;
    const int64_t grid = batch * count;
    if (grid >= static_cast<int64_t>(core_num)) {
      const int64_t gap = grid - static_cast<int64_t>(core_num);
      if (gap < best_gap) {
        best_over = {base_dim, count, grid};
        best_gap = gap;
      }
    } else if (grid > best_under.grid_size ||
               (grid == best_under.grid_size &&
                base_dim < best_under.base_dim)) {
      best_under = {base_dim, count, grid};
    }
  }
  return best_over.base_dim != 0 ? best_over : best_under;
}

bool SameDtype(gert::TilingContext* context, int32_t input,
               ge::DataType dtype) {
  const auto* desc = context->GetInputDesc(input);
  return desc != nullptr && desc->GetDataType() == dtype;
}

ge::graphStatus TilingFunc(gert::TilingContext* context) {
  if (context == nullptr || context->GetInputShape(kX) == nullptr ||
      context->GetInputShape(kWeight) == nullptr ||
      context->GetInputShape(kConvStates) == nullptr ||
      context->GetInputShape(kCacheIndices) == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape x = context->GetInputShape(kX)->GetStorageShape();
  const gert::Shape weight =
      context->GetInputShape(kWeight)->GetStorageShape();
  const gert::Shape state =
      context->GetInputShape(kConvStates)->GetStorageShape();
  const gert::Shape indices =
      context->GetInputShape(kCacheIndices)->GetStorageShape();
  if (x.GetDimNum() != 2 || weight.GetDimNum() != 2 ||
      state.GetDimNum() != 3 || indices.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }

  const int64_t batch = x.GetDim(0);
  const int64_t dim = x.GetDim(1);
  const int64_t width = weight.GetDim(0);
  const int64_t state_len = state.GetDim(1);
  const int64_t cache_lines = state.GetDim(0);
  if (batch <= 0 || dim <= 0 || dim % kDimAlignment != 0 ||
      width < 2 || width > 4 || weight.GetDim(1) != dim ||
      cache_lines <= 0 || state_len < width - 1 ||
      state.GetDim(2) != dim || indices.GetDim(0) != batch) {
    return ge::GRAPH_FAILED;
  }

  const auto* x_desc = context->GetInputDesc(kX);
  const auto* indices_desc = context->GetInputDesc(kCacheIndices);
  if (x_desc == nullptr || indices_desc == nullptr ||
      (x_desc->GetDataType() != ge::DT_FLOAT16 &&
       x_desc->GetDataType() != ge::DT_BF16) ||
      !SameDtype(context, kWeight, x_desc->GetDataType()) ||
      !SameDtype(context, kConvStates, x_desc->GetDataType()) ||
      indices_desc->GetDataType() != ge::DT_INT64) {
    return ge::GRAPH_FAILED;
  }

  bool has_bias = false;
  if (const auto* bias_shape = context->GetOptionalInputShape(kBias)) {
    const gert::Shape bias = bias_shape->GetStorageShape();
    has_bias = bias.GetDimNum() == 1 && bias.GetDim(0) > 0;
    if (has_bias &&
        (bias.GetDim(0) != dim ||
         !SameDtype(context, kBias, x_desc->GetDataType()))) {
      return ge::GRAPH_FAILED;
    }
  }

  const auto* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const int64_t* activation = attrs->GetAttrPointer<int64_t>(kActivationMode);
  const int64_t* pad_slot = attrs->GetAttrPointer<int64_t>(kPadSlotId);
  if (activation == nullptr || pad_slot == nullptr ||
      (*activation != 0 && *activation != 1)) {
    return ge::GRAPH_FAILED;
  }

  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  const uint32_t core_num = platform.GetCoreNumAiv();
  if (core_num == 0) {
    return ge::GRAPH_FAILED;
  }
  const DimTileChoice choice = ChooseBaseDim(batch, dim, core_num);
  if (choice.base_dim <= 0 || choice.base_dim_count <= 0 ||
      choice.grid_size <= 0) {
    return ge::GRAPH_FAILED;
  }

  CausalConv1dGraphA5TilingData tiling;
  tiling.set_dim(dim);
  tiling.set_cuSeqlen(batch);
  tiling.set_seqLen(1);
  tiling.set_inputMode(2);
  tiling.set_width(width);
  tiling.set_stateLen(state_len);
  tiling.set_numCacheLines(cache_lines);
  tiling.set_batch(batch);
  tiling.set_activationMode(*activation);
  tiling.set_padSlotId(*pad_slot);
  tiling.set_hasBias(has_bias ? 1 : 0);
  tiling.set_packedQDim(0);
  tiling.set_packedKDim(0);
  tiling.set_packedVDim(0);
  tiling.set_packedHeadDim(0);
  tiling.set_baseDim(choice.base_dim);
  tiling.set_baseDimCnt(choice.base_dim_count);
  tiling.set_hasNumAcceptedTokens(0);
  tiling.set_hasCacheIndices(1);
  tiling.set_hasInitialStateMode(0);
  tiling.set_hasInitStateWorkspace(0);
  tiling.set_tokenBlockSize(0);
  tiling.set_tokenBlockCnt(0);
  tiling.set_hasExplicitTokenSeqRanges(0);
  tiling.set_explicitTokenSeqRangeCount(0);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  // This A5-only graph kernel is also the acquire fence for persistent Conv
  // and SSM state published by MegaGdnPrefillOp.  The following recurrent
  // graph may run on any AIV, so launching only the causal-conv work grid
  // leaves idle AIVs with stale private cache lines.  Launch every AIV: the
  // existing grid-stride loop keeps the causal-conv work unchanged, while
  // otherwise-idle blocks execute the entry DCCI and then return.
  context->SetBlockDim(core_num);
  context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(CausalConv1dGraphA5).Tiling(TilingFunc);
}  // namespace optiling
