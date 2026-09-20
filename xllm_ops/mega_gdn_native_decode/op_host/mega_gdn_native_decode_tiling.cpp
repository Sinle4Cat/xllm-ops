/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <algorithm>
#include <array>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "mega_gdn_native_decode_tiling.h"

namespace optiling {
namespace {
bool ShapeIs(const gert::Shape& shape, std::initializer_list<int64_t> dims) {
  if (shape.GetDimNum() != dims.size()) return false;
  size_t i = 0;
  for (auto dim : dims) if (shape.GetDim(i++) != dim) return false;
  return true;
}
}
static ge::graphStatus NativeTiling(gert::TilingContext* context) {
  for (size_t i = 0; i < 13; ++i) {
    if (!context->GetInputShape(i) || !context->GetInputDesc(i)) return ge::GRAPH_FAILED;
  }
  auto shape = [context](size_t i) -> const gert::Shape& {
    return context->GetInputShape(i)->GetStorageShape();
  };
  const auto& qkv = shape(0);
  const auto& z = shape(1);
  const auto& ids = shape(9);
  if (qkv.GetDimNum() != 1 || z.GetDimNum() != 1 || ids.GetDimNum() != 2 ||
      shape(2).GetDimNum() != 2 || shape(4).GetDimNum() != 2)
    return ge::GRAPH_FAILED;
  const int64_t tokens = shape(2).GetDim(0), channels = shape(4).GetDim(1);
  const int64_t hv = shape(2).GetDim(1), batch = ids.GetDim(0), width = ids.GetDim(1);
  const int64_t qk = channels - hv * 128;
  const int64_t hk = qk / 256;
  if (tokens < 1 || tokens > std::numeric_limits<int32_t>::max() / std::max(channels, int64_t(1)) ||
      batch < 1 || batch > 32 || width < 1 || width > 17 ||
      hv < 1 || hv > 64 || qk <= 0 || qk % 256 || hk < 1 || hk > 16 ||
      (hk & (hk - 1)) || hv % hk || hv / hk > 4 ||
      !ShapeIs(shape(2), {tokens, hv}) ||
      !ShapeIs(shape(3), {tokens, hv}) || !ShapeIs(shape(4), {4, channels}) ||
      !ShapeIs(shape(6), {hv}) || !ShapeIs(shape(7), {hv}) ||
      !ShapeIs(shape(10), {batch + 1}) || !ShapeIs(shape(11), {batch}) ||
      !ShapeIs(shape(12), {128})) return ge::GRAPH_FAILED;
  const auto* attrs = context->GetAttrs();
  if (!attrs) return ge::GRAPH_FAILED;
  std::array<int64_t, 6> values;
  for (size_t i = 0; i < values.size(); ++i) {
    const auto* value = attrs->GetAttrPointer<int64_t>(i);
    if (!value || *value < 1 || *value > std::numeric_limits<int32_t>::max())
      return ge::GRAPH_FAILED;
    values[i] = *value;
  }
  const auto [conv_stride, ssm_stride, rows, slots, qkv_stride, z_stride] = values;
  if (qkv_stride < channels || z_stride < hv * 128 ||
      qkv.GetDim(0) != (tokens - 1) * qkv_stride + channels ||
      z.GetDim(0) != (tokens - 1) * z_stride + hv * 128 || rows < width + 2 || rows > 19 || slots < 2 ||
      conv_stride < rows * channels || ssm_stride < hv * 128 * 128 ||
      shape(5).GetDimNum() != 1 || shape(8).GetDimNum() != 1 ||
      shape(5).GetDim(0) != (slots - 1) * conv_stride + rows * channels ||
      shape(8).GetDim(0) != (slots - 1) * ssm_stride + hv * 128 * 128)
    return ge::GRAPH_FAILED;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  if (ub < 175360) return ge::GRAPH_FAILED;
  const uint32_t tasks = std::max(channels / 128, std::min(batch, tokens) * hv);
  const uint32_t used = std::min(tasks, platform.GetCoreNumAiv());
  const uint32_t blocks = platform.CalcTschBlockDim(
      used, platform.GetCoreNumAic(), platform.GetCoreNumAiv());
  if (!blocks) return ge::GRAPH_FAILED;
  MegaGdnNativeDecodeTilingData t;
  t.set_batch_size(batch); t.set_sequence_length(width);
  t.set_num_k_heads(hk); t.set_num_v_heads(hv);
  t.set_conv_stride(conv_stride); t.set_ssm_stride(ssm_stride);
  t.set_qkv_stride(qkv_stride); t.set_z_stride(z_stride);
  t.set_total_tokens(tokens); t.set_state_slots(slots); t.set_conv_rows(rows);
  t.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(t.GetDataSize());
  context->SetTilingKey(1); context->SetBlockDim(blocks);
  if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
  context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MegaGdnNativeDecode).Tiling(NativeTiling);
}
