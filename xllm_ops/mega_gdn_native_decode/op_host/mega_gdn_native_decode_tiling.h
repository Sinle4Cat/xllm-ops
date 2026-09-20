/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MegaGdnNativeDecodeTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, sequence_length);
TILING_DATA_FIELD_DEF(int64_t, num_k_heads);
TILING_DATA_FIELD_DEF(int64_t, num_v_heads);
TILING_DATA_FIELD_DEF(int64_t, conv_stride);
TILING_DATA_FIELD_DEF(int64_t, ssm_stride);
TILING_DATA_FIELD_DEF(int64_t, total_tokens);
TILING_DATA_FIELD_DEF(int64_t, state_slots);
TILING_DATA_FIELD_DEF(int64_t, conv_rows);
TILING_DATA_FIELD_DEF(int64_t, qkv_stride);
TILING_DATA_FIELD_DEF(int64_t, z_stride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MegaGdnNativeDecode, MegaGdnNativeDecodeTilingData)
}
