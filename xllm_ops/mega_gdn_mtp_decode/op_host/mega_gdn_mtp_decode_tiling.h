/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MegaGdnMtpDecodeTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, speculative_tokens);
TILING_DATA_FIELD_DEF(int64_t, sequence_length);
TILING_DATA_FIELD_DEF(int64_t, conv_state_length);
TILING_DATA_FIELD_DEF(int64_t, num_k_heads);
TILING_DATA_FIELD_DEF(int64_t, num_v_heads);
TILING_DATA_FIELD_DEF(int64_t, conv_tile_count);
TILING_DATA_FIELD_DEF(int64_t, checkpoint_stride);
TILING_DATA_FIELD_DEF(int64_t, use_dynamic_k);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MegaGdnMtpDecode, MegaGdnMtpDecodeTilingData)

}  // namespace optiling
