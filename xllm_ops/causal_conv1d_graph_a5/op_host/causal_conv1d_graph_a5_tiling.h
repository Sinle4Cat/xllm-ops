/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
// Keep this field order identical to CausalConv1dTilingData. The device entry
// reuses the existing, validated update implementation by layout.
BEGIN_TILING_DATA_DEF(CausalConv1dGraphA5TilingData)
  TILING_DATA_FIELD_DEF(int64_t, dim);
  TILING_DATA_FIELD_DEF(int64_t, cuSeqlen);
  TILING_DATA_FIELD_DEF(int64_t, seqLen);
  TILING_DATA_FIELD_DEF(int64_t, inputMode);
  TILING_DATA_FIELD_DEF(int64_t, width);
  TILING_DATA_FIELD_DEF(int64_t, stateLen);
  TILING_DATA_FIELD_DEF(int64_t, numCacheLines);
  TILING_DATA_FIELD_DEF(int64_t, batch);
  TILING_DATA_FIELD_DEF(int64_t, activationMode);
  TILING_DATA_FIELD_DEF(int64_t, padSlotId);
  TILING_DATA_FIELD_DEF(int64_t, hasBias);
  TILING_DATA_FIELD_DEF(int64_t, packedQDim);
  TILING_DATA_FIELD_DEF(int64_t, packedKDim);
  TILING_DATA_FIELD_DEF(int64_t, packedVDim);
  TILING_DATA_FIELD_DEF(int64_t, packedHeadDim);
  TILING_DATA_FIELD_DEF(int64_t, baseDim);
  TILING_DATA_FIELD_DEF(int64_t, baseDimCnt);
  TILING_DATA_FIELD_DEF(int64_t, hasNumAcceptedTokens);
  TILING_DATA_FIELD_DEF(int64_t, hasCacheIndices);
  TILING_DATA_FIELD_DEF(int64_t, hasInitialStateMode);
  TILING_DATA_FIELD_DEF(int64_t, hasInitStateWorkspace);
  TILING_DATA_FIELD_DEF(int64_t, tokenBlockSize);
  TILING_DATA_FIELD_DEF(int64_t, tokenBlockCnt);
  TILING_DATA_FIELD_DEF(int64_t, hasExplicitTokenSeqRanges);
  TILING_DATA_FIELD_DEF(int64_t, explicitTokenSeqRangeCount);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 128, tokenTileStartSeq);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 128, tokenTileEndSeq);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CausalConv1dGraphA5,
                           CausalConv1dGraphA5TilingData)
}  // namespace optiling
