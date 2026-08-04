/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

// Reuse the verified PTO tile aliases and vector primitives from the
// single-token kernel. The MTP schedule and state contract remain independent.
#include "mega_gdn_decode_pto_kernel.h"

namespace qwen35_mtp_decode_pto {

using namespace pto;
using qwen35_decode_pto::TileUbDataDN;
using qwen35_decode_pto::TileUbDataND;

constexpr int32_t kHeadDim = 128;
constexpr int32_t kSsmHeadElements = kHeadDim * kHeadDim;
constexpr int32_t kMaxBatchSize = 32;
constexpr int32_t kMaxNumKHeads = 16;
constexpr int32_t kMaxNumVHeads = 64;
constexpr int32_t kMaxConvDim =
    (2 * kMaxNumKHeads + kMaxNumVHeads) * kHeadDim;
constexpr int32_t kQkGroupCacheSequenceLength = 9;

constexpr int32_t kUbConvWeightHalf0 = 0;
constexpr int32_t kUbConvWeightHalf1 = 256;
constexpr int32_t kUbConvWeightHalf2 = 512;
constexpr int32_t kUbConvWeightHalf3 = 768;
constexpr int32_t kUbConvWeight0 = 1024;
constexpr int32_t kUbConvWeight1 = 1536;
constexpr int32_t kUbConvWeight2 = 2048;
constexpr int32_t kUbConvWeight3 = 2560;
constexpr int32_t kUbConvHistoryHalf0 = 3072;
constexpr int32_t kUbConvHistoryHalf1 = 3328;
constexpr int32_t kUbConvHistoryHalf2 = 3584;
constexpr int32_t kUbConvInputHalf = 3840;
constexpr int32_t kUbConvHistory0 = 4096;
constexpr int32_t kUbConvHistory1 = 4608;
constexpr int32_t kUbConvHistory2 = 5120;
constexpr int32_t kUbConvInput = 5632;
constexpr int32_t kUbConvAcc = 6144;
constexpr int32_t kUbConvTmp = 6656;
constexpr int32_t kUbConvOutput = 7168;
constexpr int32_t kUbConvOutputHalf = 7680;
constexpr int32_t kUbConvInputHalfPong = 7936;
constexpr int32_t kUbConvOutputHalfPong = 8192;
constexpr int32_t kUbQHalf = 8704;
constexpr int32_t kUbKHalf = 8960;
constexpr int32_t kUbAHalf = 9216;
constexpr int32_t kUbBHalf = 9248;
constexpr int32_t kUbQ = 9280;
constexpr int32_t kUbK = 9792;
constexpr int32_t kUbScalar = 10304;
constexpr int32_t kUbScalar2 = 10336;
constexpr int32_t kUbNormSquare = 10368;
constexpr int32_t kUbNormValue = 10880;
constexpr int32_t kUbReduceTmp = 10912;
constexpr int32_t kUbScalarTmp = 19104;
constexpr int32_t kUbExpA = 19136;
constexpr int32_t kUbScalarWork = 19168;
constexpr int32_t kUbVHalf = 19200;
constexpr int32_t kUbState = 19456;
constexpr int32_t kUbV = 84992;
constexpr int32_t kUbStateCompute = 85504;
constexpr int32_t kUbPrediction = 151040;
constexpr int32_t kUbDelta = 151552;
constexpr int32_t kUbOutputHalf = 152064;
constexpr int32_t kUbNormHalf = 152320;
constexpr int32_t kUbZHalf = 152576;
constexpr int32_t kUbNormWeightHalf = 152832;
constexpr int32_t kUbNorm = 153088;
constexpr int32_t kUbZ = 153600;
constexpr int32_t kUbNormWeight = 154112;
constexpr int32_t kUbSquare = 154624;
constexpr int32_t kUbRms = 155136;
constexpr int32_t kUbGate = 155168;
constexpr int32_t kUbFinalHalf = 155680;
constexpr int32_t kUbVectorScratch = 155936;
constexpr int32_t kUbColumnSumTmp = 156672;
constexpr int32_t kUbALogStatic = 173824;
constexpr int32_t kUbDtBiasStatic = 174080;
constexpr int32_t kUbQkCacheTail = 174112;
constexpr int32_t kQkCacheBatchBytes =
    kQkGroupCacheSequenceLength * kHeadDim *
    static_cast<int32_t>(sizeof(float));
constexpr int32_t kUbQkCacheK =
    kUbQkCacheTail + kQkCacheBatchBytes;
constexpr int32_t kUbQkCacheEnd =
    kUbQkCacheK + kQkCacheBatchBytes;
constexpr int32_t kDeferredNormRows = kQkGroupCacheSequenceLength;
constexpr int32_t kUbDeferredReadoutHalf = kUbQkCacheEnd;
constexpr int32_t kUbDeferredZHalf =
    kUbDeferredReadoutHalf +
    kQkGroupCacheSequenceLength * kHeadDim * sizeof(bfloat16_t);
constexpr int32_t kUbDeferredRowsEnd =
    kUbDeferredZHalf +
    kQkGroupCacheSequenceLength * kHeadDim * sizeof(bfloat16_t);

static_assert(kUbQHalf % 32 == 0);
static_assert(
    kUbKHalf == kUbQHalf + kHeadDim * sizeof(bfloat16_t));
static_assert(
    kUbAHalf >= kUbQHalf + 2 * kHeadDim * sizeof(bfloat16_t));
static_assert(kUbQkCacheTail ==
              kUbDtBiasStatic + 8 * sizeof(float));
static_assert(kUbQkCacheEnd == 183328);
static_assert(kUbDeferredRowsEnd == 187936);
static_assert(kUbDeferredRowsEnd <= 192 * 1024);

template <typename T, int32_t Rows, int32_t Cols>
AICORE PTO_INLINE void SiluNoCopy(
    TileUbDataND<T, Rows, Cols>& dst,
    TileUbDataND<T, Rows, Cols>& src,
    TileUbDataND<T, Rows, Cols>& tmp) {
  TMULS(tmp, src, -1);
  qwen35_decode_pto::VectorBarrier();
  TEXP(tmp, tmp);
  qwen35_decode_pto::VectorBarrier();
  TADDS(tmp, tmp, 1);
  qwen35_decode_pto::VectorBarrier();
  TRECIP(dst, tmp);
  qwen35_decode_pto::VectorBarrier();
  TMUL(dst, src, dst);
}

AICORE PTO_INLINE void LoadBf16Row(__gm__ bfloat16_t* handle,
                                   int32_t ub_address) {
  qwen35_decode_pto::CopyGmToUb<
      bfloat16_t,
      bfloat16_t,
      1,
      1,
      1,
      1,
      128,
      1,
      1,
      1,
      128,
      1,
      1,
      128,
      pto::PadValue::Zero>(handle, ub_address, 0, 1, 128);
}

AICORE PTO_INLINE void LoadQkBf16Rows(__gm__ bfloat16_t* handle,
                                      int32_t ub_address,
                                      int32_t row_stride) {
  using QkShape = pto::Shape<1, 1, 1, 2, kHeadDim>;
  using QkStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  QkShape shape;
  QkStride stride(row_stride);
  pto::GlobalTensor<bfloat16_t, QkShape, QkStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, 2, kHeadDim> tile;
  TASSIGN(tile, ub_address);
  TLOAD(tile, tensor);
}

template <int32_t Rows>
AICORE PTO_INLINE void LoadStridedBf16Rows(
    __gm__ bfloat16_t* handle,
    int32_t ub_address,
    int32_t row_stride) {
  using RowShape = pto::Shape<1, 1, 1, Rows, kHeadDim>;
  using RowStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  RowShape shape;
  RowStride stride(row_stride);
  pto::GlobalTensor<bfloat16_t, RowShape, RowStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> tile;
  TASSIGN(tile, ub_address);
  TLOAD(tile, tensor);
}

AICORE PTO_INLINE void StoreBf16Row(__gm__ bfloat16_t* handle,
                                    int32_t ub_address) {
  qwen35_decode_pto::CopyUbToGm<
      bfloat16_t,
      bfloat16_t,
      1,
      1,
      1,
      1,
      128,
      1,
      1,
      1,
      128,
      1,
      1,
      128>(handle, ub_address, 0, 1, 128);
}

template <int32_t Rows>
AICORE PTO_INLINE void StoreStridedBf16Rows(
    __gm__ bfloat16_t* handle,
    int32_t ub_address,
    int32_t row_stride) {
  using RowShape = pto::Shape<1, 1, 1, Rows, kHeadDim>;
  using RowStride = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;
  RowShape shape;
  RowStride stride(row_stride);
  pto::GlobalTensor<bfloat16_t, RowShape, RowStride> tensor(
      handle, shape, stride);
  TileUbDataND<bfloat16_t, Rows, kHeadDim> tile;
  TASSIGN(tile, ub_address);
  TSTORE(tensor, tile);
}

AICORE PTO_INLINE void LoadBf16Scalar(__gm__ bfloat16_t* handle,
                                      int32_t ub_address) {
  qwen35_decode_pto::CopyGmToUb<
      bfloat16_t,
      bfloat16_t,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      16,
      pto::PadValue::Null>(handle, ub_address, 0, 1, 1);
}

AICORE PTO_INLINE void LoadFloatScalar(__gm__ float* handle,
                                       int32_t ub_address) {
  qwen35_decode_pto::CopyGmToUb<float,
                                float,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                1,
                                8,
                                pto::PadValue::Null>(
      handle, ub_address, 0, 1, 1);
}

AICORE PTO_INLINE void LoadState(__gm__ float* handle, int32_t ub_address) {
  qwen35_decode_pto::CopyGmToUb<float,
                                float,
                                1,
                                1,
                                1,
                                128,
                                128,
                                1,
                                1,
                                1,
                                128,
                                1,
                                128,
                                128,
                                pto::PadValue::Zero>(
      handle, ub_address, 0, 128, 128);
}

AICORE PTO_INLINE void StoreState(__gm__ float* handle, int32_t ub_address) {
  qwen35_decode_pto::CopyUbToGm<float,
                                float,
                                1,
                                1,
                                1,
                                128,
                                128,
                                1,
                                1,
                                1,
                                128,
                                1,
                                128,
                                128>(
      handle, ub_address, 0, 128, 128);
}

template <bool ApplyQScale, int32_t Rows>
AICORE PTO_INLINE void NormalizeQkRows(
    int32_t rows_address,
    int32_t square_address,
    int32_t reduce_tmp_address,
    int32_t norm_address,
    int32_t norm_sqrt_address) {
  constexpr int32_t kNormCapacity = 16;
  TileUbDataND<float, Rows, kHeadDim> rows;
  TASSIGN(rows, rows_address);
  TileUbDataND<float, Rows, kHeadDim> square;
  TASSIGN(square, square_address);
  TileUbDataND<float, Rows, 64> reduce_tmp;
  TASSIGN(reduce_tmp, reduce_tmp_address);
  TileUbDataDN<float, kNormCapacity, 1, Rows, 1> norm_dn;
  TASSIGN(norm_dn, norm_address);
  TileUbDataND<float, 1, kNormCapacity, 1, Rows> norm;
  TASSIGN(norm, norm_address);
  TileUbDataND<float, 1, kNormCapacity, 1, Rows> norm_sqrt;
  TASSIGN(norm_sqrt, norm_sqrt_address);

  TMUL(square, rows, rows);
  qwen35_decode_pto::VectorBarrier();
  TROWSUM(norm_dn, square, reduce_tmp);
  qwen35_decode_pto::VectorBarrier();
  TADDS(norm, norm, 1.0e-6f);
  qwen35_decode_pto::VectorBarrier();
  TSQRT(norm_sqrt, norm);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int32_t row = 0; row < Rows; ++row) {
    TileUbDataND<float, 1, kHeadDim> row_tile;
    TASSIGN(
        row_tile,
        rows_address + row * kHeadDim * sizeof(float));
    const float row_norm = norm_sqrt.GetValue(row);
    TMULS(row_tile, row_tile, 1.0f / row_norm);
  }
  qwen35_decode_pto::VectorBarrier();
  if constexpr (ApplyQScale) {
    TMULS(rows, rows, 8.838835e-02f);
    qwen35_decode_pto::VectorBarrier();
  }
}

template <int32_t SpeculativeTokens,
          bool UseQkGroupCache,
          bool UseDeferredNorm = UseQkGroupCache>
AICORE PTO_INLINE void Run(
    __gm__ bfloat16_t* qkv_handle,
    __gm__ bfloat16_t* z_handle,
    __gm__ bfloat16_t* b_handle,
    __gm__ bfloat16_t* a_handle,
    __gm__ bfloat16_t* conv_weight_handle,
    __gm__ bfloat16_t* conv_state_handle,
    __gm__ float* a_log_handle,
    __gm__ float* dt_bias_handle,
    __gm__ float* ssm_state_handle,
    __gm__ int* read_state_indices_handle,
    __gm__ int* write_state_indices_handle,
    __gm__ int* num_accepted_tokens_handle,
    __gm__ bfloat16_t* norm_weight_handle,
    __gm__ bfloat16_t* conv_out_handle,
    __gm__ bfloat16_t* conv_state_out_handle,
    __gm__ float* ssm_state_out_handle,
    __gm__ bfloat16_t* out_handle,
    int32_t num_k_heads,
    int32_t num_v_heads,
    int32_t batch_size,
    int32_t runtime_sequence_length) {
  constexpr bool kIsDynamic = SpeculativeTokens == 0;
  static_assert(!UseDeferredNorm || !kIsDynamic);
  constexpr int32_t kRunDeferredNormRows =
      UseDeferredNorm ? SpeculativeTokens + 1 : 1;
  constexpr int32_t kRunDeferredReadoutHalf =
      UseQkGroupCache ? kUbDeferredReadoutHalf : kUbQkCacheTail;
  constexpr int32_t kRunDeferredZHalf =
      kRunDeferredReadoutHalf +
      kRunDeferredNormRows * kHeadDim * sizeof(bfloat16_t);
  constexpr int32_t kRunDeferredRowsEnd =
      kRunDeferredZHalf +
      kRunDeferredNormRows * kHeadDim * sizeof(bfloat16_t);
  static_assert(!UseDeferredNorm || kRunDeferredRowsEnd <= 192 * 1024);
  const int32_t sequence_length =
      kIsDynamic ? runtime_sequence_length : SpeculativeTokens + 1;
  const int32_t conv_state_length = sequence_length + 2;
  const int32_t conv_dim =
      (2 * num_k_heads + num_v_heads) * kHeadDim;
  const int32_t conv_tile_count = conv_dim / kHeadDim;
  const int32_t conv_state_stride = conv_state_length * conv_dim;
  const int32_t v_heads_per_k = num_v_heads / num_k_heads;
  const int32_t v_width = num_v_heads * kHeadDim;
  const int32_t ssm_checkpoint_stride =
      num_v_heads * kSsmHeadElements;

  TileUbDataND<bfloat16_t, 1, 128> w_half0;
  TASSIGN(w_half0, kUbConvWeightHalf0);
  TileUbDataND<bfloat16_t, 1, 128> w_half1;
  TASSIGN(w_half1, kUbConvWeightHalf1);
  TileUbDataND<bfloat16_t, 1, 128> w_half2;
  TASSIGN(w_half2, kUbConvWeightHalf2);
  TileUbDataND<bfloat16_t, 1, 128> w_half3;
  TASSIGN(w_half3, kUbConvWeightHalf3);
  TileUbDataND<float, 1, 128> w0;
  TASSIGN(w0, kUbConvWeight0);
  TileUbDataND<float, 1, 128> w1;
  TASSIGN(w1, kUbConvWeight1);
  TileUbDataND<float, 1, 128> w2;
  TASSIGN(w2, kUbConvWeight2);
  TileUbDataND<float, 1, 128> w3;
  TASSIGN(w3, kUbConvWeight3);
  TileUbDataND<bfloat16_t, 1, 128> hist_half0;
  TASSIGN(hist_half0, kUbConvHistoryHalf0);
  TileUbDataND<bfloat16_t, 1, 128> hist_half1;
  TASSIGN(hist_half1, kUbConvHistoryHalf1);
  TileUbDataND<bfloat16_t, 1, 128> hist_half2;
  TASSIGN(hist_half2, kUbConvHistoryHalf2);
  TileUbDataND<bfloat16_t, 1, 128> x_half;
  TASSIGN(x_half, kUbConvInputHalf);
  TileUbDataND<float, 1, 128> hist0;
  TASSIGN(hist0, kUbConvHistory0);
  TileUbDataND<float, 1, 128> hist1;
  TASSIGN(hist1, kUbConvHistory1);
  TileUbDataND<float, 1, 128> hist2;
  TASSIGN(hist2, kUbConvHistory2);
  TileUbDataND<float, 1, 128> x_fp32;
  TASSIGN(x_fp32, kUbConvInput);
  TileUbDataND<float, 1, 128> conv_acc;
  TASSIGN(conv_acc, kUbConvAcc);
  TileUbDataND<float, 1, 128> conv_tmp;
  TASSIGN(conv_tmp, kUbConvTmp);
  TileUbDataND<float, 1, 128> conv_y;
  TASSIGN(conv_y, kUbConvOutput);
  TileUbDataND<bfloat16_t, 1, 128> conv_y_half;
  TASSIGN(conv_y_half, kUbConvOutputHalf);

  TileUbDataND<bfloat16_t, 1, 128> q_half;
  TASSIGN(q_half, kUbQHalf);
  TileUbDataND<bfloat16_t, 1, 128> k_half;
  TASSIGN(k_half, kUbKHalf);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> a_half;
  TASSIGN(a_half, kUbAHalf);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> b_half;
  TASSIGN(b_half, kUbBHalf);
  TileUbDataND<float, 1, 128> q_fp32;
  TASSIGN(q_fp32, kUbQ);
  TileUbDataND<float, 1, 128> k_fp32;
  TASSIGN(k_fp32, kUbK);
  TileUbDataND<float, 1, 8, 1, 1> scalar;
  TASSIGN(scalar, kUbScalar);
  TileUbDataND<float, 1, 8, 1, 1> scalar2;
  TASSIGN(scalar2, kUbScalar2);
  TileUbDataND<float, 1, 128> norm_square;
  TASSIGN(norm_square, kUbNormSquare);
  TileUbDataND<float, 1, 8, 1, 1> norm_value;
  TASSIGN(norm_value, kUbNormValue);
  TileUbDataND<uint8_t, 128, 64> reduce_tmp;
  TASSIGN(reduce_tmp, kUbReduceTmp);
  TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, kUbScalarTmp);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> rounded_gate_half;
  TASSIGN(rounded_gate_half, kUbExpA);
  TileUbDataND<float, 1, 8, 1, 1> scalar_work;
  TASSIGN(scalar_work, kUbScalarWork);
  TileUbDataND<bfloat16_t, 1, 128> v_half;
  TASSIGN(v_half, kUbVHalf);
  TileUbDataND<float, 128, 128> state;
  TASSIGN(state, kUbState);
  TileUbDataND<float, 1, 128> v_fp32;
  TASSIGN(v_fp32, kUbV);
  TileUbDataND<float, 128, 128> compute;
  TASSIGN(compute, kUbStateCompute);
  TileUbDataND<float, 1, 128> prediction;
  TASSIGN(prediction, kUbPrediction);
  TileUbDataND<float, 1, 128> delta;
  TASSIGN(delta, kUbDelta);
  TileUbDataND<bfloat16_t, 1, 128> readout_half;
  TASSIGN(readout_half, kUbOutputHalf);
  TileUbDataND<bfloat16_t, 1, 128> norm_half;
  TASSIGN(norm_half, kUbNormHalf);
  TileUbDataND<bfloat16_t, 1, 128> z_half;
  TASSIGN(z_half, kUbZHalf);
  TileUbDataND<bfloat16_t, 1, 128> norm_weight_half;
  TASSIGN(norm_weight_half, kUbNormWeightHalf);
  TileUbDataND<float, 1, 128> norm_fp32;
  TASSIGN(norm_fp32, kUbNorm);
  TileUbDataND<float, 1, 128> z_fp32;
  TASSIGN(z_fp32, kUbZ);
  TileUbDataND<float, 1, 128> norm_weight_fp32;
  TASSIGN(norm_weight_fp32, kUbNormWeight);
  TileUbDataND<float, 1, 128> square_fp32;
  TASSIGN(square_fp32, kUbSquare);
  TileUbDataND<float, 1, 8, 1, 1> rms;
  TASSIGN(rms, kUbRms);
  TileUbDataND<float, 1, 128> gate_fp32;
  TASSIGN(gate_fp32, kUbGate);
  TileUbDataND<bfloat16_t, 1, 128> final_half;
  TASSIGN(final_half, kUbFinalHalf);
  TileUbDataND<float, 32, 128> colsum_tmp;
  TASSIGN(colsum_tmp, kUbColumnSumTmp);
  TileUbDataND<float, 1, 8, 1, 1> a_log_static;
  TASSIGN(a_log_static, kUbALogStatic);
  TileUbDataND<float, 1, 8, 1, 1> dt_bias_static;
  TASSIGN(dt_bias_static, kUbDtBiasStatic);
  // Full-span alias keeps the static UB footprint parser aware of the
  // dynamically addressed all-token Q/K subtiles.
  TileUbDataND<
      float,
      2 * kQkGroupCacheSequenceLength,
      128> qk_cache_tail;
  TASSIGN(qk_cache_tail, kUbQkCacheTail);
  TileUbDataND<
      bfloat16_t,
      kRunDeferredNormRows,
      kHeadDim>
      deferred_readout_half;
  TASSIGN(deferred_readout_half, kRunDeferredReadoutHalf);
  TileUbDataND<
      bfloat16_t,
      kRunDeferredNormRows,
      kHeadDim>
      deferred_z_half;
  TASSIGN(deferred_z_half, kRunDeferredZHalf);

#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
#if defined(PTO_NPU_ARCH_A2A3)
  const auto cid = get_block_idx();
  const auto vid = get_subblockid();
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int32_t vector_core_idx = cid * get_subblockdim() + vid;
  const int32_t vector_core_count =
      get_block_num() * get_subblockdim();
#else
  const int32_t vector_core_idx = get_block_idx();
  const int32_t vector_core_count = get_block_num();
#endif

  // Phase 1: select accepted Conv history, process all S tokens, and write
  // the private extended window.
  for (int32_t conv_tile = vector_core_idx; conv_tile < conv_tile_count;
       conv_tile += vector_core_count) {
    const int32_t channel_offset = conv_tile * kHeadDim;
    qwen35_decode_pto::CopyConvWeightsGmToUb(
        conv_weight_handle + channel_offset,
        kUbConvWeightHalf0,
        conv_dim);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(w0, w_half0, RoundMode::CAST_NONE);
    TCVT(w1, w_half1, RoundMode::CAST_NONE);
    TCVT(w2, w_half2, RoundMode::CAST_NONE);
    TCVT(w3, w_half3, RoundMode::CAST_NONE);
    qwen35_decode_pto::VectorBarrier();

    int8_t conv_pingpong_flag;
    if constexpr (SpeculativeTokens == 8) {
      conv_pingpong_flag = 0;
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    }

    for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int32_t read_slot = *(read_state_indices_handle + batch_idx);
      const int32_t write_slot = *(write_state_indices_handle + batch_idx);
      const int32_t accepted =
          *(num_accepted_tokens_handle + batch_idx);
      const int32_t read_history_offset =
          read_slot * conv_state_stride +
          (accepted - 1) * conv_dim + channel_offset;
      qwen35_decode_pto::CopyConvHistoryGmToUb(
          conv_state_handle + read_history_offset,
          kUbConvHistoryHalf0,
          conv_dim);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TCVT(hist0, hist_half0, RoundMode::CAST_NONE);
      TCVT(hist1, hist_half1, RoundMode::CAST_NONE);
      TCVT(hist2, hist_half2, RoundMode::CAST_NONE);
      qwen35_decode_pto::VectorBarrier();

      // Snapshot the accepted source rows before a same-slot write can
      // overwrite them. The explicit ready/free pair makes the single UB
      // buffer correct before introducing ping-pong overlap.
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      StoreBf16Row(
          conv_state_out_handle +
              write_slot * conv_state_stride + channel_offset,
          kUbConvHistoryHalf1);
      StoreBf16Row(
          conv_state_out_handle +
              write_slot * conv_state_stride + conv_dim + channel_offset,
          kUbConvHistoryHalf2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

      for (int32_t token_idx = 0; token_idx < sequence_length;
           ++token_idx) {
        const int32_t token_offset =
            (batch_idx * sequence_length + token_idx) * conv_dim +
            channel_offset;
        int32_t conv_input_half_address;
        int32_t conv_output_half_address;
        event_t conv_pingpong_event;
        if constexpr (SpeculativeTokens == 8) {
          conv_input_half_address =
              conv_pingpong_flag == 0 ? kUbConvInputHalf
                                      : kUbConvInputHalfPong;
          conv_output_half_address =
              conv_pingpong_flag == 0 ? kUbConvOutputHalf
                                      : kUbConvOutputHalfPong;
          conv_pingpong_event =
              static_cast<event_t>(conv_pingpong_flag + 2);
          wait_flag(PIPE_MTE3, PIPE_MTE2, conv_pingpong_event);
          LoadBf16Row(
              qkv_handle + token_offset, conv_input_half_address);
          set_flag(PIPE_MTE2, PIPE_V, conv_pingpong_event);
          wait_flag(PIPE_MTE2, PIPE_V, conv_pingpong_event);
          TileUbDataND<bfloat16_t, 1, 128> conv_input_half;
          TASSIGN(conv_input_half, conv_input_half_address);
          TCVT(x_fp32, conv_input_half, RoundMode::CAST_NONE);
        } else {
          LoadBf16Row(qkv_handle + token_offset, kUbConvInputHalf);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          TCVT(x_fp32, x_half, RoundMode::CAST_NONE);
        }
        qwen35_decode_pto::VectorBarrier();

        TMUL(conv_acc, w0, hist0);
        TMUL(conv_tmp, w1, hist1);
        qwen35_decode_pto::VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
        qwen35_decode_pto::VectorBarrier();
        TMUL(conv_tmp, w2, hist2);
        qwen35_decode_pto::VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
        qwen35_decode_pto::VectorBarrier();
        TileUbDataND<float, 1, 128> mul_add_tmp;
        TASSIGN(mul_add_tmp, kUbVectorScratch);
        qwen35_decode_pto::MulAddDst<float, 1, 128>(
            conv_acc, x_fp32, w3, mul_add_tmp);
        qwen35_decode_pto::VectorBarrier();
        TileUbDataND<float, 1, 128> silu_tmp;
        TASSIGN(silu_tmp, kUbVectorScratch);
        qwen35_decode_pto::Silu<float, 1, 128>(
            conv_y, conv_acc, silu_tmp);
        qwen35_decode_pto::VectorBarrier();
        if constexpr (SpeculativeTokens == 8) {
          wait_flag(PIPE_MTE3, PIPE_V, conv_pingpong_event);
          TileUbDataND<bfloat16_t, 1, 128> conv_output_half;
          TASSIGN(conv_output_half, conv_output_half_address);
          TCVT(conv_output_half, conv_y, RoundMode::CAST_RINT);
        } else {
          TCVT(conv_y_half, conv_y, RoundMode::CAST_RINT);
        }
        qwen35_decode_pto::VectorBarrier();
        if constexpr (SpeculativeTokens == 8) {
          set_flag(PIPE_V, PIPE_MTE3, conv_pingpong_event);
          wait_flag(PIPE_V, PIPE_MTE3, conv_pingpong_event);
          StoreBf16Row(
              conv_out_handle + token_offset,
              conv_output_half_address);
          StoreBf16Row(
              conv_state_out_handle +
                  write_slot * conv_state_stride +
                  (token_idx + 2) * conv_dim + channel_offset,
              conv_input_half_address);
          set_flag(PIPE_MTE3, PIPE_MTE2, conv_pingpong_event);
          set_flag(PIPE_MTE3, PIPE_V, conv_pingpong_event);
        } else {
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          StoreBf16Row(
              conv_out_handle + token_offset, kUbConvOutputHalf);
          StoreBf16Row(
              conv_state_out_handle +
                  write_slot * conv_state_stride +
                  (token_idx + 2) * conv_dim + channel_offset,
              kUbConvInputHalf);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        }

        TMOV(hist0, hist1);
        TMOV(hist1, hist2);
        TMOV(hist2, x_fp32);
        qwen35_decode_pto::VectorBarrier();
        if constexpr (SpeculativeTokens == 8) {
          conv_pingpong_flag = conv_pingpong_flag == 0 ? 1 : 0;
        }
      }
    }
    if constexpr (SpeculativeTokens == 8) {
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    }
  }

  // Conv output is a GM hand-off between different channel/head owners.
  qwen35_decode_pto::SyncAllAiv();

  // Phase 2: each owner keeps one complete FP32 state in UB for all S steps.
  // The K8/B4/NK8/NV24 bucket maps one batch x Q/K group to each of the
  // first 32 AIVs, caches normalized Q/K for all nine tokens, and then
  // processes the group's three value heads without recomputing Q/K.
  static_assert(!UseQkGroupCache || SpeculativeTokens == 8);
  const int32_t total_heads = batch_size * num_v_heads;
  const int32_t recurrent_loop_end =
      UseQkGroupCache ? vector_core_count * v_heads_per_k : total_heads;
  for (int32_t head_index = vector_core_idx;
       head_index < recurrent_loop_end;
       head_index += vector_core_count) {
    if constexpr (UseQkGroupCache) {
      if (vector_core_idx >= batch_size * num_k_heads) {
        break;
      }
    }
    const int32_t task_head_idx =
        UseQkGroupCache
            ? (head_index - vector_core_idx) / vector_core_count
            : 0;
    const int32_t batch_idx =
        UseQkGroupCache ? vector_core_idx / num_k_heads
                        : head_index / num_v_heads;
    const int32_t qk_head_idx =
        UseQkGroupCache ? vector_core_idx % num_k_heads
                        : (head_index % num_v_heads) /
                              v_heads_per_k;
    const int32_t head_idx =
        UseQkGroupCache
            ? qk_head_idx * v_heads_per_k + task_head_idx
            : head_index % num_v_heads;
    const bool load_norm_weight =
        !UseQkGroupCache || task_head_idx == 0;

    if constexpr (UseQkGroupCache) {
      if (task_head_idx == 0) {
        const int32_t batch_conv_offset =
            batch_idx * sequence_length * conv_dim;
        constexpr int32_t q_bf16_cache_address =
            kUbConvWeightHalf0;
        constexpr int32_t k_bf16_cache_address =
            q_bf16_cache_address +
            kDeferredNormRows * kHeadDim *
                sizeof(bfloat16_t);
        LoadStridedBf16Rows<kDeferredNormRows>(
            conv_out_handle +
                batch_conv_offset +
                qk_head_idx * kHeadDim,
            q_bf16_cache_address,
            conv_dim);
        LoadStridedBf16Rows<kDeferredNormRows>(
            conv_out_handle +
                batch_conv_offset +
                num_k_heads * kHeadDim +
                qk_head_idx * kHeadDim,
            k_bf16_cache_address,
            conv_dim);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TileUbDataND<
            bfloat16_t,
            kDeferredNormRows,
            kHeadDim> q_half_batch;
        TASSIGN(q_half_batch, q_bf16_cache_address);
        TileUbDataND<
            bfloat16_t,
            kDeferredNormRows,
            kHeadDim> k_half_batch;
        TASSIGN(k_half_batch, k_bf16_cache_address);
        TileUbDataND<float, kDeferredNormRows, kHeadDim> q_batch;
        TASSIGN(q_batch, kUbQkCacheTail);
        TileUbDataND<float, kDeferredNormRows, kHeadDim> k_batch;
        TASSIGN(k_batch, kUbQkCacheK);
        TCVT(q_batch, q_half_batch, RoundMode::CAST_NONE);
        TCVT(k_batch, k_half_batch, RoundMode::CAST_NONE);
        qwen35_decode_pto::VectorBarrier();

        constexpr int32_t qk_square_address = kUbState;
        constexpr int32_t qk_reduce_tmp_address =
            qk_square_address +
            kDeferredNormRows * kHeadDim * sizeof(float);
        constexpr int32_t qk_norm_address =
            qk_reduce_tmp_address +
            kDeferredNormRows * 64 * sizeof(float);
        constexpr int32_t qk_norm_sqrt_address =
            qk_norm_address + 16 * sizeof(float);
        static_assert(
            qk_norm_sqrt_address + 16 * sizeof(float) <= kUbV);
        NormalizeQkRows<true, kDeferredNormRows>(
            kUbQkCacheTail,
            qk_square_address,
            qk_reduce_tmp_address,
            qk_norm_address,
            qk_norm_sqrt_address);
        NormalizeQkRows<false, kDeferredNormRows>(
            kUbQkCacheK,
            qk_square_address,
            qk_reduce_tmp_address,
            qk_norm_address,
            qk_norm_sqrt_address);
      }
    }
    const int32_t read_slot = *(read_state_indices_handle + batch_idx);
    const int32_t write_slot = *(write_state_indices_handle + batch_idx);
    const int32_t accepted =
        *(num_accepted_tokens_handle + batch_idx);
    const int32_t read_checkpoint =
        read_slot * sequence_length + accepted - 1;
    const int64_t read_state_offset =
        static_cast<int64_t>(read_checkpoint) * ssm_checkpoint_stride +
        head_idx * kSsmHeadElements;

    LoadState(
        ssm_state_handle + read_state_offset,
        kUbState);
    if (load_norm_weight) {
      LoadBf16Row(norm_weight_handle, kUbNormWeightHalf);
    }
    LoadFloatScalar(a_log_handle + head_idx, kUbALogStatic);
    LoadFloatScalar(dt_bias_handle + head_idx, kUbDtBiasStatic);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (load_norm_weight) {
      TCVT(norm_weight_fp32, norm_weight_half, RoundMode::CAST_NONE);
    }
    TEXP(a_log_static, a_log_static);
    qwen35_decode_pto::VectorBarrier();

    for (int32_t token_idx = 0; token_idx < sequence_length;
         ++token_idx) {
      if constexpr (SpeculativeTokens == 8) {
        if (token_idx == 0) {
          const int32_t conv_token_offset =
              batch_idx * sequence_length * conv_dim;
          if constexpr (!UseQkGroupCache) {
            LoadQkBf16Rows(
                conv_out_handle +
                    conv_token_offset + qk_head_idx * kHeadDim,
                kUbQHalf,
                num_k_heads * kHeadDim);
          }
          LoadBf16Row(
              conv_out_handle +
                  conv_token_offset + 2 * num_k_heads * kHeadDim +
                  head_idx * kHeadDim,
              kUbVHalf);
          const int32_t scalar_offset =
              batch_idx * sequence_length * num_v_heads + head_idx;
          LoadBf16Scalar(a_handle + scalar_offset, kUbAHalf);
          LoadBf16Scalar(b_handle + scalar_offset, kUbBHalf);
          LoadBf16Row(
              z_handle +
                  batch_idx * sequence_length * v_width +
                  head_idx * kHeadDim,
              kUbZHalf);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }
      } else {
        const int32_t conv_token_offset =
            (batch_idx * sequence_length + token_idx) * conv_dim;
        LoadBf16Row(
            conv_out_handle +
                conv_token_offset + qk_head_idx * kHeadDim,
            kUbQHalf);
        LoadBf16Row(
            conv_out_handle +
                conv_token_offset + num_k_heads * kHeadDim +
                qk_head_idx * kHeadDim,
            kUbKHalf);
        LoadBf16Row(
            conv_out_handle +
                conv_token_offset + 2 * num_k_heads * kHeadDim +
                head_idx * kHeadDim,
            kUbVHalf);
        const int32_t scalar_offset =
            (batch_idx * sequence_length + token_idx) * num_v_heads +
            head_idx;
        LoadBf16Scalar(a_handle + scalar_offset, kUbAHalf);
        LoadBf16Scalar(b_handle + scalar_offset, kUbBHalf);
        LoadBf16Row(
            z_handle +
                (batch_idx * sequence_length + token_idx) * v_width +
                head_idx * kHeadDim,
            kUbZHalf);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      }
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

      if constexpr (!UseQkGroupCache) {
        TCVT(q_fp32, q_half, RoundMode::CAST_NONE);
        TCVT(k_fp32, k_half, RoundMode::CAST_NONE);
      }
      TCVT(v_fp32, v_half, RoundMode::CAST_NONE);
      if constexpr (UseDeferredNorm) {
        TileUbDataND<bfloat16_t, 1, kHeadDim> cached_z_half;
        TASSIGN(
            cached_z_half,
            kRunDeferredZHalf +
                token_idx * kHeadDim * sizeof(bfloat16_t));
        TMOV(cached_z_half, z_half);
      } else {
        TCVT(z_fp32, z_half, RoundMode::CAST_NONE);
      }
      TCVT(scalar, a_half, RoundMode::CAST_NONE);
      TCVT(scalar2, b_half, RoundMode::CAST_NONE);
      qwen35_decode_pto::VectorBarrier();
      if constexpr (SpeculativeTokens == 8) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        if (token_idx + 1 < sequence_length) {
          const int32_t next_token_idx = token_idx + 1;
          const int32_t next_conv_token_offset =
              (batch_idx * sequence_length + next_token_idx) * conv_dim;
          if constexpr (!UseQkGroupCache) {
            LoadQkBf16Rows(
                conv_out_handle +
                    next_conv_token_offset + qk_head_idx * kHeadDim,
                kUbQHalf,
                num_k_heads * kHeadDim);
          }
          LoadBf16Row(
              conv_out_handle +
                  next_conv_token_offset +
                  2 * num_k_heads * kHeadDim +
                  head_idx * kHeadDim,
              kUbVHalf);
          const int32_t next_scalar_offset =
              (batch_idx * sequence_length + next_token_idx) *
                  num_v_heads +
              head_idx;
          LoadBf16Scalar(a_handle + next_scalar_offset, kUbAHalf);
          LoadBf16Scalar(b_handle + next_scalar_offset, kUbBHalf);
          LoadBf16Row(
              z_handle +
                  (batch_idx * sequence_length + next_token_idx) *
                      v_width +
                  head_idx * kHeadDim,
              kUbZHalf);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }
      }
      if constexpr (!UseQkGroupCache) {
        qwen35_decode_pto::NormalizeQk128<true>(
            q_fp32, norm_square, norm_value, reduce_tmp, scalar_tmp);
        qwen35_decode_pto::NormalizeQk128<false>(
            k_fp32, norm_square, norm_value, reduce_tmp, scalar_tmp);
      }

      int32_t q_token_address = kUbQ;
      int32_t k_token_address = kUbK;
      if constexpr (UseQkGroupCache) {
        q_token_address =
            kUbQkCacheTail +
            token_idx * kHeadDim * sizeof(float);
        k_token_address =
            kUbQkCacheK +
            token_idx * kHeadDim * sizeof(float);
      }
      TileUbDataND<float, 1, 128> q_token;
      TASSIGN(q_token, q_token_address);
      TileUbDataND<float, 1, 128> k_token;
      TASSIGN(k_token, k_token_address);

      // decay = exp(-exp(a_log) * softplus(a + dt_bias)).
      TADD(scalar_tmp, scalar, dt_bias_static);
      qwen35_decode_pto::VectorBarrier();
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      const float gate_input = scalar_tmp.GetValue(0);
      if (gate_input > 20.0f) {
        TMOV(scalar_work, scalar_tmp);
      } else {
        TEXP(scalar_work, scalar_tmp);
        qwen35_decode_pto::VectorBarrier();
        TADDS(scalar_work, scalar_work, 1.0f);
        qwen35_decode_pto::VectorBarrier();
        TLOG(scalar_work, scalar_work);
      }
      qwen35_decode_pto::VectorBarrier();
      TMUL(scalar_tmp, a_log_static, scalar_work);
      qwen35_decode_pto::VectorBarrier();
      TMULS(scalar_tmp, scalar_tmp, -1.0f);
      qwen35_decode_pto::VectorBarrier();
      // The small-op path materializes g as BF16 before computing exp(g).
      TCVT(rounded_gate_half, scalar_tmp, RoundMode::CAST_RINT);
      qwen35_decode_pto::VectorBarrier();
      TCVT(scalar_tmp, rounded_gate_half, RoundMode::CAST_NONE);
      qwen35_decode_pto::VectorBarrier();
      TEXP(scalar_work, scalar_tmp);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      const float decay = scalar_work.GetValue(0);

      // beta = sigmoid(b).
      TMULS(scalar_tmp, scalar2, -1.0f);
      qwen35_decode_pto::VectorBarrier();
      TEXP(scalar_work, scalar_tmp);
      qwen35_decode_pto::VectorBarrier();
      TADDS(scalar_tmp, scalar_work, 1.0f);
      qwen35_decode_pto::VectorBarrier();
      TRECIP(scalar_work, scalar_tmp);
      qwen35_decode_pto::VectorBarrier();
      // torch::sigmoid(BF16) returns BF16 on the small-op path.
      TCVT(rounded_gate_half, scalar_work, RoundMode::CAST_RINT);
      qwen35_decode_pto::VectorBarrier();
      TCVT(scalar_work, rounded_gate_half, RoundMode::CAST_NONE);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      const float beta = scalar_work.GetValue(0);

      TMULS(state, state, decay);
      qwen35_decode_pto::VectorBarrier();
      {
        TileUbDataDN<float, 128, 1> k_row;
        TASSIGN(k_row, reinterpret_cast<std::uintptr_t>(k_token.data()));
        TROWEXPANDMUL(compute, state, k_row);
      }
      qwen35_decode_pto::VectorBarrier();
      qwen35_decode_pto::ColSum128(
          prediction, compute, colsum_tmp);
      TSUB(delta, v_fp32, prediction);
      qwen35_decode_pto::VectorBarrier();
      TMULS(delta, delta, beta);
      qwen35_decode_pto::VectorBarrier();
      {
        TileUbDataDN<float, 128, 1> k_row;
        TASSIGN(k_row, reinterpret_cast<std::uintptr_t>(k_token.data()));
#if defined(PTO_NPU_ARCH_A5)
        TROWEXPAND(compute, k_row);
        qwen35_decode_pto::VectorBarrier();
        TCOLEXPANDMUL(compute, compute, delta);
        qwen35_decode_pto::VectorBarrier();
        TADD(state, state, compute);
#else
        qwen35_decode_pto::OuterProductAdd128(state, delta, k_row);
#endif
      }
      qwen35_decode_pto::VectorBarrier();
      {
        TileUbDataDN<float, 128, 1> q_row;
        TASSIGN(q_row, reinterpret_cast<std::uintptr_t>(q_token.data()));
        TROWEXPANDMUL(compute, state, q_row);
      }
      qwen35_decode_pto::VectorBarrier();
      qwen35_decode_pto::ColSum128(
          prediction, compute, colsum_tmp);
      if constexpr (UseDeferredNorm) {
        TileUbDataND<bfloat16_t, 1, kHeadDim>
            cached_readout_half;
        TASSIGN(
            cached_readout_half,
            kRunDeferredReadoutHalf +
                token_idx * kHeadDim * sizeof(bfloat16_t));
        TCVT(
            cached_readout_half,
            prediction,
            RoundMode::CAST_RINT);
      } else {
        TCVT(readout_half, prediction, RoundMode::CAST_RINT);
      }
      qwen35_decode_pto::VectorBarrier();
      if constexpr (!UseDeferredNorm) {
        TMOV(norm_half, readout_half);
      }

      const int32_t write_checkpoint =
          write_slot * sequence_length + token_idx;
      const int64_t write_state_offset =
          static_cast<int64_t>(write_checkpoint) * ssm_checkpoint_stride +
          head_idx * kSsmHeadElements;
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      StoreState(
          ssm_state_out_handle + write_state_offset,
          kUbState);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);

      if constexpr (!UseDeferredNorm) {
        TCVT(norm_fp32, norm_half, RoundMode::CAST_NONE);
        qwen35_decode_pto::VectorBarrier();
        TMUL(square_fp32, norm_fp32, norm_fp32);
        qwen35_decode_pto::VectorBarrier();
        TileUbDataDN<float, 8, 1, 1, 1> rms_dn;
        TASSIGN(rms_dn, kUbRms);
        TileUbDataND<float, 1, 64, 1, 64> rms_reduce_tmp;
        TASSIGN(rms_reduce_tmp, kUbReduceTmp);
        TROWSUM(rms_dn, square_fp32, rms_reduce_tmp);
        qwen35_decode_pto::VectorBarrier();
        TMULS(rms, rms, 1.0f / 128.0f);
        qwen35_decode_pto::VectorBarrier();
        TADDS(rms, rms, 1.0e-6f);
        qwen35_decode_pto::VectorBarrier();
        TSQRT(scalar_tmp, rms);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        const float rms_value = scalar_tmp.GetValue(0);
        TMULS(norm_fp32, norm_fp32, 1.0f / rms_value);
        qwen35_decode_pto::VectorBarrier();
        TMUL(norm_fp32, norm_fp32, norm_weight_fp32);
        TileUbDataND<float, 1, 128> silu_tmp;
        TASSIGN(silu_tmp, kUbVectorScratch);
        qwen35_decode_pto::Silu<float, 1, 128>(
            gate_fp32, z_fp32, silu_tmp);
        qwen35_decode_pto::VectorBarrier();
        TMUL(norm_fp32, norm_fp32, gate_fp32);
        qwen35_decode_pto::VectorBarrier();
        TCVT(final_half, norm_fp32, RoundMode::CAST_RINT);
        qwen35_decode_pto::VectorBarrier();
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        StoreBf16Row(
            out_handle +
                (batch_idx * sequence_length + token_idx) * v_width +
                head_idx * kHeadDim,
            kUbFinalHalf);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        if constexpr (SpeculativeTokens != 8) {
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        }
      }
    }

    if constexpr (UseDeferredNorm) {
      if constexpr (true) {
      pipe_barrier(PIPE_ALL);
      constexpr int32_t input_half_batch_address = kUbStateCompute;
      constexpr int32_t z_half_batch_address =
          input_half_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(bfloat16_t);
      constexpr int32_t input_batch_address =
          z_half_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(bfloat16_t);
      constexpr int32_t z_batch_address =
          input_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(float);
      constexpr int32_t square_batch_address =
          z_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(float);
      constexpr int32_t gate_batch_address =
          square_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(float);
      constexpr int32_t final_half_batch_address =
          gate_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(float);
      constexpr int32_t silu_tmp_batch_address =
          final_half_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(bfloat16_t);
      constexpr int32_t reduce_tmp_batch_address =
          silu_tmp_batch_address +
          kRunDeferredNormRows * kHeadDim * sizeof(float);
      constexpr int32_t rms_batch_address =
          reduce_tmp_batch_address +
          kRunDeferredNormRows * 64 * sizeof(float);
      constexpr int32_t rms_batch_capacity =
          kRunDeferredNormRows <= 16 ? 16 : 32;
      constexpr int32_t rms_sqrt_batch_address =
          rms_batch_address +
          rms_batch_capacity * sizeof(float);
      static_assert(
          rms_sqrt_batch_address +
              rms_batch_capacity * sizeof(float) <=
          kUbPrediction);

      TileUbDataND<bfloat16_t, kRunDeferredNormRows, kHeadDim>
          cached_readout_half_batch;
      TASSIGN(
          cached_readout_half_batch, kRunDeferredReadoutHalf);
      TileUbDataND<bfloat16_t, kRunDeferredNormRows, kHeadDim>
          cached_z_half_batch;
      TASSIGN(cached_z_half_batch, kRunDeferredZHalf);
      TileUbDataND<bfloat16_t, kRunDeferredNormRows, kHeadDim>
          readout_half_batch;
      TASSIGN(readout_half_batch, input_half_batch_address);
      TileUbDataND<bfloat16_t, kRunDeferredNormRows, kHeadDim>
          z_half_batch;
      TASSIGN(z_half_batch, z_half_batch_address);
      TileUbDataND<float, kRunDeferredNormRows, kHeadDim> input_batch;
      TASSIGN(input_batch, input_batch_address);
      TileUbDataND<float, kRunDeferredNormRows, kHeadDim> z_batch;
      TASSIGN(z_batch, z_batch_address);
      TileUbDataND<float, kRunDeferredNormRows, kHeadDim> square_batch;
      TASSIGN(square_batch, square_batch_address);
      TileUbDataND<float, kRunDeferredNormRows, kHeadDim> gate_batch;
      TASSIGN(gate_batch, gate_batch_address);
      TileUbDataND<bfloat16_t, kRunDeferredNormRows, kHeadDim>
          final_half_batch;
      TASSIGN(final_half_batch, final_half_batch_address);
      TileUbDataND<float, kRunDeferredNormRows, kHeadDim>
          silu_tmp_batch;
      TASSIGN(silu_tmp_batch, silu_tmp_batch_address);
      TileUbDataDN<float,
                   rms_batch_capacity,
                   1,
                   kRunDeferredNormRows,
                   1> rms_batch_dn;
      TASSIGN(rms_batch_dn, rms_batch_address);
      TileUbDataND<float,
                   1,
                   rms_batch_capacity,
                   1,
                   kRunDeferredNormRows> rms_batch;
      TASSIGN(rms_batch, rms_batch_address);
      TileUbDataND<float,
                   1,
                   rms_batch_capacity,
                   1,
                   kRunDeferredNormRows> rms_sqrt_batch;
      TASSIGN(rms_sqrt_batch, rms_sqrt_batch_address);
      TileUbDataND<float, kRunDeferredNormRows, 64> reduce_tmp_batch;
      TASSIGN(reduce_tmp_batch, reduce_tmp_batch_address);

      for (int32_t row = 0; row < kRunDeferredNormRows; ++row) {
        TileUbDataND<bfloat16_t, 1, kHeadDim> cached_readout_row;
        TASSIGN(
            cached_readout_row,
            kRunDeferredReadoutHalf +
                row * kHeadDim * sizeof(bfloat16_t));
        TileUbDataND<bfloat16_t, 1, kHeadDim> cached_z_row;
        TASSIGN(
            cached_z_row,
            kRunDeferredZHalf +
                row * kHeadDim * sizeof(bfloat16_t));
        TileUbDataND<float, 1, kHeadDim> input_row;
        TASSIGN(
            input_row,
            input_batch_address +
                row * kHeadDim * sizeof(float));
        TileUbDataND<float, 1, kHeadDim> z_row;
        TASSIGN(
            z_row,
            z_batch_address +
                row * kHeadDim * sizeof(float));
        TCVT(input_row, cached_readout_row, RoundMode::CAST_NONE);
        TCVT(z_row, cached_z_row, RoundMode::CAST_NONE);
      }
      qwen35_decode_pto::VectorBarrier();
      TMUL(square_batch, input_batch, input_batch);
      qwen35_decode_pto::VectorBarrier();
      TROWSUM(rms_batch_dn, square_batch, reduce_tmp_batch);
      qwen35_decode_pto::VectorBarrier();
      TMULS(rms_batch, rms_batch, 1.0f / 128.0f);
      qwen35_decode_pto::VectorBarrier();
      TADDS(rms_batch, rms_batch, 1.0e-6f);
      qwen35_decode_pto::VectorBarrier();
      TSQRT(rms_sqrt_batch, rms_batch);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (int32_t row = 0; row < kRunDeferredNormRows; ++row) {
        TileUbDataND<float, 1, kHeadDim> input_row;
        TASSIGN(
            input_row,
            input_batch_address +
                row * kHeadDim * sizeof(float));
        const float rms_value = rms_sqrt_batch.GetValue(row);
        TMULS(input_row, input_row, 1.0f / rms_value);
      }
      qwen35_decode_pto::VectorBarrier();
      for (int32_t row = 0; row < kRunDeferredNormRows; ++row) {
        TileUbDataND<float, 1, kHeadDim> input_row;
        TASSIGN(
            input_row,
            input_batch_address +
                row * kHeadDim * sizeof(float));
        TMUL(input_row, input_row, norm_weight_fp32);
      }
      qwen35_decode_pto::VectorBarrier();
      SiluNoCopy<float, kRunDeferredNormRows, kHeadDim>(
          gate_batch, z_batch, silu_tmp_batch);
      qwen35_decode_pto::VectorBarrier();
      TMUL(input_batch, input_batch, gate_batch);
      qwen35_decode_pto::VectorBarrier();
      TCVT(final_half_batch, input_batch, RoundMode::CAST_RINT);
      qwen35_decode_pto::VectorBarrier();

      const int64_t output_base_offset =
          static_cast<int64_t>(batch_idx) *
              sequence_length * v_width +
          head_idx * kHeadDim;
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      StoreStridedBf16Rows<kRunDeferredNormRows>(
          out_handle + output_base_offset,
          final_half_batch_address,
          v_width);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      } else {
        const int64_t output_base_offset =
            static_cast<int64_t>(batch_idx) *
                sequence_length * v_width +
            head_idx * kHeadDim;
        for (int32_t row = 0; row < sequence_length; ++row) {
          TileUbDataND<bfloat16_t, 1, kHeadDim>
              cached_readout_half;
          TASSIGN(
              cached_readout_half,
              kRunDeferredReadoutHalf +
                  row * kHeadDim * sizeof(bfloat16_t));
          TileUbDataND<bfloat16_t, 1, kHeadDim> cached_z_half;
          TASSIGN(
              cached_z_half,
              kRunDeferredZHalf +
                  row * kHeadDim * sizeof(bfloat16_t));
          TCVT(norm_fp32, cached_readout_half, RoundMode::CAST_NONE);
          TCVT(z_fp32, cached_z_half, RoundMode::CAST_NONE);
          qwen35_decode_pto::VectorBarrier();
          TMUL(square_fp32, norm_fp32, norm_fp32);
          qwen35_decode_pto::VectorBarrier();
          TileUbDataDN<float, 8, 1, 1, 1> cached_rms_dn;
          TASSIGN(cached_rms_dn, kUbRms);
          TileUbDataND<float, 1, 64, 1, 64>
              cached_rms_reduce_tmp;
          TASSIGN(cached_rms_reduce_tmp, kUbReduceTmp);
          TROWSUM(
              cached_rms_dn, square_fp32, cached_rms_reduce_tmp);
          qwen35_decode_pto::VectorBarrier();
          TMULS(rms, rms, 1.0f / 128.0f);
          qwen35_decode_pto::VectorBarrier();
          TADDS(rms, rms, 1.0e-6f);
          qwen35_decode_pto::VectorBarrier();
          TSQRT(scalar_tmp, rms);
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          const float cached_rms_value = scalar_tmp.GetValue(0);
          TMULS(norm_fp32, norm_fp32, 1.0f / cached_rms_value);
          qwen35_decode_pto::VectorBarrier();
          TMUL(norm_fp32, norm_fp32, norm_weight_fp32);
          TileUbDataND<float, 1, kHeadDim> cached_silu_tmp;
          TASSIGN(cached_silu_tmp, kUbVectorScratch);
          qwen35_decode_pto::Silu<float, 1, kHeadDim>(
              gate_fp32, z_fp32, cached_silu_tmp);
          qwen35_decode_pto::VectorBarrier();
          TMUL(norm_fp32, norm_fp32, gate_fp32);
          qwen35_decode_pto::VectorBarrier();
          TCVT(final_half, norm_fp32, RoundMode::CAST_RINT);
          qwen35_decode_pto::VectorBarrier();
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          StoreBf16Row(
              out_handle + output_base_offset + row * v_width,
              kUbFinalHalf);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        }
      }
    }
  }
#endif
}

}  // namespace qwen35_mtp_decode_pto
