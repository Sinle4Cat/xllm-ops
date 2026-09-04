// Resident H->O specialization for dense Qwen3.5 prefill.
//
// This file is included from chunk_h.cpp after the common H helpers are
// defined.  It keeps scaled Q@S in L0C, retains V_new in L1 after the state
// update, and accumulates gated(QK)@V_new directly into the resident tile.
// The Vector side communicates only scaled-Q, raw/gated-QK, and the final
// combined tile through per-core mailboxes.

#ifdef __CCE_AICORE__

namespace {

constexpr int32_t kResidentHoGatedReadyFlag = 9;
constexpr int32_t kResidentHoCombinedReadyFlag = 10;
constexpr int32_t kResidentHoCombinedFreeFlag = 11;

// Permanent Vector data.  The regular non-segmented H path does not use the
// alternate-state region, so one half of the causal mask can live there for
// the whole fused H/O stage.
constexpr int32_t kResidentHoNormWeightBf16Addr = 0;
constexpr int32_t kResidentHoNormWeightFp32Addr = 512;
constexpr int32_t kResidentHoMaskAddr = 156928;

// Gating scratch.  It aliases H buffers that are dead after V_new/K_tilde and
// the state update have been published for the current chunk.
constexpr int32_t kResidentHoGAllAddr = 74496;
constexpr int32_t kResidentHoGRowsAddr = 75008;
constexpr int32_t kResidentHoRowMatrixAddr = 75520;
constexpr int32_t kResidentHoCoeffAddr = 91904;
constexpr int32_t kResidentHoQkHalfAddr = 108288;
constexpr int32_t kResidentHoQkFp32Addr = 116480;
constexpr int32_t kResidentHoQHalfAddr = 132864;

// Thirty-two rows amortize the Vector setup and event cost while leaving the
// recurrent FP32 state resident in UB.
constexpr int32_t kResidentHoOutputFp32Addr = 74496;
constexpr int32_t kResidentHoOutputComputeAddr = 90880;
constexpr int32_t kResidentHoZAddr = 99072;
constexpr int32_t kResidentHoSquareAddr = 107264;
constexpr int32_t kResidentHoReduceTmpAddr = 123648;
constexpr int32_t kResidentHoRowSumAddr = 140032;
constexpr int32_t kResidentHoRowScaleAddr = 140544;
constexpr int32_t kResidentHoOutputPublicAddr = 141568;

template <typename T, int32_t Rows, int32_t Cols,
          int32_t ValidRows = Rows, int32_t ValidCols = Cols>
using ResidentHoL1 =
    Tile<TileType::Mat, T, Rows, Cols, BLayout::ColMajor,
         ValidRows, ValidCols, SLayout::RowMajor, 512, PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols,
          int32_t ValidRows = Rows, int32_t ValidCols = Cols>
using ResidentHoUb =
    Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor,
         ValidRows, ValidCols, SLayout::NoneBox, 512, PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols,
          int32_t ValidRows = Rows, int32_t ValidCols = Cols>
using ResidentHoUbDn =
    Tile<TileType::Vec, T, Rows, Cols, BLayout::ColMajor,
         ValidRows, ValidCols, SLayout::NoneBox, 512>;

template <typename T, int32_t Elements>
AICORE inline void ResidentHoInvalidateMailbox(__gm__ T *mailbox)
{
#if defined(__DAV_C220_CUBE__)
  constexpr int32_t CacheLineBytes = 64;
  static_assert(CacheLineBytes % sizeof(T) == 0,
                "mailbox element size must divide the cache-line size");
  constexpr int32_t ElementsPerCacheLine = CacheLineBytes / sizeof(T);
  for (int32_t offset = 0; offset < Elements;
       offset += ElementsPerCacheLine) {
    dcci(reinterpret_cast<__gm__ int64_t *>(mailbox + offset),
         cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
  }
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize,
          bool FuseGatedRmsNorm>
AICORE inline void ResidentHoPrepareVectorConstants(
    __gm__ float *mask_handle,
    __gm__ GDN_PUBLIC_DTYPE *norm_weight_handle, int32_t vec_id)
{
#if defined(__DAV_C220_VEC__)
  constexpr int32_t HalfChunk = ChunkSize / 2;
  {
    GmShape2D mask_shape(HalfChunk, ChunkSize);
    GmStride2D mask_stride(ChunkSize);
    GmTensor2D<float> mask_global(
        mask_handle + static_cast<int64_t>(vec_id) * HalfChunk * ChunkSize,
        mask_shape, mask_stride);
    ResidentHoUb<float, HalfChunk, ChunkSize> mask_ub;
    TASSIGN(mask_ub, kResidentHoMaskAddr);
    TLOAD(mask_ub, mask_global);
  }
  if constexpr (FuseGatedRmsNorm) {
    GmShape2D weight_shape(1, HiddenSize);
    GmStride2D weight_stride(HiddenSize);
    GmTensor2D<GDN_PUBLIC_DTYPE> weight_global(
        norm_weight_handle, weight_shape, weight_stride);
    ResidentHoUb<GDN_PUBLIC_DTYPE, 1, HiddenSize> weight_public;
    TASSIGN(weight_public, kResidentHoNormWeightBf16Addr);
    TLOAD(weight_public, weight_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  if constexpr (FuseGatedRmsNorm) {
    ResidentHoUb<GDN_PUBLIC_DTYPE, 1, HiddenSize> weight_public;
    ResidentHoUb<float, 1, HiddenSize> weight_fp32;
    TASSIGN(weight_public, kResidentHoNormWeightBf16Addr);
    TASSIGN(weight_fp32, kResidentHoNormWeightFp32Addr);
    TCVT(weight_fp32, weight_public, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
  }
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void ResidentHoPrecomputeGroupQk(
    __gm__ ComputeT *q_handle, __gm__ ComputeT *k_handle,
    __gm__ ComputeT *raw_qk_cache, int64_t total_tokens,
    int32_t num_key_heads)
{
#if defined(__DAV_C220_CUBE__)
  static_assert(HiddenSize == ChunkSize,
                "resident H/O currently requires D == C");
  const int64_t chunk_count = total_tokens / ChunkSize;
  const int64_t work_count = chunk_count * num_key_heads;
  const int64_t core_id = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());
  const int32_t qk_stride = num_key_heads * HiddenSize;

  for (int64_t work = core_id; work < work_count; work += block_num) {
    const int64_t chunk_idx = work / num_key_heads;
    const int64_t head_g = work - chunk_idx * num_key_heads;
    const int64_t qk_offset =
        (chunk_idx * ChunkSize * num_key_heads + head_g) * HiddenSize;
    mk_o::PublishQKTile<HiddenSize, ChunkSize>(
        q_handle, k_handle, raw_qk_cache, work, qk_offset, qk_stride,
        ChunkSize, -1, 0u);
  }
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void ResidentHoCubeOutput(
    __gm__ ComputeT *combined_mailbox,
    __gm__ ComputeT *gated_qk_mailbox,
    int64_t core_id, bool wait_for_previous_output,
    TileMatL1<ComputeT, HiddenSize, HiddenSize,
              HiddenSize, HiddenSize> &state_l1,
    TileMatL1<ComputeT, ChunkSize, HiddenSize,
              ChunkSize, HiddenSize> &work_l1,
    TileMatL1<ComputeT, ChunkSize, HiddenSize,
              ChunkSize, HiddenSize> &v_new_l1,
    TileAcc<float, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> &combined_l0)
{
#if defined(__DAV_C220_CUBE__)
  static_assert(HiddenSize == ChunkSize,
                "resident H/O currently requires D == C");
  constexpr int32_t TileElements = ChunkSize * HiddenSize;

  wait_flag_dev(kResidentHoGatedReadyFlag);
  ResidentHoInvalidateMailbox<ComputeT, TileElements>(
      combined_mailbox + core_id * static_cast<int64_t>(TileElements));
  ResidentHoInvalidateMailbox<ComputeT, TileElements>(
      gated_qk_mailbox + core_id * static_cast<int64_t>(TileElements));
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  {
    GmShape2D scaled_q_shape(ChunkSize, HiddenSize);
    GmStride2D scaled_q_stride(HiddenSize);
    GmTensor2D<ComputeT> scaled_q_global(
        combined_mailbox + core_id * static_cast<int64_t>(TileElements),
        scaled_q_shape, scaled_q_stride);
    TLOAD(work_l1, scaled_q_global);
  }
  ResidentHoL1<ComputeT, ChunkSize, ChunkSize> gated_qk_l1;
  TASSIGN(gated_qk_l1, 65536);
  {
    GmShape2D gated_shape(ChunkSize, ChunkSize);
    GmStride2D gated_stride(ChunkSize);
    GmTensor2D<ComputeT> gated_global(
        gated_qk_mailbox +
            core_id * static_cast<int64_t>(ChunkSize) * ChunkSize,
        gated_shape, gated_stride);
    TLOAD(gated_qk_l1, gated_global);
  }

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO_DEBUG_V_NEW_SELF_TERM
  gemm_v0<ComputeT, float, ChunkSize, HiddenSize, ChunkSize,
          ChunkSize, HiddenSize, ChunkSize, ChunkSize, false, true>(
      v_new_l1, v_new_l1, combined_l0, true);
#else
  gemm_v0<ComputeT, float, ChunkSize, HiddenSize, HiddenSize,
          ChunkSize, HiddenSize, HiddenSize, HiddenSize, false, false>(
      work_l1, state_l1, combined_l0, true);
  gemm_v0<ComputeT, float, ChunkSize, HiddenSize, ChunkSize,
          ChunkSize, HiddenSize, ChunkSize, ChunkSize, false, false>(
      gated_qk_l1, v_new_l1, combined_l0, false);
#endif

  if (wait_for_previous_output) {
    wait_flag_dev(kResidentHoCombinedFreeFlag);
  }
  {
    GmShape2D combined_shape(ChunkSize, HiddenSize);
    GmStride2D combined_stride(HiddenSize);
    GmTensor2D<ComputeT> combined_global(
        combined_mailbox + core_id * static_cast<int64_t>(TileElements),
        combined_shape, combined_stride);
    DynAccTile<float, ChunkSize, HiddenSize> combined_store(
        ChunkSize, HiddenSize);
    TASSIGN(combined_store, 0);
    TSTORE(combined_global, combined_store);
  }
  ffts_cross_core_sync(
      PIPE_FIX,
      1 | (2 << 4) | (kResidentHoCombinedReadyFlag << 8));
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize>
AICORE inline void ResidentHoGateQkAndScaleQ(
    __gm__ ComputeT *q_handle, __gm__ float *g_handle,
    __gm__ ComputeT *raw_qk_mailbox,
    __gm__ ComputeT *scaled_q_mailbox,
    __gm__ ComputeT *gated_qk_mailbox,
    int64_t core_id, int64_t chunk_start, int64_t total_tokens,
    int32_t head, int32_t head_g, int32_t num_key_heads,
    int32_t vec_id)
{
#if defined(__DAV_C220_VEC__)
  static_assert(HiddenSize == ChunkSize,
                "resident H/O currently requires D == C");
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t RowTile = 32;
  constexpr int32_t RowTilesPerVec = HalfChunk / RowTile;
  constexpr int32_t TileElements = ChunkSize * HiddenSize;
  const int32_t q_stride = num_key_heads * HiddenSize;

  {
    GmShape2D g_shape(1, ChunkSize);
    GmStride2D g_stride(1);
    GmTensor2D<float> g_global(
        g_handle + static_cast<int64_t>(head) * total_tokens + chunk_start,
        g_shape, g_stride);
    ResidentHoUb<float, 1, ChunkSize> g_all;
    TASSIGN(g_all, kResidentHoGAllAddr);
    TLOAD(g_all, g_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  const int64_t chunk_idx = chunk_start / ChunkSize;
  const int64_t raw_qk_offset =
      (chunk_idx * num_key_heads + head_g) *
      static_cast<int64_t>(TileElements);

  for (int32_t row_tile_idx = 0;
       row_tile_idx < RowTilesPerVec; ++row_tile_idx) {
    const int32_t local_row_start = row_tile_idx * RowTile;
    const int32_t row_start = vec_id * HalfChunk + local_row_start;

    ResidentHoUb<float, 1, ChunkSize> g_all;
    ResidentHoUb<float, 1, RowTile> g_rows;
    ResidentHoUbDn<float, RowTile, 1> g_rows_col;
    ResidentHoUb<float, RowTile, ChunkSize> row_matrix;
    ResidentHoUb<float, RowTile, ChunkSize> coefficients;
    ResidentHoUb<float, RowTile, ChunkSize> causal_mask;
    TASSIGN(g_all, kResidentHoGAllAddr);
    TASSIGN(g_rows, kResidentHoGRowsAddr);
    TASSIGN(g_rows_col, kResidentHoGRowsAddr);
    TASSIGN(row_matrix, kResidentHoRowMatrixAddr);
    TASSIGN(coefficients, kResidentHoCoeffAddr);
    TASSIGN(
        causal_mask,
        kResidentHoMaskAddr +
            local_row_start * ChunkSize * static_cast<int32_t>(sizeof(float)));

    ResidentHoUb<float, 1, RowTile> g_rows_source;
    TASSIGN(
        g_rows_source,
        kResidentHoGAllAddr +
            row_start * static_cast<int32_t>(sizeof(float)));
    TMOV(g_rows, g_rows_source);
    TROWEXPAND(row_matrix, g_rows_col);
    TCOLEXPAND(coefficients, g_all);
    TSUB(coefficients, row_matrix, coefficients);
    pipe_barrier(PIPE_V);
    TMINS(coefficients, coefficients, 0.0f);
    pipe_barrier(PIPE_V);
    TEXP(coefficients, coefficients);
    pipe_barrier(PIPE_V);
    TMUL(coefficients, coefficients, causal_mask);

    ResidentHoUb<ComputeT, RowTile, ChunkSize> qk_half;
    ResidentHoUb<float, RowTile, ChunkSize> qk_fp32;
    TASSIGN(qk_half, kResidentHoQkHalfAddr);
    TASSIGN(qk_fp32, kResidentHoQkFp32Addr);
    {
      GmShape2D qk_shape(RowTile, ChunkSize);
      GmStride2D qk_stride_gm(ChunkSize);
      GmTensor2D<ComputeT> qk_global(
          raw_qk_mailbox + raw_qk_offset +
              static_cast<int64_t>(row_start) * ChunkSize,
          qk_shape, qk_stride_gm);
      TLOAD(qk_half, qk_global);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(qk_fp32, qk_half, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TMUL(qk_fp32, qk_fp32, coefficients);
    pipe_barrier(PIPE_V);
    TCVT(qk_half, qk_fp32, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      GmShape2D gated_shape(RowTile, ChunkSize);
      GmStride2D gated_stride(ChunkSize);
      GmTensor2D<ComputeT> gated_global(
          gated_qk_mailbox +
              core_id * static_cast<int64_t>(ChunkSize) * ChunkSize +
              static_cast<int64_t>(row_start) * ChunkSize,
          gated_shape, gated_stride);
      TSTORE(gated_global, qk_half);
    }

    TEXP(g_rows, g_rows);
    pipe_barrier(PIPE_V);
    TROWEXPAND(row_matrix, g_rows_col);

    // The QK store reads qk_half on MTE3.  Order the following Q load before
    // reusing the same Vector scratch in the next conversion sequence.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    ResidentHoUb<ComputeT, RowTile, HiddenSize> q_half;
    ResidentHoUb<float, RowTile, HiddenSize> q_fp32;
    TASSIGN(q_half, kResidentHoQHalfAddr);
    TASSIGN(q_fp32, kResidentHoQkFp32Addr);
    {
      const int64_t q_offset =
          ((chunk_start + row_start) * num_key_heads + head_g) *
          HiddenSize;
      GmShape2D q_shape(RowTile, HiddenSize);
      GmStride2D q_gm_stride(q_stride);
      GmTensor2D<ComputeT> q_global(
          q_handle + q_offset, q_shape, q_gm_stride);
      TLOAD(q_half, q_global);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(q_fp32, q_half, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TMUL(q_fp32, q_fp32, row_matrix);
    pipe_barrier(PIPE_V);
    TCVT(q_half, q_fp32, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      GmShape2D scaled_q_shape(RowTile, HiddenSize);
      GmStride2D scaled_q_stride(HiddenSize);
      GmTensor2D<ComputeT> scaled_q_global(
          scaled_q_mailbox +
              core_id * static_cast<int64_t>(TileElements) +
              static_cast<int64_t>(row_start) * HiddenSize,
          scaled_q_shape, scaled_q_stride);
      TSTORE(scaled_q_global, q_half);
    }
  }

  ffts_cross_core_sync(
      PIPE_MTE3,
      1 | (2 << 4) | (kResidentHoGatedReadyFlag << 8));
#endif
}

template <int32_t HiddenSize, int32_t Rows,
          bool FuseGatedRmsNorm>
AICORE inline void ResidentHoStoreOutputRows(
    __gm__ GDN_PUBLIC_DTYPE *output_handle,
    __gm__ GDN_PUBLIC_DTYPE *z_handle, int64_t output_offset,
    int32_t row_stride)
{
#if defined(__DAV_C220_VEC__)
  constexpr float OutputScale = 0.08837890625f;
  constexpr int32_t PublicAddr =
      std::is_same_v<ComputeT, GDN_PUBLIC_DTYPE>
          ? kResidentHoOutputComputeAddr
          : kResidentHoOutputPublicAddr;

  ResidentHoUb<float, Rows, HiddenSize> output_fp32;
  ResidentHoUb<ComputeT, Rows, HiddenSize> output_compute;
  ResidentHoUb<GDN_PUBLIC_DTYPE, Rows, HiddenSize> output_public;
  TASSIGN(output_fp32, kResidentHoOutputFp32Addr);
  TASSIGN(output_compute, kResidentHoOutputComputeAddr);
  TASSIGN(output_public, PublicAddr);
  TCVT(output_compute, output_fp32, pto::RoundMode::CAST_NONE);

#ifdef MEGA_CHUNK_GDN_RESIDENT_HO_DEBUG_RAW_COMBINED
  pipe_barrier(PIPE_V);
  TCVT(output_public, output_fp32, pto::RoundMode::CAST_NONE);
#else
  if constexpr (FuseGatedRmsNorm) {
    {
      GmShape2D z_shape(Rows, HiddenSize);
      GmStride2D z_stride(row_stride);
      GmTensor2D<GDN_PUBLIC_DTYPE> z_global(
          z_handle + output_offset, z_shape, z_stride);
      ResidentHoUb<GDN_PUBLIC_DTYPE, Rows, HiddenSize> z_public;
      TASSIGN(z_public, kResidentHoZAddr);
      TLOAD(z_public, z_global);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pipe_barrier(PIPE_V);
    TCVT(output_fp32, output_compute, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TMULS(output_fp32, output_fp32, OutputScale);
    pipe_barrier(PIPE_V);
    TCVT(output_compute, output_fp32, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TCVT(output_fp32, output_compute, pto::RoundMode::CAST_NONE);
    if constexpr (!std::is_same_v<ComputeT, GDN_PUBLIC_DTYPE>) {
      pipe_barrier(PIPE_V);
      TCVT(output_public, output_fp32, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TCVT(output_fp32, output_public, pto::RoundMode::CAST_NONE);
    }

    mk_o::NormalizeRmsRows<
        HiddenSize, Rows, kResidentHoOutputFp32Addr,
        kResidentHoSquareAddr, kResidentHoReduceTmpAddr,
        kResidentHoRowSumAddr, kResidentHoRowScaleAddr>();

    ResidentHoUb<float, 1, HiddenSize> norm_weight_fp32;
    TASSIGN(norm_weight_fp32, kResidentHoNormWeightFp32Addr);
    TCOLEXPANDMUL(output_fp32, output_fp32, norm_weight_fp32);

    ResidentHoUb<float, Rows, HiddenSize> scratch;
    ResidentHoUb<float, Rows, HiddenSize> reduce_tmp;
    TASSIGN(scratch, kResidentHoSquareAddr);
    TASSIGN(reduce_tmp, kResidentHoReduceTmpAddr);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    ResidentHoUb<GDN_PUBLIC_DTYPE, Rows, HiddenSize> z_public;
    TASSIGN(z_public, kResidentHoZAddr);
    TCVT(scratch, z_public, pto::RoundMode::CAST_NONE);
    TNEG(reduce_tmp, scratch);
    TEXP(reduce_tmp, reduce_tmp);
    TADDS(reduce_tmp, reduce_tmp, 1.0f);
    pipe_barrier(PIPE_V);
    TMUL(output_fp32, output_fp32, scratch);
    TDIV(output_fp32, output_fp32, reduce_tmp);
    TCVT(output_public, output_fp32, pto::RoundMode::CAST_ROUND);
  } else if constexpr (!std::is_same_v<ComputeT, GDN_PUBLIC_DTYPE>) {
    pipe_barrier(PIPE_V);
    TCVT(output_fp32, output_compute, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TCVT(output_public, output_fp32, pto::RoundMode::CAST_NONE);
  }
#endif

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  {
    GmShape2D output_shape(Rows, HiddenSize);
    GmStride2D output_stride(row_stride);
    GmTensor2D<GDN_PUBLIC_DTYPE> output_global(
        output_handle + output_offset, output_shape, output_stride);
    ResidentHoUb<GDN_PUBLIC_DTYPE, Rows, HiddenSize> output_store;
    TASSIGN(output_store, PublicAddr);
    TSTORE(output_global, output_store);
  }
  // The next row tile reuses OutputComputeAddr. Keep its MTE2 load from
  // overwriting data that the current output store is still consuming.
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize,
          bool FuseGatedRmsNorm>
AICORE inline void ResidentHoConsumeCombined(
    __gm__ ComputeT *combined_mailbox,
    __gm__ GDN_PUBLIC_DTYPE *output_handle,
    __gm__ GDN_PUBLIC_DTYPE *z_handle,
    int64_t core_id, int64_t chunk_start, int32_t head,
    int32_t num_heads, int32_t vec_id, bool drain_output)
{
#if defined(__DAV_C220_VEC__)
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t RowTile = 32;
  constexpr int32_t RowTilesPerVec = HalfChunk / RowTile;
  constexpr int32_t TileElements = ChunkSize * HiddenSize;
  const int32_t output_stride = num_heads * HiddenSize;

  wait_flag_dev(kResidentHoCombinedReadyFlag);
  for (int32_t row_tile_idx = 0;
       row_tile_idx < RowTilesPerVec; ++row_tile_idx) {
    const int32_t row_start =
        vec_id * HalfChunk + row_tile_idx * RowTile;
    ResidentHoUb<ComputeT, RowTile, HiddenSize> combined_half;
    ResidentHoUb<float, RowTile, HiddenSize> output_fp32;
    TASSIGN(combined_half, kResidentHoOutputComputeAddr);
    TASSIGN(output_fp32, kResidentHoOutputFp32Addr);
    {
      GmShape2D combined_shape(RowTile, HiddenSize);
      GmStride2D combined_stride(HiddenSize);
      GmTensor2D<ComputeT> combined_global(
          combined_mailbox +
              core_id * static_cast<int64_t>(TileElements) +
              static_cast<int64_t>(row_start) * HiddenSize,
          combined_shape, combined_stride);
      TLOAD(combined_half, combined_global);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(output_fp32, combined_half, pto::RoundMode::CAST_NONE);

    const int64_t output_offset =
        (chunk_start * static_cast<int64_t>(num_heads) + head) *
            HiddenSize +
        static_cast<int64_t>(row_start) * output_stride;
    ResidentHoStoreOutputRows<HiddenSize, RowTile,
                              FuseGatedRmsNorm>(
        output_handle, z_handle, output_offset, output_stride);
  }

  if (drain_output) {
    pipe_barrier(PIPE_ALL);
  }

  ffts_cross_core_sync(
      PIPE_MTE2,
      1 | (2 << 4) | (kResidentHoCombinedFreeFlag << 8));
#endif
}

}  // namespace

#endif  // __CCE_AICORE__
