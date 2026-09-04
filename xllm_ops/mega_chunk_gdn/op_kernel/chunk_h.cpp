// ============================================================================
// chunk_h_kernel.cpp — Recurrent hidden state update for GatedDeltaNet
//
// Mathematical recurrence per chunk c:
//   S_{c+1} = exp(g_last) * S_c  +  K^T @ V
//
// where g_last = exp(g[valid-1]) is the chunk's final gate value, S is the
// D×D hidden state, K ∈ ℝ^{C×D}, V ∈ ℝ^{C×D}, and g ∈ ℝ^C is the per-token
// gate.
//
// ── Cube phase (two GEMMs per chunk, sequentially): ──────────────────────
//   1. WS = W @ S       project current state through W (wy_fast output)
//      W ∈ ℝ^{C×D}, S ∈ ℝ^{D×D}  →  WS ∈ ℝ^{C×D}
//   2. KV = K^T @ V     outer product of keys and values (transpose_A!)
//      K stored as D×C, V ∈ ℝ^{C×D}  →  KV ∈ ℝ^{D×D}
//
// ── Vec phase (two sub-blocks handle upper/lower C/2 rows): ─────────────
//   For each chunk:
//     1. Load K, G (pre-transposed), U (from wy_fast)
//     2. Compute coeff[i] = exp(g[i] - g[valid-1])  — time-decay scaling
//        Uses TROWEXPAND to broadcast coefficients across D columns
//     3. Scale K: K_scaled[i,:] = K[i,:] * coeff[i]
//     4. Load WS from Cube workspace, compute V_new = U - WS (residual)
//     5. Store V_new and K_scaled to workspace for Cube's next iteration
//     6. Update state: S = exp(g_last) * S + KV (from Cube workspace)
//     7. Store final state FS after last chunk
//
// Cross-core sync: Cube→Vec flags for WS/KV ready, Vec→Cube flags for
// K/S ready.
//
// Inputs:
//   K  [total_tokens, Hg, D] ComputeT   — keys (BSND layout; GQA/MQA group heads)
//   W  [total_tokens, H, D]  ComputeT   — wy_fast output (BSND layout)
//   U  [total_tokens, H, D]  ComputeT   — values pre-residual (BSND layout)
//   G  [H, total_tokens]     float  — pre-transposed cumulative gates
//   S  [total_chunks, H, D, D] ComputeT — per-chunk state snapshots (output)
//   V  [total_tokens, H, D]  ComputeT   — residual-corrected values (output)
//   FS [batch, H, D, D]      ComputeT   — final state per sequence (output)
//   H0 [batch, H, D, D]      ComputeT   — optional initial state per sequence
//   workspace [per-core scratch]     — Cube↔Vec communication buffer
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B/L0C (Cube GEMM registers)
//   GM → UB (Vec-accessible, on-chip SRAM)
//   Cross-core sync via FFTS (Fast Fine-grained Task Synchronization)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This is the most complex kernel in the GDN suite. It implements the
// recurrent state update, requiring sequential chunk processing (chunks
// within a sequence CANNOT be parallelized — each depends on the previous).
//
// Key PTO APIs (numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data        (DMA: GM→L1 or GM→UB)
//   TSTORE(gm, src)         — gm_data = src        (DMA: UB/L0C→GM)
//   TASSIGN(tile, addr)     — tile = memory[addr]   (bind tile to buffer address)
//   TCVT(dst, src, mode)    — converts between float and ComputeT
//   TMOV(dst, src)          — dst = src.clone()
//   TADD(d, a, b)           — d = a + b
//   TSUB(d, a, b)           — d = a - b
//   TMUL(d, a, b)           — d = a * b
//   TMULS(d, s, scalar)     — d = s * scalar       (scalar multiply)
//   TADDS(d, s, scalar)     — d = s + scalar       (scalar add)
//   TEXP(d, s)              — d = torch.exp(s)
//   TEXPANDS(tile, scalar)  — tile[:] = scalar     (fill with constant)
//   TROWEXPAND(2d, col)     — 2d[i,j] = col[i]    (broadcast col across row dim)
//   TFILLPAD(dst, src)      — zero-fill L1 tile padding (for tail chunks)
//   TEXTRACT(l0, l1, r, c)  — L1 sub-tile → L0A/L0B
//   TRESHAPE(zn, nz)        — reinterpret layout NZ↔ZN (logical transpose, free)
//   TMATMUL(C, A, B)        — C = A @ B (Cube GEMM, ComputeT inputs → FP32 accum)
//   set_flag/wait_flag      — pipe sync within same core
//   ffts_cross_core_sync    — cross-core signal Cube↔Vec
//   wait_flag_dev(flag)     — wait for cross-core signal
//   GetValue(idx)           — read a single scalar from a UB tile (slow, use sparingly)
//
// ── Workspace memory layout (shared between Cube and Vec via GM) ──────
// The workspace is field-major so adjacent cores do not all hit GM with the
// same 128 KiB stride:
//   WS_WS [C×D]:  Cube writes WS = W @ S here → Vec reads it
//   WS_K  [D×C]:  Vec writes K_scaled here → Cube reads it for KV = K^T @ V
//   WS_S  [D×D]:  Vec writes current state S here → Cube reads it for GEMM 1
//   WS_KV [D×D]:  Cube writes KV = K^T @ V here → Vec reads it to update S
//
// Data flow per chunk (think of it as a ping-pong between Cube and Vec):
//   Vec: write S₀ to WS_S → signal Cube (flag 3)
//   Cube: read S from WS_S, load W → compute WS = W@S → write WS_WS → signal Vec (flag 0)
//   Vec: read WS, compute V_new = U - WS, compute K_scaled → write WS_K → signal Cube (flag 1)
//   Cube: read K from WS_K, load V → compute KV = K^T@V → write WS_KV → signal Vec (flag 2)
//   Vec: read KV, update S = exp(g_last)*S + KV → write S to WS_S → signal Cube (flag 3)
//   ... repeat for next chunk ...
// ============================================================================

#include <pto/pto-inst.hpp>
#include <type_traits>
#include "acl/acl.h"
#include "gdn_sync.h"
using namespace pto;

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

#ifndef GDN_PAIR_DEBUG_STAGE
#define GDN_PAIR_DEBUG_STAGE 0
#endif

#ifdef __CCE_AICORE__

namespace {

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                           pto::BLayout::ColMajor, pto::DYNAMIC,
                           pto::DYNAMIC, pto::SLayout::RowMajor, 512,
                           pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                             pto::BLayout::RowMajor, pto::DYNAMIC,
                             pto::DYNAMIC, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols>
using DynAccTile = pto::TileAcc<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512,
                              pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0A = pto::TileLeft<T, Rows, Cols, RowValid, ColValid>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0B = pto::TileRight<T, Rows, Cols, RowValid, ColValid>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                               pto::BLayout::RowMajor, RowValid, ColValid,
                               pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                               pto::BLayout::ColMajor, RowValid, ColValid,
                               pto::SLayout::NoneBox, 512, PadVal>;

// PTO cheat sheet for the recurrent kernel:
//   - `GlobalTensor<T>` is a GM tensor view with explicit runtime shape/stride.
//   - `Tile<..., Mat, ...>` lives in L1 and feeds Cube matmul instructions.
//   - `Tile<..., Vec, ...>` lives in UB for elementwise vector work.
//   - `TileAcc<T, ...>` is a Cube accumulator tile.
//   - `TLOAD` / `TSTORE` are DMA copies between GM and on-chip memory.
//   - `TROWEXPAND` broadcasts a column vector across the feature dimension.
//   - `TFILLPAD(_INPLACE)` zero-pads tail rows so full-tile code can still run.

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail = K, bool transpose_A = false,
          bool transpose_B = false>
AICORE PTO_INLINE void
gemm_v0(std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                           TileMatL1<T1, M, K, validM, validK>> &A,
        std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                           TileMatL1<T1, K, N, validK, validN>> &B,
        pto::TileAcc<T2, M, N, validM, validN> &C, bool clear)
{
  // Local K-sliced matmul helper:
  //   C = A @ B
  // PTO exposes the L1/L0 staging explicitly, so this stays as a tiny file-
  // local helper instead of a shared wrapper.
  //
  // PyTorch mental model:
  //   C = 0
  //   for k0 in range(0, K, kL0Size):
  //       C += A[:, k0:k1] @ B[k0:k1, :]
  constexpr uint32_t kL0Size = 128;
  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; ++kL0Idx) {
    const bool initflag = clear && (kL0Idx == 0);
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * K_tail);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * K_tail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }
    } else {
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * kL0Size);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * kL0Size, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);

  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

#if defined(__DAV_C220_CUBE__)
template <int32_t D, int32_t C, int32_t FlagBase,
          bool WaitStateReady = true, bool PublishReady = true>
AICORE PTO_INLINE void paired_cube_ws(
    __gm__ ComputeT *W_handle, __gm__ ComputeT *S_handle,
    __gm__ ComputeT *workspace_handle, int64_t head,
    int64_t chunk_start, int64_t chunk_offset, int32_t ci,
    int32_t valid, int32_t H, int64_t ws_ws_base,
    TileMatL1<ComputeT, D, D, D, D> &s_l1,
    TileMatL1<ComputeT, C, D, C, D> &w_l1,
    TileAcc<float, C, D, C, D> &ws_l0)
{
  if constexpr (WaitStateReady) {
    wait_flag_dev(FlagBase + 3);
  }

  const int64_t w_offset = (chunk_start * H + head) * D;
  {
    GmShape2D w_shape(valid, D);
    GmStride2D w_stride(H * D);
    GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
    TLOAD(w_l1, w_global);
    if (valid != C) {
      TFILLPAD(w_l1, w_l1);
    }
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  {
    constexpr int32_t DD = D * D;
    const int64_t state_offset =
        ((chunk_offset + ci) * H + head) * DD;
    GmShape2D s_shape(D, D);
    GmStride2D s_stride(D);
    GmTensor2D<ComputeT> s_global(S_handle + state_offset, s_shape, s_stride);
    TLOAD(s_l1, s_global);
  }
#ifdef MEGA_STOP_AFTER_H
  if constexpr (GDN_PAIR_DEBUG_STAGE == 8) {
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    ffts_cross_core_sync(PIPE_FIX,
                         1 | (2 << 4) | (FlagBase << 8));
    return;
  }
#endif

#ifdef MEGA_STOP_AFTER_H
  if constexpr (GDN_PAIR_DEBUG_STAGE == 14 ||
                GDN_PAIR_DEBUG_STAGE == 15) {
    if constexpr (GDN_PAIR_DEBUG_STAGE == 14) {
      ffts_cross_core_sync(PIPE_MTE2,
                           1 | (2 << 4) | (FlagBase << 8));
      return;
    }
    TileMatL0A<ComputeT, C, D, C, D> w_l0a;
    TileMatL0B<ComputeT, D, D, D, D> s_l0b;
    TASSIGN(w_l0a, 0);
    TASSIGN(s_l0b, 0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(w_l0a, w_l1, 0, 0);
    TEXTRACT(s_l0b, s_l1, 0, 0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL(ws_l0, w_l0a, s_l0b);
    ffts_cross_core_sync(PIPE_M, 1 | (2 << 4) | (FlagBase << 8));
    return;
  }
#endif

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
      w_l1, s_l1, ws_l0, true);
#ifdef MEGA_STOP_AFTER_H
  if constexpr (GDN_PAIR_DEBUG_STAGE == 9) {
    ffts_cross_core_sync(PIPE_M,
                         1 | (2 << 4) | (FlagBase << 8));
    return;
  }
#endif

  {
    GmShape2D ws_shape(C, D);
    GmStride2D ws_stride(D);
    GmTensor2D<ComputeT> ws_global(workspace_handle + ws_ws_base, ws_shape,
                                  ws_stride);
    DynAccTile<float, C, D> ws_store(C, D);
    TASSIGN(ws_store, 0);
    TSTORE(ws_global, ws_store);
  }
  if constexpr (PublishReady) {
    ffts_cross_core_sync(PIPE_FIX,
                         1 | (2 << 4) | (FlagBase << 8));
  }
}

template <int32_t D, int32_t C>
AICORE PTO_INLINE void paired_cube_qs(
    __gm__ ComputeT *Q_handle, __gm__ ComputeT *S_handle, int64_t head,
    int64_t head_g, int64_t chunk_start, int64_t chunk_offset, int32_t ci,
    int32_t valid, int32_t H, int32_t Hg,
    TileMatL1<ComputeT, D, D, D, D> &s_l1,
    TileMatL1<ComputeT, C, D, C, D> &q_l1,
    TileAcc<float, C, D, C, D> &ws_l0)
{
  constexpr int32_t DD = D * D;
  const int64_t q_offset = (chunk_start * Hg + head_g) * D;
  GmShape2D q_shape(valid, D);
  GmStride2D q_stride(Hg * D);
  GmTensor2D<ComputeT> q_global(Q_handle + q_offset, q_shape, q_stride);
  DynMatL1<ComputeT, C, D> q_l1_load(valid, D);
  TASSIGN(q_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
  TLOAD(q_l1_load, q_global);
  if (valid != C) {
    TFILLPAD(q_l1_load, q_l1_load);
  }

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
      q_l1, s_l1, ws_l0, true);

  const int64_t qs_offset = ((chunk_offset + ci) * H + head) * DD;
  GmShape2D qs_shape(C, D);
  GmStride2D qs_stride(D);
  GmTensor2D<ComputeT> qs_global(S_handle + qs_offset, qs_shape, qs_stride);
  DynAccTile<float, C, D> qs_store(C, D);
  TASSIGN(qs_store, 0);
  TSTORE(qs_global, qs_store);
}

template <int32_t D, int32_t C, int32_t FlagBase,
          bool WaitInputReady = true, bool PublishReady = true>
AICORE PTO_INLINE void paired_cube_kv(
    __gm__ ComputeT *V_handle, __gm__ ComputeT *workspace_handle,
    int32_t valid, int32_t v_stride, int64_t v_offset,
    int64_t ws_k_base, int64_t ws_kv_base,
    TileMatL1<ComputeT, D, C, D, C> &k_l1,
    TileMatL1<ComputeT, C, D, C, D> &v_l1,
    TileAcc<float, D, D, D, D> &kv_l0)
{
  if constexpr (WaitInputReady) {
    wait_flag_dev(FlagBase + 1);
  }
  {
    GmShape2D k_shape(D, C);
    GmStride2D k_stride(C);
    GmTensor2D<ComputeT> k_global(workspace_handle + ws_k_base, k_shape,
                                 k_stride);
    DynMatL1<ComputeT, D, C> k_l1_load(D, C);
    TASSIGN(k_l1_load,
            (D * D + C * D) * static_cast<int32_t>(sizeof(ComputeT)));
    TLOAD(k_l1_load, k_global);
  }
  {
    GmShape2D v_shape(valid, D);
    GmStride2D v_gm_stride(v_stride);
    GmTensor2D<ComputeT> v_global(V_handle + v_offset, v_shape,
                                 v_gm_stride);
    DynMatL1<ComputeT, C, D> v_l1_load(valid, D);
    TASSIGN(v_l1_load,
            (D * D + C * D + D * C) *
                static_cast<int32_t>(sizeof(ComputeT)));
    TLOAD(v_l1_load, v_global);
    if (valid != C) {
      TFILLPAD(v_l1_load, v_l1_load);
    }
  }

  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, D, D, C, D, D, C, C, true, false>(
      k_l1, v_l1, kv_l0, true);
  {
    GmShape2D kv_shape(D, D);
    GmStride2D kv_stride(D);
    GmTensor2D<ComputeT> kv_global(workspace_handle + ws_kv_base, kv_shape,
                                  kv_stride);
    DynAccTile<float, D, D> kv_store(D, D);
    TASSIGN(kv_store, C * D * static_cast<int32_t>(sizeof(float)));
    TSTORE(kv_global, kv_store);
  }
  if constexpr (PublishReady) {
    ffts_cross_core_sync(
        PIPE_FIX, 1 | (2 << 4) | ((FlagBase + 2) << 8));
  }
}

#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
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
            ChunkSize, HiddenSize> &combined_l0);
#endif

AICORE PTO_INLINE void wait_wy_tile_ready(
    __gm__ int32_t *ready_handle, int64_t owner, int64_t slot)
{
  constexpr int64_t ReadyStride = 16;
  AscendC::GlobalTensor<int32_t> ready_global;
  ready_global.SetGlobalBuffer(ready_handle);
  for (int32_t vid = 0; vid < 2; ++vid) {
    const int64_t offset = (owner * 2 + vid) * ReadyStride;
    while (true) {
      __asm__ __volatile__("");
      AscendC::DataCacheCleanAndInvalid<
          int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
          AscendC::DcciDst::CACHELINE_OUT>(ready_global[offset]);
      __asm__ __volatile__("");
      if (ready_global.GetValue(offset) >= slot + 1) {
        break;
      }
    }
  }
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
}

template <int32_t D, int32_t C, int32_t FlagBase,
          bool FuseResidentOutput = false>
AICORE PTO_INLINE void fused_wy_h_cube_chunk(
    __gm__ ComputeT *Q_handle, __gm__ ComputeT *K_handle,
    __gm__ ComputeT *WyV_handle, __gm__ ComputeT *A1_handle,
    __gm__ ComputeT *A2_handle, __gm__ int32_t *WyReady_handle,
    __gm__ ComputeT *S_handle,
    __gm__ ComputeT *V_handle, __gm__ ComputeT *workspace_handle,
    int64_t head, int64_t head_g, int32_t ci, int32_t H, int32_t Hg,
    int64_t ws_ws_base, int64_t ws_u_base, int64_t ws_k_base,
    int64_t ws_kv_base, int64_t v_head_base, int64_t v_chunk_stride,
    uint32_t wy_workspace_slots, uint32_t wy_group_size,
    TileMatL1<ComputeT, D, D, D, D> &s_l1,
    TileMatL1<ComputeT, C, D, C, D> &w_l1,
    TileMatL1<ComputeT, C, D, C, D> &q_l1,
    TileMatL1<ComputeT, D, C, D, C> &k_l1,
    TileMatL1<ComputeT, C, D, C, D> &v_l1,
    TileAcc<float, C, D, C, D> &ws_l0,
    TileAcc<float, D, D, D, D> &kv_l0,
    int32_t output_v_stride = D,
    bool emit_precomputed_qs = true,
    bool use_public_u = false
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
    ,
    __gm__ ComputeT *resident_raw_qk_mailbox = nullptr,
    __gm__ ComputeT *resident_combined_mailbox = nullptr,
    __gm__ ComputeT *resident_gated_qk_mailbox = nullptr,
    int64_t resident_core_id = 0,
    bool resident_output_in_flight = false
#endif
    )
{
  static_assert(D == C, "The fused WY->H path requires D == C.");
  constexpr int64_t TileElements = static_cast<int64_t>(C) * C;
  const int64_t block_num = static_cast<int64_t>(get_block_num());
  const int64_t matrix_id = static_cast<int64_t>(ci) * H + head;
  const int64_t group = matrix_id / static_cast<int64_t>(wy_group_size);
  const int64_t owner = group % block_num;
  const int64_t slot =
      (group / block_num) * static_cast<int64_t>(wy_group_size) +
      matrix_id % static_cast<int64_t>(wy_group_size);
  const int64_t wy_offset =
      (owner * static_cast<int64_t>(wy_workspace_slots) + slot) *
      TileElements;
  const int64_t chunk_start = static_cast<int64_t>(ci) * C;
  wait_wy_tile_ready(WyReady_handle, owner, slot);

  if (!use_public_u) {
    // U = A2 @ V. Keep the public FP16 rounding boundary, but write the result
    // into the head-chain owner's contiguous mailbox instead of BSND GM.
    {
      GmShape2D a2_shape(C, C);
      GmStride2D a2_stride(C);
      GmTensor2D<ComputeT> a2_global(A2_handle + wy_offset, a2_shape,
                                     a2_stride);
      TLOAD(s_l1, a2_global);

      const int64_t input_v_offset = (chunk_start * H + head) * D;
      GmShape2D input_v_shape(C, D);
      GmStride2D input_v_stride(H * D);
      GmTensor2D<ComputeT> input_v_global(
          WyV_handle + input_v_offset, input_v_shape, input_v_stride);
      TLOAD(v_l1, input_v_global);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    gemm_v0<ComputeT, float, C, D, C, C, D, C, C, false, false>(
        s_l1, v_l1, ws_l0, true);
    {
      GmShape2D u_shape(C, D);
      GmStride2D u_stride(D);
      GmTensor2D<ComputeT> u_global(workspace_handle + ws_u_base, u_shape,
                                    u_stride);
      DynAccTile<float, C, D> u_store(C, D);
      TASSIGN(u_store, 0);
      TSTORE(u_global, u_store);
    }
  }

  // W = A1 @ K, then round directly from L0C into L1. This preserves the
  // original FP32->FP16 boundary while eliminating W's GM store and reload.
  {
    GmShape2D a1_shape(C, C);
    GmStride2D a1_stride(C);
    GmTensor2D<ComputeT> a1_global(A1_handle + wy_offset, a1_shape,
                                   a1_stride);
    TLOAD(s_l1, a1_global);

    const int64_t input_k_offset = (chunk_start * Hg + head_g) * D;
    GmShape2D input_k_shape(C, D);
    GmStride2D input_k_stride(Hg * D);
    GmTensor2D<ComputeT> input_k_global(
        K_handle + input_k_offset, input_k_shape, input_k_stride);
    TLOAD(v_l1, input_k_global);
  }
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, C, D, C, C, D, C, C, false, false>(
      s_l1, v_l1, ws_l0, true);

  {
    GmShape2D w_shape(C, D);
    GmStride2D w_stride(D);
    GmTensor2D<ComputeT> w_global(
        workspace_handle + ws_ws_base, w_shape, w_stride);
    DynAccTile<float, C, D> w_store(C, D);
    TASSIGN(w_store, 0);
    TSTORE(w_global, w_store);
  }
  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID2);
  {
    GmShape2D w_shape(C, D);
    GmStride2D w_stride(D);
    GmTensor2D<ComputeT> w_global(
        workspace_handle + ws_ws_base, w_shape, w_stride);
    TLOAD(w_l1, w_global);
  }

  wait_flag_dev(FlagBase + 3);
  {
    constexpr int32_t DD = D * D;
    const int64_t state_offset =
        (static_cast<int64_t>(ci) * H + head) * DD;
    GmShape2D s_shape(D, D);
    GmStride2D s_stride(D);
    GmTensor2D<ComputeT> s_global(S_handle + state_offset, s_shape,
                                  s_stride);
    TLOAD(s_l1, s_global);
  }
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
      w_l1, s_l1, ws_l0, true);
  {
    GmShape2D ws_shape(C, D);
    GmStride2D ws_stride(D);
    GmTensor2D<ComputeT> ws_global(workspace_handle + ws_ws_base, ws_shape,
                                   ws_stride);
    DynAccTile<float, C, D> ws_store(C, D);
    TASSIGN(ws_store, 0);
    TSTORE(ws_global, ws_store);
  }
  ffts_cross_core_sync(
      PIPE_FIX, 1 | (2 << 4) | (FlagBase << 8));

  if (emit_precomputed_qs) {
    paired_cube_qs<D, C>(Q_handle, S_handle, head, head_g, chunk_start, 0, ci,
                         C, H, Hg, s_l1, q_l1, ws_l0);
  }
  paired_cube_kv<D, C, FlagBase, true, true>(
      V_handle, workspace_handle, C, output_v_stride,
      v_head_base + static_cast<int64_t>(ci) * v_chunk_stride, ws_k_base,
      ws_kv_base, k_l1, v_l1, kv_l0);
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
  if constexpr (FuseResidentOutput) {
    ResidentHoCubeOutput<D, C>(
        resident_combined_mailbox, resident_gated_qk_mailbox,
        resident_core_id, resident_output_in_flight, s_l1, q_l1, v_l1,
        ws_l0);
  }
#endif
}

template <int32_t D, int32_t C, int32_t FlagBase>
AICORE PTO_INLINE void fused_wy_h_cube_head_range(
    __gm__ ComputeT *Q_handle, __gm__ ComputeT *K_handle,
    __gm__ ComputeT *WyV_handle, __gm__ ComputeT *A1_handle,
    __gm__ ComputeT *A2_handle, __gm__ int32_t *WyReady_handle,
    __gm__ ComputeT *S_handle, __gm__ ComputeT *V_handle,
    __gm__ ComputeT *workspace_handle, int64_t head,
    int32_t chunk_begin, int32_t chunk_end, int32_t H, int32_t Hg,
    uint32_t wy_workspace_slots, uint32_t wy_group_size,
    TileMatL1<ComputeT, D, D, D, D> &s_l1,
    TileMatL1<ComputeT, C, D, C, D> &w_l1,
    TileMatL1<ComputeT, C, D, C, D> &q_l1,
    TileMatL1<ComputeT, D, C, D, C> &k_l1,
    TileMatL1<ComputeT, C, D, C, D> &v_l1,
    TileAcc<float, C, D, C, D> &ws_l0,
    TileAcc<float, D, D, D, D> &kv_l0)
{
  constexpr int32_t DD = D * D;
  constexpr int32_t WS_FIELD_STRIDE =
      DD + GDN_H_WORKSPACE_PAD_BYTES / sizeof(ComputeT);
  const int64_t cid = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());
  const int64_t ws_core_offset = cid * WS_FIELD_STRIDE;
  const int64_t ws_field_span = block_num * WS_FIELD_STRIDE;
  const int64_t ws_ws_base = ws_core_offset;
  const int64_t ws_k_base = ws_field_span + ws_core_offset;
  const int64_t ws_u_base = 2 * ws_field_span + ws_core_offset;
  const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
  const int64_t head_g = head / (H / Hg);
  const int64_t v_head_base = head * static_cast<int64_t>(C) * D;
  const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;

  for (int32_t ci = chunk_begin; ci < chunk_end; ++ci) {
    fused_wy_h_cube_chunk<D, C, FlagBase>(
        Q_handle, K_handle, WyV_handle, A1_handle, A2_handle,
        WyReady_handle, S_handle, V_handle, workspace_handle, head, head_g,
        ci, H, Hg, ws_ws_base, ws_u_base, ws_k_base, ws_kv_base,
        v_head_base, v_chunk_stride, wy_workspace_slots, wy_group_size,
        s_l1, w_l1, q_l1, k_l1, v_l1, ws_l0, kv_l0);
  }
}
#endif

template <int32_t D, int32_t C, int32_t FlagBase>
AICORE PTO_INLINE void paired_cube_chunk(
    __gm__ ComputeT *Q_handle, __gm__ ComputeT *W_handle,
    __gm__ ComputeT *S_handle, __gm__ ComputeT *V_handle,
    __gm__ ComputeT *workspace_handle, int64_t head, int64_t head_g,
    int32_t ci, int32_t H, int32_t Hg, int64_t ws_ws_base,
    int64_t ws_k_base, int64_t ws_kv_base, int64_t v_head_base,
    int64_t v_chunk_stride,
    TileMatL1<ComputeT, D, D, D, D> &s_l1,
    TileMatL1<ComputeT, C, D, C, D> &w_l1,
    TileMatL1<ComputeT, C, D, C, D> &q_l1,
    TileMatL1<ComputeT, D, C, D, C> &k_l1,
    TileMatL1<ComputeT, C, D, C, D> &v_l1,
    TileAcc<float, C, D, C, D> &ws_l0,
    TileAcc<float, D, D, D, D> &kv_l0)
{
  const int64_t chunk_start = static_cast<int64_t>(ci) * C;
  paired_cube_ws<D, C, FlagBase, true, true>(
      W_handle, S_handle, workspace_handle, head, chunk_start, 0, ci, C, H,
      ws_ws_base, s_l1, w_l1, ws_l0);
  paired_cube_qs<D, C>(Q_handle, S_handle, head, head_g, chunk_start, 0, ci,
                       C, H, Hg, s_l1, q_l1, ws_l0);
  paired_cube_kv<D, C, FlagBase, true, true>(
      V_handle, workspace_handle, C, D,
      v_head_base + static_cast<int64_t>(ci) * v_chunk_stride, ws_k_base,
      ws_kv_base, k_l1, v_l1, kv_l0);
}
#endif

#if defined(__DAV_C220_VEC__)
template <int32_t D, int32_t C, int32_t FlagBase,
          bool PublishReady = true>
AICORE PTO_INLINE void paired_vec_init_zero(
    __gm__ ComputeT *S_handle, int64_t head, int64_t chunk_offset,
    int32_t H, int32_t vid,
    TileUbDataND<float, C / 2, D, C / 2, D> &state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D> &state_half)
{
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t DD = D * D;
  TEXPANDS(state_ub, 0.0f);
  TCVT(state_half, state_ub, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  {
    const int64_t s_out_offset = (chunk_offset * H + head) * DD;
    GmShape2D s_out_shape(HalfC, D);
    GmStride2D s_out_stride(D);
    GmTensor2D<ComputeT> s_out_global(
        S_handle + s_out_offset + vid * HalfC * D, s_out_shape,
        s_out_stride);
    TSTORE(s_out_global, state_half);
  }
  if constexpr (PublishReady) {
    ffts_cross_core_sync(
        PIPE_MTE3, 1 | (2 << 4) | ((FlagBase + 3) << 8));
  } else {
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  }
}

template <int32_t D, int32_t C, int32_t FlagBase>
AICORE PTO_INLINE void paired_vec_load_state(
    __gm__ ComputeT *S_handle, int64_t head, int64_t chunk_offset,
    int32_t chunk_idx, int32_t H, int32_t vid,
    TileUbDataND<float, C / 2, D, C / 2, D> &state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D> &state_half)
{
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t DD = D * D;
  const int64_t state_offset =
      ((chunk_offset + chunk_idx) * H + head) * DD + vid * HalfC * D;
  GmShape2D state_shape(HalfC, D);
  GmStride2D state_stride(D);
  GmTensor2D<ComputeT> state_global(S_handle + state_offset, state_shape,
                                    state_stride);
  TLOAD(state_half, state_global);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(state_ub, state_half, pto::RoundMode::CAST_NONE);

  ffts_cross_core_sync(
      PIPE_V, 1 | (2 << 4) | ((FlagBase + 3) << 8));
}

template <int32_t C>
AICORE PTO_INLINE void publish_segment_state(
    __gm__ int32_t *h_o_ready_handle, int64_t head,
    int32_t ready_chunk, int32_t vid)
{
  constexpr int64_t H_O_READY_STRIDE = 16;
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);

  const int64_t ready_offset =
      head * H_O_READY_STRIDE + 1 + vid;
  AscendC::GlobalTensor<int32_t> ready_global;
  ready_global.SetGlobalBuffer(h_o_ready_handle);
  ready_global.SetValue(ready_offset, ready_chunk);
  __asm__ __volatile__("");
  AscendC::DataCacheCleanAndInvalid<
      int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
      AscendC::DcciDst::CACHELINE_ALL>(ready_global[ready_offset]);
  __asm__ __volatile__("");
}

template <int32_t C>
AICORE PTO_INLINE void wait_segment_state(
    __gm__ int32_t *h_o_ready_handle, int64_t head,
    int32_t ready_chunk, int32_t vid)
{
  constexpr int64_t H_O_READY_STRIDE = 16;
  const int64_t ready_offset =
      head * H_O_READY_STRIDE + 1 + vid;
  AscendC::GlobalTensor<int32_t> ready_global;
  ready_global.SetGlobalBuffer(h_o_ready_handle);
  while (true) {
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(ready_global[ready_offset]);
    __asm__ __volatile__("");
    if (ready_global.GetValue(ready_offset) >= ready_chunk) {
      break;
    }
  }
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
}

template <int32_t D, int32_t C, int32_t FlagBase, int32_t GUbAddr,
          int32_t CoeffUbAddr, int32_t UUbAddr, bool FusedWyH = false>
AICORE PTO_INLINE void paired_vec_front(
    __gm__ ComputeT *K_handle, __gm__ ComputeT *U_handle,
    __gm__ float *G_handle, __gm__ ComputeT *V_handle,
    __gm__ ComputeT *workspace_handle, int64_t head, int64_t head_g,
    int64_t chunk_start, int64_t valid, int32_t ci, int32_t H,
    int32_t Hg, int64_t total_tokens, int32_t vid, int32_t v_stride,
    int64_t v_head_base, int64_t v_chunk_stride, int64_t ws_ws_base,
    int64_t ws_k_base,
    TileUbDataND<float, C / 2, D, C / 2, D> &state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D,
                 pto::PadValue::Zero> &k_ub_half,
    TileUbDataND<float, 1, C, 1, C, pto::PadValue::Zero> &g_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D,
                 pto::PadValue::Zero> &u_ub_half,
    TileUbDataND<float, C / 2, D, C / 2, D> &k_ub,
    TileUbDataND<float, 1, 8, 1, 8> &g_last_tail_ub,
    TileUbDataND<float, 1, 64, 1, 64> &coeff_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &u_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &ws_ub,
    int64_t ws_u_base = 0, bool use_public_u = false)
{
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t ExpTailElements = 8;
  const int32_t qkv_stride = H * D;
  const int32_t k_stride = Hg * D;
  int32_t valid_rows =
      static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
  if (valid_rows < 0) valid_rows = 0;
  if (valid_rows > HalfC) valid_rows = HalfC;
  const int64_t v_offset =
      v_head_base + static_cast<int64_t>(ci) * v_chunk_stride +
      static_cast<int64_t>(vid) * HalfC * v_stride;

  if (!FusedWyH || use_public_u) {
    const int64_t u_offset =
        (chunk_start * H + head) * D + vid * HalfC * qkv_stride;
    if (valid_rows > 0) {
      GmShape2D u_shape(valid_rows, D);
      GmStride2D u_stride(qkv_stride);
      GmTensor2D<ComputeT> u_global(U_handle + u_offset, u_shape, u_stride);
      TLOAD(u_ub_half, u_global);
    } else {
      TEXPANDS(u_ub, 0.0f);
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
    }
  }

  const int64_t k_offset =
      (chunk_start * Hg + head_g) * D + vid * HalfC * k_stride;
  if (valid_rows > 0) {
    GmShape2D k_shape(valid_rows, D);
    GmStride2D k_gm_stride(k_stride);
    GmTensor2D<ComputeT> k_global(K_handle + k_offset, k_shape,
                                 k_gm_stride);
    TLOAD(k_ub_half, k_global);
  } else {
    TEXPANDS(k_ub, 0.0f);
    TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
  }

  {
    GmShape2D g_shape(1, static_cast<int32_t>(valid));
    GmStride2D g_stride(1);
    GmTensor2D<float> g_global(G_handle + head * total_tokens + chunk_start,
                               g_shape, g_stride);
    TLOAD(g_ub, g_global);
  }

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

  TileUbDataND<float, 1, 64, 1, 64> g_ub_temp;
  TASSIGN(g_ub_temp, GUbAddr + vid * 64 * sizeof(float));
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
  TADDS(coeff_ub, g_ub_temp, -g_last);
  pipe_barrier(PIPE_V);
  TNEG(coeff_ub, coeff_ub);
  pipe_barrier(PIPE_V);
  TEXP(coeff_ub, coeff_ub);
  if (valid == C) {
    TEXP(g_last_tail_ub, g_last_tail_ub);
  } else {
    TEXP(g_ub, g_ub);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  TileUbDataDN<float, HalfC, 1, HalfC, 1> coeff_col_ub;
  TASSIGN(coeff_col_ub, CoeffUbAddr);
  TileUbDataND<float, HalfC, D, HalfC, D> coeff_2d_ub;
  TASSIGN(coeff_2d_ub, UUbAddr);
  TROWEXPAND(coeff_2d_ub, coeff_col_ub);
  pipe_barrier(PIPE_V);
  TMUL(k_ub, k_ub, coeff_2d_ub);
  pipe_barrier(PIPE_V);
  TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
  if (!FusedWyH || use_public_u) {
    TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);
  }

  wait_flag_dev(FlagBase);
  if constexpr (FusedWyH) {
    if (!use_public_u) {
      GmShape2D u_shape(HalfC, D);
      GmStride2D u_stride(D);
      GmTensor2D<ComputeT> u_global(
          workspace_handle + ws_u_base + vid * HalfC * D, u_shape,
          u_stride);
      TLOAD(u_ub_half, u_global);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
  }
  {
    GmShape2D ws_shape(HalfC, D);
    GmStride2D ws_stride(D);
    GmTensor2D<ComputeT> ws_global(
        workspace_handle + ws_ws_base + vid * HalfC * D, ws_shape,
        ws_stride);
    TLOAD(u_ub_half, ws_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
  TSUB(u_ub, u_ub, ws_ub);
  TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  if (valid_rows > 0) {
    GmShape2D v_shape(valid_rows, D);
    GmStride2D v_gm_stride(v_stride);
    GmTensor2D<ComputeT> v_global(V_handle + v_offset, v_shape,
                                 v_gm_stride);
    TSTORE(v_global, u_ub_half);
  }
  {
    GmShape2D k_shape(HalfC, D);
    GmStride2D k_ws_stride(D);
    GmTensor2D<ComputeT> k_global(
        workspace_handle + ws_k_base + vid * HalfC * D, k_shape,
        k_ws_stride);
    TSTORE(k_global, k_ub_half);
  }
  ffts_cross_core_sync(
      PIPE_MTE3, 1 | (2 << 4) | ((FlagBase + 1) << 8));
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  const float exp_g_last =
      valid == C ? g_last_tail_ub.GetValue(ExpTailElements - 1)
                 : g_ub.GetValue(static_cast<int32_t>(valid) - 1);
  TMULS(state_ub, state_ub, exp_g_last);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}

template <int32_t D, int32_t C, int32_t FlagBase>
AICORE PTO_INLINE void paired_vec_finish(
    __gm__ ComputeT *S_handle, __gm__ ComputeT *workspace_handle,
    __gm__ int32_t *h_o_ready_handle, int64_t head,
    int64_t chunk_offset, int32_t ci, int32_t num_chunks, int32_t H,
    int32_t vid, int64_t ws_kv_base,
    bool emit_precomputed_qs,
    TileUbDataND<float, C / 2, D, C / 2, D> &state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D> &state_half,
    TileUbDataND<float, C / 2, D, C / 2, D> &kv_ub)
{
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t DD = D * D;
  constexpr int64_t H_O_READY_STRIDE = 16;
  wait_flag_dev(FlagBase + 2);
  if (emit_precomputed_qs && vid == 0) {
    const int64_t ready_offset = head * H_O_READY_STRIDE;
    AscendC::GlobalTensor<int32_t> ready_global;
    ready_global.SetGlobalBuffer(h_o_ready_handle);
    ready_global.SetValue(ready_offset, ci + 1);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<
        int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_ALL>(ready_global[ready_offset]);
    __asm__ __volatile__("");
  }
  {
    GmShape2D kv_shape(HalfC, D);
    GmStride2D kv_stride(D);
    GmTensor2D<ComputeT> kv_global(
        workspace_handle + ws_kv_base + vid * HalfC * D, kv_shape,
        kv_stride);
    TLOAD(state_half, kv_global);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(kv_ub, state_half, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_ALL);
  TADD(state_ub, state_ub, kv_ub);
  TCVT(state_half, state_ub, pto::RoundMode::CAST_NONE);

  if (ci + 1 < num_chunks) {
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    const int64_t s_out_offset = ((chunk_offset + ci + 1) * H + head) * DD;
    {
      GmShape2D s_out_shape(HalfC, D);
      GmStride2D s_out_stride(D);
      GmTensor2D<ComputeT> s_out_global(
          S_handle + s_out_offset + vid * HalfC * D, s_out_shape,
          s_out_stride);
      TSTORE(s_out_global, state_half);
    }
    ffts_cross_core_sync(
        PIPE_MTE3, 1 | (2 << 4) | ((FlagBase + 3) << 8));
  }
}

template <int32_t D, int32_t C, int32_t FlagBase, int32_t GUbAddr,
          int32_t CoeffUbAddr, int32_t UUbAddr, bool FusedWyH = false>
AICORE PTO_INLINE void paired_vec_chunk(
    __gm__ ComputeT *K_handle, __gm__ ComputeT *U_handle,
    __gm__ float *G_handle, __gm__ ComputeT *S_handle,
    __gm__ ComputeT *V_handle, __gm__ ComputeT *workspace_handle,
    __gm__ int32_t *h_o_ready_handle, int64_t head, int64_t head_g,
    int32_t ci, int32_t num_chunks, int32_t H, int32_t Hg,
    int64_t total_tokens, int32_t vid, int64_t ws_ws_base,
    int64_t ws_k_base, int64_t ws_kv_base, int64_t v_head_base,
    int64_t v_chunk_stride,
    TileUbDataND<float, C / 2, D, C / 2, D> &state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D> &state_half,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D,
                 pto::PadValue::Zero> &k_ub_half,
    TileUbDataND<float, 1, C, 1, C, pto::PadValue::Zero> &g_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D,
                 pto::PadValue::Zero> &u_ub_half,
    TileUbDataND<float, C / 2, D, C / 2, D> &k_ub,
    TileUbDataND<float, 1, 8, 1, 8> &g_last_tail_ub,
    TileUbDataND<float, 1, 64, 1, 64> &coeff_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &u_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &ws_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &kv_ub,
    int64_t ws_u_base = 0, int32_t output_v_stride = D,
    bool emit_precomputed_qs = true, bool use_public_u = false)
{
  const int64_t chunk_start = static_cast<int64_t>(ci) * C;
  paired_vec_front<D, C, FlagBase, GUbAddr, CoeffUbAddr, UUbAddr,
                   FusedWyH>(
      K_handle, U_handle, G_handle, V_handle, workspace_handle, head, head_g,
      chunk_start, C, ci, H, Hg, total_tokens, vid, output_v_stride, v_head_base,
      v_chunk_stride, ws_ws_base, ws_k_base, state_ub, k_ub_half, g_ub,
      u_ub_half, k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, ws_u_base,
      use_public_u);
  paired_vec_finish<D, C, FlagBase>(
      S_handle, workspace_handle, h_o_ready_handle, head, 0, ci, num_chunks,
      H, vid, ws_kv_base, emit_precomputed_qs, state_ub, state_half, kv_ub);
}

template <int32_t D, int32_t C, bool StoreFinalStateCache>
AICORE PTO_INLINE void paired_vec_store_final(
    __gm__ ComputeT *FS_handle, __gm__ float *final_state_cache,
    __gm__ int32_t *state_indices, int64_t head, int32_t H,
    int32_t vid, int32_t state_index_stride, int64_t state_cache_slots,
    int64_t output_final_state,
    TileUbDataND<float, C / 2, D, C / 2, D> &state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D> &state_half)
{
  if (output_final_state == 0) return;
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t DD = D * D;
  TCVT(state_half, state_ub, pto::RoundMode::CAST_NONE);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  if constexpr (StoreFinalStateCache) {
    TCVT(state_ub, state_half, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    const int32_t state_index = state_indices[0] * state_index_stride;
    if (state_index >= 0 && state_index < state_cache_slots) {
      const int64_t cache_offset =
          (static_cast<int64_t>(state_index) * H + head) * DD +
          vid * HalfC * D;
      GmShape2D cache_shape(HalfC, D);
      GmStride2D cache_stride(D);
      GmTensor2D<float> cache_global(final_state_cache + cache_offset,
                                     cache_shape, cache_stride);
      TSTORE(cache_global, state_ub);
    }
  } else {
    const int64_t fs_offset = head * DD + vid * HalfC * D;
    GmShape2D fs_shape(HalfC, D);
    GmStride2D fs_stride(D);
    GmTensor2D<ComputeT> fs_global(FS_handle + fs_offset, fs_shape,
                                  fs_stride);
    TSTORE(fs_global, state_half);
  }
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template <int32_t D, int32_t C, int32_t FlagBase, int32_t GUbAddr,
          int32_t CoeffUbAddr, int32_t UUbAddr,
          bool StoreFinalStateCache, bool InitializeZero, bool LoadState,
          bool StoreFinal, bool PublishSegment, bool SignalCompletion>
AICORE PTO_INLINE void fused_wy_h_vec_head_range(
    __gm__ ComputeT *K_handle, __gm__ ComputeT *U_handle,
    __gm__ float *G_handle, __gm__ ComputeT *S_handle,
    __gm__ ComputeT *V_handle, __gm__ ComputeT *FS_handle,
    __gm__ ComputeT *workspace_handle,
    __gm__ int32_t *h_o_ready_handle,
    __gm__ float *final_state_cache, __gm__ int32_t *state_indices,
    int64_t head, int32_t chunk_begin, int32_t chunk_end,
    int32_t num_chunks, int32_t H, int32_t Hg, int64_t total_tokens,
    int32_t vid, int32_t state_index_stride, int64_t state_cache_slots,
    int64_t output_final_state,
    TileUbDataND<float, C / 2, D, C / 2, D> &state_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D> &state_half,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D,
                 pto::PadValue::Zero> &k_ub_half,
    TileUbDataND<float, 1, C, 1, C, pto::PadValue::Zero> &g_ub,
    TileUbDataND<ComputeT, C / 2, D, C / 2, D,
                 pto::PadValue::Zero> &u_ub_half,
    TileUbDataND<float, C / 2, D, C / 2, D> &k_ub,
    TileUbDataND<float, 1, 8, 1, 8> &g_last_tail_ub,
    TileUbDataND<float, 1, 64, 1, 64> &coeff_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &u_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &ws_ub,
    TileUbDataND<float, C / 2, D, C / 2, D> &kv_ub)
{
  static_assert(!(InitializeZero && LoadState));
  constexpr int32_t DD = D * D;
  constexpr int32_t WS_FIELD_STRIDE =
      DD + GDN_H_WORKSPACE_PAD_BYTES / sizeof(ComputeT);
  const int64_t cid = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());
  const int64_t ws_core_offset = cid * WS_FIELD_STRIDE;
  const int64_t ws_field_span = block_num * WS_FIELD_STRIDE;
  const int64_t ws_ws_base = ws_core_offset;
  const int64_t ws_k_base = ws_field_span + ws_core_offset;
  const int64_t ws_u_base = 2 * ws_field_span + ws_core_offset;
  const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
  const int64_t head_g = head / (H / Hg);
  const int64_t v_head_base = head * static_cast<int64_t>(C) * D;
  const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;

  if constexpr (InitializeZero) {
    paired_vec_init_zero<D, C, FlagBase, true>(
        S_handle, head, 0, H, vid, state_ub, state_half);
  }
  if constexpr (LoadState) {
    wait_segment_state<C>(h_o_ready_handle, head, chunk_begin, vid);
    paired_vec_load_state<D, C, FlagBase>(
        S_handle, head, 0, chunk_begin, H, vid, state_ub, state_half);
  }

  for (int32_t ci = chunk_begin; ci < chunk_end; ++ci) {
    paired_vec_chunk<D, C, FlagBase, GUbAddr, CoeffUbAddr, UUbAddr, true>(
        K_handle, U_handle, G_handle, S_handle, V_handle, workspace_handle,
        h_o_ready_handle, head, head_g, ci, num_chunks, H, Hg,
        total_tokens, vid, ws_ws_base, ws_k_base, ws_kv_base, v_head_base,
        v_chunk_stride, state_ub, state_half, k_ub_half, g_ub, u_ub_half,
        k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub, ws_u_base);
  }

  if constexpr (PublishSegment) {
    publish_segment_state<C>(h_o_ready_handle, head, chunk_end, vid);
  }
  if constexpr (StoreFinal) {
    paired_vec_store_final<D, C, StoreFinalStateCache>(
        FS_handle, final_state_cache, state_indices, head, H, vid,
        state_index_stride, state_cache_slots, output_final_state, state_ub,
        state_half);
  }
  if constexpr (SignalCompletion) {
    ffts_cross_core_sync(
        PIPE_MTE3, 1 | (2 << 4) | ((FlagBase + 3) << 8));
  }
}
#endif

} // namespace

#endif

#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
#include "chunk_ho.cpp"
#endif

#if defined(__DAV_C220_CUBE__)
#define GDN_CHUNK_H_KERNEL chunk_h_kernel_aic
#elif defined(__DAV_C220_VEC__)
#define GDN_CHUNK_H_KERNEL chunk_h_kernel_aiv
#else
#define GDN_CHUNK_H_KERNEL chunk_h_kernel
#endif

template <int32_t HiddenSize, int32_t ChunkSize,
          bool StoreFinalStateCache = false,
          bool LoadInitialStateCache = false
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
          ,
          bool FuseResidentOutput = false,
          bool FuseResidentGatedRmsNorm = false
#endif
          >
AICORE void GDN_CHUNK_H_KERNEL(
    __gm__ ComputeT *Q_handle, __gm__ ComputeT *K_handle,
    __gm__ ComputeT *W_handle, __gm__ ComputeT *U_handle,
    __gm__ float *G_handle,
    __gm__ ComputeT *S_handle, __gm__ ComputeT *V_handle, __gm__ ComputeT *FS_handle,
    __gm__ ComputeT *H0_handle,
    int64_t has_initial_state,
    int64_t output_final_state,
    __gm__ ComputeT *workspace_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_heads,
    uint32_t num_key_heads,
    uint32_t precompute_qs,
    __gm__ int32_t *h_o_ready_handle,
    uint64_t ffts_addr,
    __gm__ float *initial_state_cache,
    __gm__ int32_t *initial_state_indices,
    __gm__ float *final_state_cache,
    __gm__ int32_t *state_indices,
    int32_t state_index_stride,
    int64_t state_cache_slots,
    bool fuse_wy_h,
    bool fuse_wy_h_public_u,
    __gm__ ComputeT *wy_v_handle,
    __gm__ ComputeT *wy_a1_handle,
    __gm__ ComputeT *wy_a2_handle,
    __gm__ int32_t *wy_ready_handle,
    uint32_t wy_workspace_slots,
    uint32_t wy_group_size
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
    ,
    __gm__ float *resident_mask_handle,
    __gm__ ComputeT *resident_raw_qk_mailbox,
    __gm__ ComputeT *resident_combined_mailbox,
    __gm__ ComputeT *resident_gated_qk_mailbox,
    __gm__ GDN_PUBLIC_DTYPE *resident_output_handle,
    __gm__ GDN_PUBLIC_DTYPE *resident_z_handle,
    __gm__ GDN_PUBLIC_DTYPE *resident_norm_weight_handle
#endif
    )
{
  // chunk_h advances the recurrent hidden state chunk by chunk:
  //   ws_i      = W_i @ S_i
  //   v_i_new   = U_i - ws_i
  //   k_i_tilde = exp(g_last - g_i) * K_i
  //   S_{i+1}   = exp(g_last) * S_i + k_i_tilde^T @ v_i_new.
  //
  // Shapes for one (sequence, head, chunk):
  //   W_i, U_i, K_i, V_i_new : [valid, D]
  //   S_i, S_{i+1}           : [D, D]
  //
  // PyTorch / NumPy sketch:
  //   ws = W_i @ S_i
  //   v_new = U_i - ws
  //   decay = exp(g_last - g_i)[:, None]
  //   k_tilde = decay * K_i
  //   kv = k_tilde.T @ v_new
  //   S = exp(g_last) * S + kv
  //
  // PTO split:
  //   Cube forms the two matmuls (`W_i @ S_i` and `K_i^T @ V_i_new`).
  //   Vec does the elementwise gating/decay and carries the running state.
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  gdn_sync::InitAddress(ffts_addr);

  constexpr int32_t D = HiddenSize;
  constexpr int32_t C = ChunkSize;
  const int32_t H = static_cast<int32_t>(num_heads);
  const int32_t Hg = static_cast<int32_t>(num_key_heads);
  if (H <= 0 || Hg <= 0 || (H % Hg) != 0) return;
  const int32_t GROUP = H / Hg;
  constexpr int32_t HalfC = C / 2;
  const int32_t BSND_QKV_STRIDE = H * D;
  const int32_t BSND_K_STRIDE = Hg * D;
  constexpr int32_t DD = D * D;
#if defined(GDN_A5_CUBE_KERNEL)
  // C310 exposes 128 KiB of L0C; 0x20000 is the first byte past it.  The two
  // GEMMs are separated by a Cube->Vector->Cube handshake, so reuse slot 0.
  constexpr int32_t SecondL0CAddr = 0;
#else
  constexpr int32_t SecondL0CAddr = C * D * sizeof(float);
#endif

  static_assert(GDN_H_WORKSPACE_PAD_BYTES % sizeof(ComputeT) == 0);
  constexpr int32_t WS_FIELD_STRIDE =
      DD + GDN_H_WORKSPACE_PAD_BYTES / sizeof(ComputeT);

  TileMatL1<ComputeT, D, D, D, D> s_l1;
  TASSIGN(s_l1, 0);
  TileMatL1<ComputeT, C, D, C, D> w_l1;
  TASSIGN(w_l1, D * D * sizeof(ComputeT));
  TileMatL1<ComputeT, C, D, C, D> q_l1;
  TASSIGN(q_l1, D * D * sizeof(ComputeT));
  TileAcc<float, C, D, C, D> ws_l0;
  TASSIGN(ws_l0, 0);
  TileMatL1<ComputeT, D, C, D, C> k_l1;
  TASSIGN(k_l1, (DD + C * D) * sizeof(ComputeT));
  TileMatL1<ComputeT, C, D, C, D> v_l1;
  TASSIGN(v_l1, (DD + C * D + D * C) * sizeof(ComputeT));
  TileAcc<float, D, D, D, D> kv_l0;
  TASSIGN(kv_l0, SecondL0CAddr);

  constexpr int32_t G_BLOCK_UB = 0;
  // Leading UB scratch: legacy kernels used ``C * H * sizeof(float)``, which overflows UB when
  // Keep the same slack as the historical H=16 build (8192 bytes).
  constexpr int32_t ZERO_UB =
      ChunkSize * 16 * static_cast<int32_t>(sizeof(float));
  constexpr int32_t S_UB = ZERO_UB + 64 * sizeof(float);
  constexpr int32_t K_UB_HALF = S_UB + HalfC * D * sizeof(float);
  constexpr int32_t G_UB = K_UB_HALF + HalfC * D * sizeof(ComputeT);
  constexpr int32_t U_UB_HALF = G_UB + C * sizeof(float);
  constexpr int32_t K_UB = U_UB_HALF + HalfC * D * sizeof(ComputeT);
  constexpr int32_t G_V_UB = K_UB + HalfC * D * sizeof(float);
  constexpr int32_t COEFF_UB = G_V_UB + 64 * sizeof(float);
  constexpr int32_t U_UB = COEFF_UB + 64 * sizeof(float);
  // k_ub is dead after k_tilde is rounded to BF16. Reuse that 32 KiB region
  // for W@S instead of keeping a second full FP32 tile live.
  constexpr int32_t WS_UB = K_UB;
  constexpr int32_t KV_UB = U_UB_HALF;
  constexpr int32_t S_UB_HALF = U_UB + HalfC * D * sizeof(float);
  constexpr int32_t S_ALT_UB =
      S_UB_HALF + HalfC * D * sizeof(ComputeT);

  TileUbDataND<float, HalfC, D, HalfC, D> s_ub;
  TASSIGN(s_ub, S_UB);
  TileUbDataND<ComputeT, HalfC, D, HalfC, D, pto::PadValue::Zero> k_ub_half;
  TASSIGN(k_ub_half, K_UB_HALF);
  TileUbDataND<float, 1, C, 1, C, pto::PadValue::Zero> g_ub;
  TASSIGN(g_ub, G_UB);
  TileUbDataND<ComputeT, HalfC, D, HalfC, D> s_ub_half;
  TASSIGN(s_ub_half, S_UB_HALF);
  TileUbDataND<ComputeT, HalfC, D, HalfC, D, pto::PadValue::Zero> u_ub_half;
  TASSIGN(u_ub_half, U_UB_HALF);
  TileUbDataND<float, HalfC, D, HalfC, D> k_ub;
  TASSIGN(k_ub, K_UB);
  constexpr int32_t ExpTailElements = 8;
  TileUbDataND<float, 1, ExpTailElements, 1, ExpTailElements>
      g_last_tail_ub;
  TASSIGN(
      g_last_tail_ub,
      G_UB + (C - ExpTailElements) * static_cast<int32_t>(sizeof(float)));
  TileUbDataND<float, 1, 64, 1, 64> coeff_ub;
  TASSIGN(coeff_ub, COEFF_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> u_ub;
  TASSIGN(u_ub, U_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> ws_ub;
  TASSIGN(ws_ub, WS_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> kv_ub;
  TASSIGN(kv_ub, KV_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> s_alt_ub;
  TASSIGN(s_alt_ub, S_ALT_UB);

  auto vid = get_subblockid();
  constexpr int64_t H_O_READY_STRIDE = 16;
  TileUbDataND<int32_t, 1, 8, 1, 8> h_o_ready_ub;
  TASSIGN(h_o_ready_ub, ZERO_UB);
  AscendC::GlobalTensor<int32_t> h_o_ready_gm;
  h_o_ready_gm.SetGlobalBuffer(h_o_ready_handle);

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;
  const int64_t paired_core_count = total_work - block_num;
  const int64_t dense_chunk_count = total_tokens / C;
  bool no_initial_state = has_initial_state == 0;
  if constexpr (LoadInitialStateCache) {
    no_initial_state =
        no_initial_state ||
        (batch_size == 1 && initial_state_indices != nullptr &&
         initial_state_indices[0] < 0);
  }
#ifdef MEGA_CHUNK_GDN_OVERFLOW_SEGMENT_PIPELINE
  const int64_t overflow_head_count = total_work - block_num;
  const int32_t overflow_stage_count =
      overflow_head_count > 0
          ? static_cast<int32_t>(block_num / overflow_head_count)
          : 0;
  const bool use_overflow_segment_pipeline =
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
      !FuseResidentOutput &&
#endif
      batch_size == 1 && has_initial_state == 0 && precompute_qs != 0 &&
      C == D && cu_seqlens != nullptr && total_tokens > 0 &&
      total_tokens % C == 0 && dense_chunk_count > 0 &&
      overflow_head_count > 0 && overflow_head_count < block_num &&
      block_num % overflow_head_count == 0 && overflow_stage_count >= 2 &&
      dense_chunk_count >= overflow_stage_count && total_work == H &&
      static_cast<int64_t>(cu_seqlens[0]) == 0 &&
      static_cast<int64_t>(cu_seqlens[1]) == total_tokens;
#else
  constexpr int64_t overflow_head_count = 0;
  constexpr int32_t overflow_stage_count = 0;
  constexpr bool use_overflow_segment_pipeline = false;
#endif
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
  const int64_t fused_wy_group_count =
      wy_group_size == 0
          ? 0
          : dense_chunk_count * H /
                static_cast<int64_t>(wy_group_size);
  const int64_t fused_wy_producer_waves =
      fused_wy_group_count == 0
          ? 0
          : (fused_wy_group_count + block_num - 1) / block_num;
  const int64_t fused_wy_required_slots =
      fused_wy_producer_waves *
      static_cast<int64_t>(wy_group_size);
  const bool fused_wy_group_size_supported =
      GROUP >= 1 && GROUP <= 4 &&
      wy_group_size == static_cast<uint32_t>(GROUP);
  const bool use_fused_wy_h =
      fuse_wy_h && no_initial_state && D == C && batch_size == 1 &&
      cu_seqlens != nullptr && total_tokens > 0 && total_tokens % C == 0 &&
      fused_wy_group_size_supported && total_work == H &&
      static_cast<int64_t>(cu_seqlens[0]) == 0 &&
      static_cast<int64_t>(cu_seqlens[1]) == total_tokens &&
      wy_v_handle != nullptr &&
      wy_a1_handle != nullptr && wy_a2_handle != nullptr &&
      fused_wy_required_slots > 0 &&
      static_cast<int64_t>(wy_workspace_slots) >=
          fused_wy_required_slots;
  const bool use_fused_wy_h_public_u =
      use_fused_wy_h && fuse_wy_h_public_u && GROUP == 1;
  // H/O overlap consumes chunk-packed V. The stage-barrier fallback consumes
  // ordinary BSND V, so fused WY->H must preserve the selected public layout.
  const bool fused_output_packed = precompute_qs != 0;
  const int32_t fused_output_v_stride =
      fused_output_packed ? D : H * D;
#else
  constexpr bool use_fused_wy_h = false;
  constexpr bool use_fused_wy_h_public_u = false;
  constexpr bool fused_output_packed = false;
  constexpr int32_t fused_output_v_stride = D;
#endif
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H_INTERLEAVED_OVERFLOW
  const bool use_fused_wy_h_interleaved_overflow =
      use_fused_wy_h && block_num == 20 && total_work == 24 &&
      paired_core_count == 4;
#else
  constexpr bool use_fused_wy_h_interleaved_overflow = false;
#endif
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H_BALANCED_OVERFLOW
  const bool use_fused_wy_h_balanced_overflow =
      use_fused_wy_h && block_num == 20 && total_work == 24 &&
      paired_core_count == 4;
#else
  constexpr bool use_fused_wy_h_balanced_overflow = false;
#endif
#ifdef MEGA_CHUNK_GDN_SPLIT_HEAD_H
  const bool use_split_head_pipeline =
      batch_size == 1 && has_initial_state == 0 && precompute_qs != 0 &&
      C == D && cu_seqlens != nullptr && total_tokens > 0 &&
      total_tokens % C == 0 && dense_chunk_count > 0 &&
      (dense_chunk_count & 1) == 0 &&
      static_cast<int64_t>(cu_seqlens[0]) == 0 &&
      static_cast<int64_t>(cu_seqlens[1]) == total_tokens &&
      total_work > block_num && total_work < 2 * block_num &&
      2 * paired_core_count <= block_num;
#else
  constexpr bool use_split_head_pipeline = false;
#endif
#ifdef MEGA_CHUNK_GDN_SEGMENTED_H
  constexpr int32_t SEGMENT_CHUNKS = 4;
  constexpr int32_t SEGMENTED_H_RESERVED_O_WORKERS = 4;
  const int64_t segmented_h_worker_count =
      block_num > SEGMENTED_H_RESERVED_O_WORKERS
          ? block_num - SEGMENTED_H_RESERVED_O_WORKERS
          : block_num;
  const bool use_segmented_head_pipeline =
      batch_size == 1 && has_initial_state == 0 && precompute_qs != 0 &&
      C == D && cu_seqlens != nullptr && total_tokens > 0 &&
      total_tokens % C == 0 && dense_chunk_count >= SEGMENT_CHUNKS &&
      dense_chunk_count >= 2 * SEGMENT_CHUNKS &&
      dense_chunk_count % SEGMENT_CHUNKS == 0 &&
      static_cast<int64_t>(cu_seqlens[0]) == 0 &&
      static_cast<int64_t>(cu_seqlens[1]) == total_tokens &&
      total_work * (dense_chunk_count / SEGMENT_CHUNKS) > block_num;
#else
  constexpr int32_t SEGMENT_CHUNKS = 4;
  constexpr int64_t segmented_h_worker_count = 0;
  constexpr bool use_segmented_head_pipeline = false;
#endif

#if defined(__DAV_C220_CUBE__)
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
  if (use_fused_wy_h) {
    if (use_fused_wy_h_interleaved_overflow) {
      const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
      if (cid < paired_core_count) {
        const int64_t primary_head = cid;
        const int64_t overflow_head = block_num + cid;
        const int64_t primary_head_g = primary_head / GROUP;
        const int64_t overflow_head_g = overflow_head / GROUP;
        const int64_t ws_core_offset =
            static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
        const int64_t ws_field_span =
            static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
        const int64_t ws_slot_span = 4 * ws_field_span;
        const int64_t primary_ws_ws = ws_core_offset;
        const int64_t primary_ws_k = ws_field_span + ws_core_offset;
        const int64_t primary_ws_u = 2 * ws_field_span + ws_core_offset;
        const int64_t primary_ws_kv = 3 * ws_field_span + ws_core_offset;
        const int64_t overflow_ws_ws = ws_slot_span + primary_ws_ws;
        const int64_t overflow_ws_k = ws_slot_span + primary_ws_k;
        const int64_t overflow_ws_u = ws_slot_span + primary_ws_u;
        const int64_t overflow_ws_kv = ws_slot_span + primary_ws_kv;
        const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
        const int64_t primary_v_base =
            primary_head * static_cast<int64_t>(C) * D;
        const int64_t overflow_v_base =
            overflow_head * static_cast<int64_t>(C) * D;

        for (int32_t ci = 0; ci < num_chunks; ++ci) {
          fused_wy_h_cube_chunk<D, C, 0>(
              Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
              wy_ready_handle, S_handle, V_handle, workspace_handle,
              primary_head, primary_head_g, ci, H, Hg, primary_ws_ws,
              primary_ws_u, primary_ws_k, primary_ws_kv, primary_v_base,
              v_chunk_stride, wy_workspace_slots, wy_group_size, s_l1, w_l1,
              q_l1, k_l1, v_l1, ws_l0, kv_l0);
          fused_wy_h_cube_chunk<D, C, 4>(
              Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
              wy_ready_handle, S_handle, V_handle, workspace_handle,
              overflow_head, overflow_head_g, ci, H, Hg, overflow_ws_ws,
              overflow_ws_u, overflow_ws_k, overflow_ws_kv, overflow_v_base,
              v_chunk_stride, wy_workspace_slots, wy_group_size, s_l1, w_l1,
              q_l1, k_l1, v_l1, ws_l0, kv_l0);
        }
        wait_flag_dev(3);
        wait_flag_dev(7);
      } else {
        fused_wy_h_cube_head_range<D, C, 0>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle, cid, 0,
            num_chunks, H, Hg, wy_workspace_slots, wy_group_size, s_l1, w_l1,
            q_l1, k_l1, v_l1, ws_l0, kv_l0);
        wait_flag_dev(3);
      }
    } else if (use_fused_wy_h_balanced_overflow) {
      constexpr int32_t SplitChunk = 8;
      const int64_t overflow_head_count = paired_core_count;
      if (cid < overflow_head_count) {
        const int64_t overflow_head = block_num + cid;
        fused_wy_h_cube_head_range<D, C, 0>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle,
            overflow_head, 0, SplitChunk, H, Hg, wy_workspace_slots,
            wy_group_size, s_l1, w_l1, q_l1, k_l1, v_l1, ws_l0, kv_l0);
        wait_flag_dev(3);
        fused_wy_h_cube_head_range<D, C, 4>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle, cid, 0,
            static_cast<int32_t>(dense_chunk_count), H, Hg,
            wy_workspace_slots, wy_group_size, s_l1, w_l1, q_l1, k_l1,
            v_l1, ws_l0, kv_l0);
        wait_flag_dev(7);
      } else if (cid < 2 * overflow_head_count) {
        fused_wy_h_cube_head_range<D, C, 0>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle, cid, 0,
            static_cast<int32_t>(dense_chunk_count), H, Hg,
            wy_workspace_slots, wy_group_size, s_l1, w_l1, q_l1, k_l1,
            v_l1, ws_l0, kv_l0);
        wait_flag_dev(3);
        const int64_t overflow_head =
            block_num + cid - overflow_head_count;
        fused_wy_h_cube_head_range<D, C, 4>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle,
            overflow_head, SplitChunk,
            static_cast<int32_t>(dense_chunk_count), H, Hg,
            wy_workspace_slots, wy_group_size, s_l1, w_l1, q_l1, k_l1,
            v_l1, ws_l0, kv_l0);
        wait_flag_dev(7);
      } else {
        fused_wy_h_cube_head_range<D, C, 0>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle, cid, 0,
            static_cast<int32_t>(dense_chunk_count), H, Hg,
            wy_workspace_slots, wy_group_size, s_l1, w_l1, q_l1, k_l1,
            v_l1, ws_l0, kv_l0);
        wait_flag_dev(3);
      }
    } else {
    const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_ws_base = ws_core_offset;
    const int64_t ws_k_base = ws_field_span + ws_core_offset;
    const int64_t ws_u_base = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
    bool resident_output_in_flight = false;
#endif

    for (int64_t wi = 0;
         wi < (total_work + block_num - 1) / block_num; ++wi) {
      const int64_t head = wi * block_num + cid;
      if (head >= total_work) break;
      const int64_t head_g = head / GROUP;
      const int64_t v_head_base =
          fused_output_packed
              ? head * static_cast<int64_t>(C) * D
              : head * static_cast<int64_t>(D);
      for (int32_t ci = 0; ci < num_chunks; ++ci) {
        if (wi == 0) {
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
          fused_wy_h_cube_chunk<D, C, 0, FuseResidentOutput>(
#else
          fused_wy_h_cube_chunk<D, C, 0>(
#endif
              Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
              wy_ready_handle, S_handle, V_handle, workspace_handle, head,
              head_g, ci, H, Hg, ws_ws_base, ws_u_base, ws_k_base,
              ws_kv_base, v_head_base, v_chunk_stride, wy_workspace_slots,
              wy_group_size, s_l1, w_l1, q_l1, k_l1, v_l1, ws_l0, kv_l0,
              fused_output_v_stride, fused_output_packed,
              use_fused_wy_h_public_u
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
              , resident_raw_qk_mailbox, resident_combined_mailbox,
              resident_gated_qk_mailbox, static_cast<int64_t>(cid),
              resident_output_in_flight
#endif
              );
        } else {
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
          fused_wy_h_cube_chunk<D, C, 4, FuseResidentOutput>(
#else
          fused_wy_h_cube_chunk<D, C, 4>(
#endif
              Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
              wy_ready_handle, S_handle, V_handle, workspace_handle, head,
              head_g, ci, H, Hg, ws_ws_base, ws_u_base, ws_k_base,
              ws_kv_base, v_head_base, v_chunk_stride, wy_workspace_slots,
              wy_group_size, s_l1, w_l1, q_l1, k_l1, v_l1, ws_l0, kv_l0,
              fused_output_v_stride, fused_output_packed,
              use_fused_wy_h_public_u
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
              , resident_raw_qk_mailbox, resident_combined_mailbox,
              resident_gated_qk_mailbox, static_cast<int64_t>(cid),
              resident_output_in_flight
#endif
              );
        }
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
        if constexpr (FuseResidentOutput) {
          resident_output_in_flight = true;
        }
#endif
      }
      wait_flag_dev(wi == 0 ? 3 : 7);
    }
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
    if constexpr (FuseResidentOutput) {
      if (resident_output_in_flight) {
        wait_flag_dev(kResidentHoCombinedFreeFlag);
      }
    }
#endif
    }
  } else
#endif
#ifdef MEGA_CHUNK_GDN_OVERFLOW_SEGMENT_PIPELINE
  if (use_overflow_segment_pipeline) {
    const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
    const int32_t stage =
        static_cast<int32_t>(cid / overflow_head_count);
    const int32_t chain =
        static_cast<int32_t>(cid % overflow_head_count);
    const int32_t base_chunks = num_chunks / overflow_stage_count;
    const int32_t extra_chunks = num_chunks % overflow_stage_count;
    const int32_t segment_begin =
        stage * base_chunks +
        (stage < extra_chunks ? stage : extra_chunks);
    const int32_t segment_end =
        segment_begin + base_chunks + (stage < extra_chunks ? 1 : 0);
    const int64_t primary_head = cid;
    const int64_t overflow_head = block_num + chain;
    const int64_t primary_head_g = primary_head / GROUP;
    const int64_t overflow_head_g = overflow_head / GROUP;

    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_slot_span = 4 * ws_field_span;
    const int64_t primary_ws_ws = ws_core_offset;
    const int64_t primary_ws_k = ws_field_span + ws_core_offset;
    const int64_t primary_ws_u = 2 * ws_field_span + ws_core_offset;
    const int64_t primary_ws_kv = 3 * ws_field_span + ws_core_offset;
    const int64_t overflow_ws_ws = ws_slot_span + primary_ws_ws;
    const int64_t overflow_ws_k = ws_slot_span + primary_ws_k;
    const int64_t overflow_ws_u = ws_slot_span + primary_ws_u;
    const int64_t overflow_ws_kv = ws_slot_span + primary_ws_kv;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
    const int64_t primary_v_base =
        primary_head * static_cast<int64_t>(C) * D;
    const int64_t overflow_v_base =
        overflow_head * static_cast<int64_t>(C) * D;

    for (int32_t ci = 0; ci < segment_begin; ++ci) {
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
      if (use_fused_wy_h) {
        fused_wy_h_cube_chunk<D, C, 0>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle, primary_head,
            primary_head_g, ci, H, Hg, primary_ws_ws, primary_ws_u,
            primary_ws_k, primary_ws_kv, primary_v_base, v_chunk_stride,
            wy_workspace_slots, wy_group_size, s_l1, w_l1, q_l1, k_l1,
            v_l1, ws_l0, kv_l0);
      } else
#endif
      paired_cube_chunk<D, C, 0>(
          Q_handle, W_handle, S_handle, V_handle, workspace_handle,
          primary_head, primary_head_g, ci, H, Hg, primary_ws_ws,
          primary_ws_k, primary_ws_kv, primary_v_base, v_chunk_stride,
          s_l1, w_l1, q_l1, k_l1, v_l1, ws_l0, kv_l0);
    }
    for (int32_t ci = segment_begin; ci < segment_end; ++ci) {
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
      if (use_fused_wy_h) {
        fused_wy_h_cube_chunk<D, C, 4>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle, overflow_head,
            overflow_head_g, ci, H, Hg, overflow_ws_ws, overflow_ws_u,
            overflow_ws_k, overflow_ws_kv, overflow_v_base, v_chunk_stride,
            wy_workspace_slots, wy_group_size, s_l1, w_l1, q_l1, k_l1,
            v_l1, ws_l0, kv_l0);
      } else
#endif
      paired_cube_chunk<D, C, 4>(
          Q_handle, W_handle, S_handle, V_handle, workspace_handle,
          overflow_head, overflow_head_g, ci, H, Hg, overflow_ws_ws,
          overflow_ws_k, overflow_ws_kv, overflow_v_base, v_chunk_stride,
          s_l1, w_l1, q_l1, k_l1, v_l1, ws_l0, kv_l0);
    }
    for (int32_t ci = segment_begin; ci < num_chunks; ++ci) {
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
      if (use_fused_wy_h) {
        fused_wy_h_cube_chunk<D, C, 0>(
            Q_handle, K_handle, wy_v_handle, wy_a1_handle, wy_a2_handle,
            wy_ready_handle, S_handle, V_handle, workspace_handle, primary_head,
            primary_head_g, ci, H, Hg, primary_ws_ws, primary_ws_u,
            primary_ws_k, primary_ws_kv, primary_v_base, v_chunk_stride,
            wy_workspace_slots, wy_group_size, s_l1, w_l1, q_l1, k_l1,
            v_l1, ws_l0, kv_l0);
      } else
#endif
      paired_cube_chunk<D, C, 0>(
          Q_handle, W_handle, S_handle, V_handle, workspace_handle,
          primary_head, primary_head_g, ci, H, Hg, primary_ws_ws,
          primary_ws_k, primary_ws_kv, primary_v_base, v_chunk_stride,
          s_l1, w_l1, q_l1, k_l1, v_l1, ws_l0, kv_l0);
    }
    wait_flag_dev(3);
    wait_flag_dev(7);
  } else
#endif
  if (use_segmented_head_pipeline) {
    const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
    const int32_t segments_per_head = num_chunks / SEGMENT_CHUNKS;
    const int64_t total_segments =
        static_cast<int64_t>(segments_per_head) * H;
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_ws_base = ws_core_offset;
    const int64_t ws_k_base = ws_field_span + ws_core_offset;
    const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;

    for (int64_t task = cid;
         cid < segmented_h_worker_count && task < total_segments;
         task += segmented_h_worker_count) {
      const int32_t segment_idx = static_cast<int32_t>(task / H);
      const int64_t head = task % H;
      const int64_t head_g = head / GROUP;
      const int32_t chunk_begin = segment_idx * SEGMENT_CHUNKS;
      const int32_t chunk_end = chunk_begin + SEGMENT_CHUNKS;
      const int64_t v_head_base = head * static_cast<int64_t>(C) * D;

      for (int32_t ci = chunk_begin; ci < chunk_end; ++ci) {
        const int64_t chunk_start = static_cast<int64_t>(ci) * C;
        paired_cube_ws<D, C, 0, true, true>(
            W_handle, S_handle, workspace_handle, head, chunk_start, 0, ci,
            C, H, ws_ws_base, s_l1, w_l1, ws_l0);
        paired_cube_qs<D, C>(Q_handle, S_handle, head, head_g, chunk_start,
                             0, ci, C, H, Hg, s_l1, q_l1, ws_l0);
        paired_cube_kv<D, C, 0, true, true>(
            V_handle, workspace_handle, C, D,
            v_head_base + static_cast<int64_t>(ci) * v_chunk_stride,
            ws_k_base, ws_kv_base, k_l1, v_l1, kv_l0);
      }
      wait_flag_dev(3);
    }
  } else if (use_split_head_pipeline && cid < paired_core_count) {
    const int64_t head0 = cid;
    const int64_t head1 = block_num + cid;
    const int64_t head_g0 = head0 / GROUP;
    const int64_t head_g1 = head1 / GROUP;
    const int64_t bos = static_cast<int64_t>(cu_seqlens[0]);
    const int64_t slen = static_cast<int64_t>(cu_seqlens[1]) - bos;
    const int32_t num_chunks = static_cast<int32_t>(slen / C);
    const int32_t split_chunk = num_chunks / 2;
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_slot_span = 4 * ws_field_span;
    const int64_t ws_ws_base0 = ws_core_offset;
    const int64_t ws_k_base0 = ws_field_span + ws_core_offset;
    const int64_t ws_s_base0 = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base0 = 3 * ws_field_span + ws_core_offset;
    const int64_t ws_ws_base1 = ws_slot_span + ws_ws_base0;
    const int64_t ws_k_base1 = ws_slot_span + ws_k_base0;
    const int64_t ws_s_base1 = ws_slot_span + ws_s_base0;
    const int64_t ws_kv_base1 = ws_slot_span + ws_kv_base0;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
    const int64_t v_head_base0 = head0 * static_cast<int64_t>(C) * D;
    const int64_t v_head_base1 = head1 * static_cast<int64_t>(C) * D;

#ifdef MEGA_STOP_AFTER_H
    if constexpr (GDN_PAIR_DEBUG_STAGE == 1) return;
    if constexpr (GDN_PAIR_DEBUG_STAGE == 10) {
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 12) {
      const int64_t w_offset = (bos * H + head0) * D;
      GmShape2D w_shape(C, D);
      GmStride2D w_stride(H * D);
      GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
      DynMatL1<ComputeT, C, D> w_l1_load(C, D);
      TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
      TLOAD(w_l1_load, w_global);
      ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (0 << 8));
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 13) {
      const int64_t w_offset = (bos * H + head0) * D;
      GmShape2D w_shape(C, D);
      GmStride2D w_stride(H * D);
      GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
      DynMatL1<ComputeT, C, D> w_l1_load(C, D);
      TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
      TLOAD(w_l1_load, w_global);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

      GmShape2D s_shape(D, D);
      GmStride2D s_stride(D);
      GmTensor2D<ComputeT> s_global(workspace_handle + ws_s_base0, s_shape,
                                   s_stride);
      DynMatL1<ComputeT, D, D> s_l1_load(D, D);
      TASSIGN(s_l1_load, 0);
      TLOAD(s_l1_load, s_global);
      ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (0 << 8));
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 15) {
      const int64_t w_offset = (bos * H + head0) * D;
      GmShape2D w_shape(C, D);
      GmStride2D w_stride(H * D);
      GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
      DynMatL1<ComputeT, C, D> w_l1_load(C, D);
      TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
      TLOAD(w_l1_load, w_global);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

      GmShape2D s_shape(D, D);
      GmStride2D s_stride(D);
      GmTensor2D<ComputeT> s_global(workspace_handle + ws_s_base0, s_shape,
                                   s_stride);
      DynMatL1<ComputeT, D, D> s_l1_load(D, D);
      TASSIGN(s_l1_load, 0);
      TLOAD(s_l1_load, s_global);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

      TileMatL0A<ComputeT, C, D, C, D> w_l0a;
      TileMatL0B<ComputeT, D, D, D, D> s_l0b;
      TASSIGN(w_l0a, 0);
      TASSIGN(s_l0b, 0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      ffts_cross_core_sync(PIPE_MTE2, 1 | (2 << 4) | (0 << 8));
      return;
    }
#endif
    for (int32_t ci = 0; ci < split_chunk; ++ci) {
      const int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      paired_cube_ws<D, C, 0, true, true>(
          W_handle, S_handle, workspace_handle, head0, chunk_start, 0, ci,
          C, H, ws_ws_base0, s_l1, w_l1, ws_l0);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 2 ||
                    GDN_PAIR_DEBUG_STAGE == 8 ||
                    GDN_PAIR_DEBUG_STAGE == 9 ||
                    GDN_PAIR_DEBUG_STAGE == 11 ||
                    GDN_PAIR_DEBUG_STAGE == 14 ||
                    GDN_PAIR_DEBUG_STAGE == 15)
        return;
#endif
      paired_cube_qs<D, C>(Q_handle, S_handle, head0, head_g0, chunk_start, 0,
                           ci, C, H, Hg, s_l1, q_l1, ws_l0);

      paired_cube_ws<D, C, 4, true, true>(
          W_handle, S_handle, workspace_handle, head1, chunk_start, 0, ci,
          C, H, ws_ws_base1, s_l1, w_l1, ws_l0);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 3) return;
#endif
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 4) return;
#endif
      paired_cube_qs<D, C>(Q_handle, S_handle, head1, head_g1, chunk_start, 0,
                           ci, C, H, Hg, s_l1, q_l1, ws_l0);

      paired_cube_kv<D, C, 0, true, true>(
          V_handle, workspace_handle, C, D,
          v_head_base0 + static_cast<int64_t>(ci) * v_chunk_stride,
          ws_k_base0, ws_kv_base0, k_l1, v_l1, kv_l0);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 5) return;
#endif
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 6) return;
#endif

      paired_cube_kv<D, C, 4, true, true>(
          V_handle, workspace_handle, C, D,
          v_head_base1 + static_cast<int64_t>(ci) * v_chunk_stride,
          ws_k_base1, ws_kv_base1, k_l1, v_l1, kv_l0);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 7) return;
#endif
    }
#ifndef MEGA_STOP_AFTER_H
    wait_flag_dev(3);
    wait_flag_dev(7);
#endif
  } else {
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
  bool resident_output_in_flight = false;
#endif
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t head_g = head / GROUP;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
    const bool emit_precomputed_qs =
        !FuseResidentOutput && precompute_qs != 0 && C == D && H >= 8 &&
        cu_seqlens != nullptr;
    const bool use_resident_packed_v =
        FuseResidentOutput && precompute_qs != 0 && C == D && H >= 8 &&
        cu_seqlens != nullptr;
    const bool use_packed_v =
        (emit_precomputed_qs || use_resident_packed_v) &&
        batch_size == 1 && (slen % C) == 0;
#else
    const bool emit_precomputed_qs =
        precompute_qs != 0 && C == D && H >= 8 && cu_seqlens != nullptr;
    const bool use_packed_v =
        emit_precomputed_qs && batch_size == 1 && (slen % C) == 0;
#endif
    const int32_t v_stride =
        use_packed_v ? D : BSND_QKV_STRIDE;
    const int64_t v_head_base =
        use_packed_v
            ? (chunk_offset * H + head) *
                  static_cast<int64_t>(C) * D
            : (bos * H + head) * static_cast<int64_t>(D);
    const int64_t v_chunk_stride =
        static_cast<int64_t>(H) * C * D;
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_ws_base = ws_core_offset;
    const int64_t ws_k_base = ws_field_span + ws_core_offset;
    const int64_t ws_s_base = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
    // One per-core scratch region stores:
    //   WS_WS : ws = W_i @ S_i
    //   WS_K  : k_tilde
    //   WS_S  : running state S_i
    //   WS_KV : k_tilde^T @ v_i_new

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
#if defined(PTO_NPU_ARCH_A5)
      gdn_sync::Wait<PIPE_MTE2>(3);
#else
      wait_flag_dev(3);
#endif
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

#if defined(GDN_A5_KERNEL)
      // Consumer-side invalidate for the Vec-published state: A5 Cube MTE2
      // loads can return clean stale lines left by a previous launch.
      for (int32_t r = 0; r < D * D; r += 16) {
        dcci(static_cast<__gm__ void *>(workspace_handle + ws_s_base + r),
             SINGLE_CACHE_LINE);
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      {
        const int64_t s_state_offset =
            ((chunk_offset + ci) * H + head) * DD;
        GmShape2D s_shape(D, D);
        GmStride2D s_stride(D);
        GmTensor2D<ComputeT> s_global(S_handle + s_state_offset, s_shape,
                                     s_stride);
        DynMatL1<ComputeT, D, D> s_l1_load(D, D);
        TASSIGN(s_l1_load, 0);
        // Load S_i before this slot is overwritten with precomputed Q_i @ S_i.
        TLOAD(s_l1_load, s_global);
      }

      int64_t w_offset = ((chunk_start) * H + head) * D;
#if defined(GDN_A5_KERNEL)
      // W is published by the WY stage Vector cores; same stale-line hazard.
      // Rows are BSND-strided, so invalidate row by row.
      for (int32_t row = 0; row < valid; ++row) {
        for (int32_t r = 0; r < D; r += 16) {
          dcci(static_cast<__gm__ void *>(
                   W_handle + w_offset + row * BSND_QKV_STRIDE + r),
               SINGLE_CACHE_LINE);
        }
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      {
        GmShape2D w_shape(static_cast<int32_t>(valid), D);
        GmStride2D w_stride(BSND_QKV_STRIDE);
        GmTensor2D<ComputeT> w_global(W_handle + w_offset, w_shape, w_stride);
        DynMatL1<ComputeT, C, D> w_l1_load(static_cast<int32_t>(valid), D);
        TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(ComputeT)));
        TLOAD(w_l1_load, w_global);
        if (valid != C) {
          TFILLPAD(w_l1_load, w_l1_load);
        }
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      // Apply the carried recurrent state to every token in this chunk.
      gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
          w_l1, s_l1, ws_l0, (bool)1);

      if (emit_precomputed_qs) {
        const int64_t q_offset = (chunk_start * Hg + head_g) * D;
        GmShape2D q_shape(static_cast<int32_t>(valid), D);
        GmStride2D q_stride(BSND_K_STRIDE);
        GmTensor2D<ComputeT> q_global(Q_handle + q_offset, q_shape, q_stride);
        DynMatL1<ComputeT, C, D> q_l1_load(static_cast<int32_t>(valid), D);
        TASSIGN(q_l1_load,
                D * D * static_cast<int32_t>(sizeof(ComputeT)));
        TLOAD(q_l1_load, q_global);
        if (valid != C) {
          TFILLPAD(q_l1_load, q_l1_load);
        }
      }

      {
        GmShape2D ws_shape(C, D);
        GmStride2D ws_stride(D);
        GmTensor2D<ComputeT> ws_global(workspace_handle + ws_ws_base,
                                   ws_shape, ws_stride);
        DynAccTile<float, C, D> ws_store(C, D);
        TASSIGN(ws_store, 0);
        // Save ws_i so the Vec phase can do `v_new = U_i - ws_i`.
        TSTORE(ws_global, ws_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      if (emit_precomputed_qs) {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        gemm_v0<ComputeT, float, C, D, D, C, D, D, D, false, false>(
            q_l1, s_l1, ws_l0, (bool)1);

        const int64_t qs_offset = ((chunk_offset + ci) * H + head) * DD;
        GmShape2D qs_shape(C, D);
        GmStride2D qs_stride(D);
        GmTensor2D<ComputeT> qs_global(S_handle + qs_offset, qs_shape,
                                     qs_stride);
        DynAccTile<float, C, D> qs_store(C, D);
        TASSIGN(qs_store, 0);
        TSTORE(qs_global, qs_store);
      }

#if defined(PTO_NPU_ARCH_A5)
      gdn_sync::Wait<PIPE_MTE2>(1);
#else
      wait_flag_dev(1);
#endif

#if defined(GDN_A5_KERNEL)
      // Consumer-side invalidate for the Vec-published k_tilde tile before the
      // KV GEMM; a stale line here corrupts the state update (final_state).
      for (int32_t r = 0; r < D * C; r += 16) {
        dcci(static_cast<__gm__ void *>(workspace_handle + ws_k_base + r),
             SINGLE_CACHE_LINE);
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      const int64_t v_offset =
          v_head_base + static_cast<int64_t>(ci) * v_chunk_stride;
      {
        GmShape2D k_shape(D, C);
        GmStride2D k_stride(C);
        GmTensor2D<ComputeT> k_global(workspace_handle + ws_k_base, k_shape,
                                  k_stride);
        DynMatL1<ComputeT, D, C> k_l1_load(D, C);
        TASSIGN(k_l1_load, (DD + C * D) * static_cast<int32_t>(sizeof(ComputeT)));
        TLOAD(k_l1_load, k_global);
      }

#if defined(GDN_A5_KERNEL)
      // V_new is published by this stage's Vector cores earlier in the chunk
      // loop; the Cube MTE2 load can still hold a stale line from a previous
      // work item or launch, which corrupts the KV product (final_state).
      for (int32_t row = 0; row < valid; ++row) {
        for (int32_t r = 0; r < D; r += 16) {
          dcci(static_cast<__gm__ void *>(
                   V_handle + v_offset + row * v_stride + r),
               SINGLE_CACHE_LINE);
        }
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      {
        GmShape2D v_shape(static_cast<int32_t>(valid), D);
        GmStride2D v_gm_stride(v_stride);
        GmTensor2D<ComputeT> v_global(
            V_handle + v_offset, v_shape, v_gm_stride);
        DynMatL1<ComputeT, C, D> v_l1_load(static_cast<int32_t>(valid), D);
        TASSIGN(v_l1_load,
                (DD + C * D + D * C) * static_cast<int32_t>(sizeof(ComputeT)));
        TLOAD(v_l1_load, v_global);
        if (valid != C) {
          TFILLPAD(v_l1_load, v_l1_load);
        }
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      // This chunk contributes the additive update K_i^T V_i to the state recurrence.
      gemm_v0<ComputeT, float, D, D, C, D, D, C, C, true, false>(
          k_l1, v_l1, kv_l0, (bool)1);

      {
        GmShape2D kv_shape(D, D);
        GmStride2D kv_stride(D);
        GmTensor2D<ComputeT> kv_global(workspace_handle + ws_kv_base,
                                   kv_shape, kv_stride);
        DynAccTile<float, D, D> kv_store(D, D);
        TASSIGN(kv_store, SecondL0CAddr);
        // Save kv = k_tilde^T @ v_i_new so Vec can finish the state update.
        TSTORE(kv_global, kv_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
      if constexpr (FuseResidentOutput) {
        ResidentHoCubeOutput<D, C>(
            resident_combined_mailbox, resident_gated_qk_mailbox,
            static_cast<int64_t>(cid), resident_output_in_flight, s_l1,
            w_l1, v_l1, ws_l0);
        resident_output_in_flight = true;
      }
#endif
    }
  }
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
  if constexpr (FuseResidentOutput) {
    if (resident_output_in_flight) {
      wait_flag_dev(kResidentHoCombinedFreeFlag);
    }
  }
#endif
  }

  if (use_split_head_pipeline) {
    if (cid < 2 * paired_core_count) {
      const bool helper_core = cid >= paired_core_count;
      const int64_t head =
          helper_core ? block_num + cid - paired_core_count : cid;
      const int64_t head_g = head / GROUP;
      const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
      const int32_t split_chunk = num_chunks / 2;
      const int64_t ws_core_offset =
          static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
      const int64_t ws_field_span =
          static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
      const int64_t ws_ws_base = ws_core_offset;
      const int64_t ws_k_base = ws_field_span + ws_core_offset;
      const int64_t ws_s_base = 2 * ws_field_span + ws_core_offset;
      const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
      const int64_t v_head_base =
          fused_output_packed
              ? head * static_cast<int64_t>(C) * D
              : head * static_cast<int64_t>(D);
      const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;

      for (int32_t ci = split_chunk; ci < num_chunks; ++ci) {
        const int64_t chunk_start = static_cast<int64_t>(ci) * C;
        paired_cube_ws<D, C, 0, true, true>(
            W_handle, S_handle, workspace_handle, head, chunk_start, 0, ci,
            C, H, ws_ws_base, s_l1, w_l1, ws_l0);
        paired_cube_qs<D, C>(Q_handle, S_handle, head, head_g, chunk_start,
                             0, ci, C, H, Hg, s_l1, q_l1, ws_l0);
        paired_cube_kv<D, C, 0, true, true>(
            V_handle, workspace_handle, C, D,
            v_head_base + static_cast<int64_t>(ci) * v_chunk_stride,
            ws_k_base, ws_kv_base, k_l1, v_l1, kv_l0);
      }
      wait_flag_dev(3);
    }
  }
#endif
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
  if constexpr (FuseResidentOutput) {
    ResidentHoPrepareVectorConstants<
        D, C, FuseResidentGatedRmsNorm>(
        resident_mask_handle, resident_norm_weight_handle,
        static_cast<int32_t>(vid));
  }
#endif

#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
  if (use_fused_wy_h) {
    if (use_fused_wy_h_interleaved_overflow) {
      const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
      if (cid < paired_core_count) {
        const int64_t primary_head = cid;
        const int64_t overflow_head = block_num + cid;
        const int64_t primary_head_g = primary_head / GROUP;
        const int64_t overflow_head_g = overflow_head / GROUP;
        const int64_t ws_core_offset =
            static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
        const int64_t ws_field_span =
            static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
        const int64_t ws_slot_span = 4 * ws_field_span;
        const int64_t primary_ws_ws = ws_core_offset;
        const int64_t primary_ws_k = ws_field_span + ws_core_offset;
        const int64_t primary_ws_u = 2 * ws_field_span + ws_core_offset;
        const int64_t primary_ws_kv = 3 * ws_field_span + ws_core_offset;
        const int64_t overflow_ws_ws = ws_slot_span + primary_ws_ws;
        const int64_t overflow_ws_k = ws_slot_span + primary_ws_k;
        const int64_t overflow_ws_u = ws_slot_span + primary_ws_u;
        const int64_t overflow_ws_kv = ws_slot_span + primary_ws_kv;
        const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
        const int64_t primary_v_base =
            primary_head * static_cast<int64_t>(C) * D;
        const int64_t overflow_v_base =
            overflow_head * static_cast<int64_t>(C) * D;

        paired_vec_init_zero<D, C, 0, true>(
            S_handle, primary_head, 0, H, vid, s_ub, s_ub_half);
        paired_vec_init_zero<D, C, 4, true>(
            S_handle, overflow_head, 0, H, vid, s_alt_ub, s_ub_half);
        for (int32_t ci = 0; ci < num_chunks; ++ci) {
          paired_vec_chunk<D, C, 0, G_UB, COEFF_UB, U_UB, true>(
              K_handle, U_handle, G_handle, S_handle, V_handle,
              workspace_handle, h_o_ready_handle, primary_head,
              primary_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
              primary_ws_ws, primary_ws_k, primary_ws_kv, primary_v_base,
              v_chunk_stride, s_ub, s_ub_half, k_ub_half, g_ub, u_ub_half,
              k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub,
              primary_ws_u);
          paired_vec_chunk<D, C, 4, G_UB, COEFF_UB, U_UB, true>(
              K_handle, U_handle, G_handle, S_handle, V_handle,
              workspace_handle, h_o_ready_handle, overflow_head,
              overflow_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
              overflow_ws_ws, overflow_ws_k, overflow_ws_kv, overflow_v_base,
              v_chunk_stride, s_alt_ub, s_ub_half, k_ub_half, g_ub,
              u_ub_half, k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub,
              overflow_ws_u);
        }
        paired_vec_store_final<D, C, StoreFinalStateCache>(
            FS_handle, final_state_cache, state_indices, primary_head, H, vid,
            state_index_stride, state_cache_slots, output_final_state, s_ub,
            s_ub_half);
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
        paired_vec_store_final<D, C, StoreFinalStateCache>(
            FS_handle, final_state_cache, state_indices, overflow_head, H,
            vid, state_index_stride, state_cache_slots, output_final_state,
            s_alt_ub, s_ub_half);
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (7 << 8));
      } else {
        fused_wy_h_vec_head_range<
            D, C, 0, G_UB, COEFF_UB, U_UB, StoreFinalStateCache,
            true, false, true, false, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle, FS_handle,
            workspace_handle, h_o_ready_handle, final_state_cache,
            state_indices, cid, 0, num_chunks, num_chunks, H, Hg,
            total_tokens, vid, state_index_stride, state_cache_slots,
            output_final_state, s_ub, s_ub_half, k_ub_half, g_ub,
            u_ub_half, k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
      }
    } else if (use_fused_wy_h_balanced_overflow) {
      constexpr int32_t SplitChunk = 8;
      const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
      const int64_t overflow_head_count = paired_core_count;
      if (cid < overflow_head_count) {
        const int64_t overflow_head = block_num + cid;
        fused_wy_h_vec_head_range<
            D, C, 0, G_UB, COEFF_UB, U_UB, StoreFinalStateCache,
            true, false, false, true, false>(
            K_handle, U_handle, G_handle, S_handle, V_handle, FS_handle,
            workspace_handle, h_o_ready_handle, final_state_cache,
            state_indices, overflow_head, 0, SplitChunk, num_chunks, H, Hg,
            total_tokens, vid, state_index_stride, state_cache_slots,
            output_final_state, s_ub, s_ub_half, k_ub_half, g_ub,
            u_ub_half, k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
        fused_wy_h_vec_head_range<
            D, C, 4, G_UB, COEFF_UB, U_UB, StoreFinalStateCache,
            true, false, true, false, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle, FS_handle,
            workspace_handle, h_o_ready_handle, final_state_cache,
            state_indices, cid, 0, num_chunks, num_chunks, H, Hg,
            total_tokens, vid, state_index_stride, state_cache_slots,
            output_final_state, s_ub, s_ub_half, k_ub_half, g_ub,
            u_ub_half, k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
      } else if (cid < 2 * overflow_head_count) {
        fused_wy_h_vec_head_range<
            D, C, 0, G_UB, COEFF_UB, U_UB, StoreFinalStateCache,
            true, false, true, false, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle, FS_handle,
            workspace_handle, h_o_ready_handle, final_state_cache,
            state_indices, cid, 0, num_chunks, num_chunks, H, Hg,
            total_tokens, vid, state_index_stride, state_cache_slots,
            output_final_state, s_ub, s_ub_half, k_ub_half, g_ub,
            u_ub_half, k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
        const int64_t overflow_head =
            block_num + cid - overflow_head_count;
        fused_wy_h_vec_head_range<
            D, C, 4, G_UB, COEFF_UB, U_UB, StoreFinalStateCache,
            false, true, true, false, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle, FS_handle,
            workspace_handle, h_o_ready_handle, final_state_cache,
            state_indices, overflow_head, SplitChunk, num_chunks,
            num_chunks, H, Hg, total_tokens, vid, state_index_stride,
            state_cache_slots, output_final_state, s_ub, s_ub_half,
            k_ub_half, g_ub, u_ub_half, k_ub, g_last_tail_ub, coeff_ub,
            u_ub, ws_ub, kv_ub);
      } else {
        fused_wy_h_vec_head_range<
            D, C, 0, G_UB, COEFF_UB, U_UB, StoreFinalStateCache,
            true, false, true, false, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle, FS_handle,
            workspace_handle, h_o_ready_handle, final_state_cache,
            state_indices, cid, 0, num_chunks, num_chunks, H, Hg,
            total_tokens, vid, state_index_stride, state_cache_slots,
            output_final_state, s_ub, s_ub_half, k_ub_half, g_ub,
            u_ub_half, k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
      }
    } else {
    const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_ws_base = ws_core_offset;
    const int64_t ws_k_base = ws_field_span + ws_core_offset;
    const int64_t ws_u_base = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;

    for (int64_t wi = 0;
         wi < (total_work + block_num - 1) / block_num; ++wi) {
      const int64_t head = wi * block_num + cid;
      if (head >= total_work) break;
      const int64_t head_g = head / GROUP;
      const int64_t v_head_base =
          fused_output_packed
              ? head * static_cast<int64_t>(C) * D
              : head * static_cast<int64_t>(D);
      if (wi == 0) {
        paired_vec_init_zero<D, C, 0, true>(
            S_handle, head, 0, H, vid, s_ub, s_ub_half);
      } else {
        paired_vec_init_zero<D, C, 4, true>(
            S_handle, head, 0, H, vid, s_ub, s_ub_half);
      }
      for (int32_t ci = 0; ci < num_chunks; ++ci) {
        if (wi == 0) {
          paired_vec_chunk<D, C, 0, G_UB, COEFF_UB, U_UB, true>(
              K_handle, U_handle, G_handle, S_handle, V_handle,
              workspace_handle, h_o_ready_handle, head, head_g, ci,
              num_chunks, H, Hg, total_tokens, vid, ws_ws_base, ws_k_base,
              ws_kv_base, v_head_base, v_chunk_stride, s_ub, s_ub_half,
              k_ub_half, g_ub, u_ub_half, k_ub, g_last_tail_ub, coeff_ub,
              u_ub, ws_ub, kv_ub, ws_u_base, fused_output_v_stride,
              fused_output_packed, use_fused_wy_h_public_u);
        } else {
          paired_vec_chunk<D, C, 4, G_UB, COEFF_UB, U_UB, true>(
              K_handle, U_handle, G_handle, S_handle, V_handle,
              workspace_handle, h_o_ready_handle, head, head_g, ci,
              num_chunks, H, Hg, total_tokens, vid, ws_ws_base, ws_k_base,
              ws_kv_base, v_head_base, v_chunk_stride, s_ub, s_ub_half,
              k_ub_half, g_ub, u_ub_half, k_ub, g_last_tail_ub, coeff_ub,
              u_ub, ws_ub, kv_ub, ws_u_base, fused_output_v_stride,
              fused_output_packed, use_fused_wy_h_public_u);
        }
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
        if constexpr (FuseResidentOutput) {
          const int64_t chunk_start = static_cast<int64_t>(ci) * C;
          ResidentHoGateQkAndScaleQ<D, C>(
              Q_handle, G_handle, resident_raw_qk_mailbox,
              resident_combined_mailbox, resident_gated_qk_mailbox,
              static_cast<int64_t>(cid), chunk_start, total_tokens,
              static_cast<int32_t>(head), static_cast<int32_t>(head_g), Hg,
              static_cast<int32_t>(vid));
          ResidentHoConsumeCombined<D, C, FuseResidentGatedRmsNorm>(
              resident_combined_mailbox, resident_output_handle,
              resident_z_handle, static_cast<int64_t>(cid), chunk_start,
              static_cast<int32_t>(head), H, static_cast<int32_t>(vid),
              ci + 1 == num_chunks);
        }
#endif
      }
      paired_vec_store_final<D, C, StoreFinalStateCache>(
          FS_handle, final_state_cache, state_indices, head, H, vid,
          state_index_stride, state_cache_slots, output_final_state, s_ub,
          s_ub_half);
      ffts_cross_core_sync(
          PIPE_MTE3,
          1 | (2 << 4) | (static_cast<int32_t>(wi == 0 ? 3 : 7) << 8));
    }
    }
  } else
#endif
#ifdef MEGA_CHUNK_GDN_OVERFLOW_SEGMENT_PIPELINE
  if (use_overflow_segment_pipeline) {
    const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
    const int32_t stage =
        static_cast<int32_t>(cid / overflow_head_count);
    const int32_t chain =
        static_cast<int32_t>(cid % overflow_head_count);
    const int32_t base_chunks = num_chunks / overflow_stage_count;
    const int32_t extra_chunks = num_chunks % overflow_stage_count;
    const int32_t segment_begin =
        stage * base_chunks +
        (stage < extra_chunks ? stage : extra_chunks);
    const int32_t segment_end =
        segment_begin + base_chunks + (stage < extra_chunks ? 1 : 0);
    const int64_t primary_head = cid;
    const int64_t overflow_head = block_num + chain;
    const int64_t primary_head_g = primary_head / GROUP;
    const int64_t overflow_head_g = overflow_head / GROUP;

    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_slot_span = 4 * ws_field_span;
    const int64_t primary_ws_ws = ws_core_offset;
    const int64_t primary_ws_k = ws_field_span + ws_core_offset;
    const int64_t primary_ws_u = 2 * ws_field_span + ws_core_offset;
    const int64_t primary_ws_kv = 3 * ws_field_span + ws_core_offset;
    const int64_t overflow_ws_ws = ws_slot_span + primary_ws_ws;
    const int64_t overflow_ws_k = ws_slot_span + primary_ws_k;
    const int64_t overflow_ws_u = ws_slot_span + primary_ws_u;
    const int64_t overflow_ws_kv = ws_slot_span + primary_ws_kv;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
    const int64_t primary_v_base =
        primary_head * static_cast<int64_t>(C) * D;
    const int64_t overflow_v_base =
        overflow_head * static_cast<int64_t>(C) * D;

    paired_vec_init_zero<D, C, 0, true>(
        S_handle, primary_head, 0, H, vid, s_ub, s_ub_half);
    for (int32_t ci = 0; ci < segment_begin; ++ci) {
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
      if (use_fused_wy_h) {
        paired_vec_chunk<D, C, 0, G_UB, COEFF_UB, U_UB, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle,
            workspace_handle, h_o_ready_handle, primary_head,
            primary_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
            primary_ws_ws, primary_ws_k, primary_ws_kv, primary_v_base,
            v_chunk_stride, s_ub, s_ub_half, k_ub_half, g_ub, u_ub_half,
            k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub,
            primary_ws_u);
      } else
#endif
      paired_vec_chunk<D, C, 0, G_UB, COEFF_UB, U_UB>(
          K_handle, U_handle, G_handle, S_handle, V_handle,
          workspace_handle, h_o_ready_handle, primary_head,
          primary_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
          primary_ws_ws, primary_ws_k, primary_ws_kv, primary_v_base,
          v_chunk_stride, s_ub, s_ub_half, k_ub_half, g_ub, u_ub_half,
          k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
    }

    if (stage == 0) {
      paired_vec_init_zero<D, C, 4, true>(
          S_handle, overflow_head, 0, H, vid, s_alt_ub, s_ub_half);
    } else {
      wait_segment_state<C>(h_o_ready_handle, overflow_head,
                            segment_begin, vid);
      paired_vec_load_state<D, C, 4>(
          S_handle, overflow_head, 0, segment_begin, H, vid, s_alt_ub,
          s_ub_half);
    }
    for (int32_t ci = segment_begin; ci < segment_end; ++ci) {
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
      if (use_fused_wy_h) {
        paired_vec_chunk<D, C, 4, G_UB, COEFF_UB, U_UB, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle,
            workspace_handle, h_o_ready_handle, overflow_head,
            overflow_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
            overflow_ws_ws, overflow_ws_k, overflow_ws_kv, overflow_v_base,
            v_chunk_stride, s_alt_ub, s_ub_half, k_ub_half, g_ub, u_ub_half,
            k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub,
            overflow_ws_u);
      } else
#endif
      paired_vec_chunk<D, C, 4, G_UB, COEFF_UB, U_UB>(
          K_handle, U_handle, G_handle, S_handle, V_handle,
          workspace_handle, h_o_ready_handle, overflow_head,
          overflow_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
          overflow_ws_ws, overflow_ws_k, overflow_ws_kv, overflow_v_base,
          v_chunk_stride, s_alt_ub, s_ub_half, k_ub_half, g_ub, u_ub_half,
          k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
    }
    if (stage + 1 < overflow_stage_count) {
      publish_segment_state<C>(h_o_ready_handle, overflow_head,
                               segment_end, vid);
    } else {
      paired_vec_store_final<D, C, StoreFinalStateCache>(
          FS_handle, final_state_cache, state_indices, overflow_head, H, vid,
          state_index_stride, state_cache_slots, output_final_state,
          s_alt_ub, s_ub_half);
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (7 << 8));
    }

    for (int32_t ci = segment_begin; ci < num_chunks; ++ci) {
#ifdef MEGA_CHUNK_GDN_FUSED_WY_H
      if (use_fused_wy_h) {
        paired_vec_chunk<D, C, 0, G_UB, COEFF_UB, U_UB, true>(
            K_handle, U_handle, G_handle, S_handle, V_handle,
            workspace_handle, h_o_ready_handle, primary_head,
            primary_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
            primary_ws_ws, primary_ws_k, primary_ws_kv, primary_v_base,
            v_chunk_stride, s_ub, s_ub_half, k_ub_half, g_ub, u_ub_half,
            k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub,
            primary_ws_u);
      } else
#endif
      paired_vec_chunk<D, C, 0, G_UB, COEFF_UB, U_UB>(
          K_handle, U_handle, G_handle, S_handle, V_handle,
          workspace_handle, h_o_ready_handle, primary_head,
          primary_head_g, ci, num_chunks, H, Hg, total_tokens, vid,
          primary_ws_ws, primary_ws_k, primary_ws_kv, primary_v_base,
          v_chunk_stride, s_ub, s_ub_half, k_ub_half, g_ub, u_ub_half,
          k_ub, g_last_tail_ub, coeff_ub, u_ub, ws_ub, kv_ub);
    }
    paired_vec_store_final<D, C, StoreFinalStateCache>(
        FS_handle, final_state_cache, state_indices, primary_head, H, vid,
        state_index_stride, state_cache_slots, output_final_state, s_ub,
        s_ub_half);
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
  } else
#endif
  if (use_segmented_head_pipeline) {
    const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
    const int32_t segments_per_head = num_chunks / SEGMENT_CHUNKS;
    const int64_t total_segments =
        static_cast<int64_t>(segments_per_head) * H;
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_ws_base = ws_core_offset;
    const int64_t ws_k_base = ws_field_span + ws_core_offset;
    const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;

    for (int64_t task = cid;
         cid < segmented_h_worker_count && task < total_segments;
         task += segmented_h_worker_count) {
      const int32_t segment_idx = static_cast<int32_t>(task / H);
      const int64_t head = task % H;
      const int64_t head_g = head / GROUP;
      const int32_t chunk_begin = segment_idx * SEGMENT_CHUNKS;
      const int32_t chunk_end = chunk_begin + SEGMENT_CHUNKS;
      const int64_t v_head_base = head * static_cast<int64_t>(C) * D;

      if (segment_idx == 0) {
        paired_vec_init_zero<D, C, 0, true>(
            S_handle, head, 0, H, vid, s_ub, s_ub_half);
      } else {
        wait_segment_state<C>(
            h_o_ready_handle, head, chunk_begin, vid);
        paired_vec_load_state<D, C, 0>(
            S_handle, head, 0, chunk_begin, H, vid, s_ub, s_ub_half);
      }

      for (int32_t ci = chunk_begin; ci < chunk_end; ++ci) {
        const int64_t chunk_start = static_cast<int64_t>(ci) * C;
        paired_vec_front<D, C, 0, G_UB, COEFF_UB, U_UB>(
            K_handle, U_handle, G_handle, V_handle, workspace_handle, head,
            head_g, chunk_start, C, ci, H, Hg, total_tokens, vid, D,
            v_head_base, v_chunk_stride, ws_ws_base, ws_k_base, s_ub,
            k_ub_half, g_ub, u_ub_half, k_ub, g_last_tail_ub, coeff_ub,
            u_ub, ws_ub);
        paired_vec_finish<D, C, 0>(
            S_handle, workspace_handle, h_o_ready_handle, head, 0, ci,
            num_chunks, H, vid, ws_kv_base, true, s_ub,
            s_ub_half, kv_ub);
      }

      if (segment_idx + 1 < segments_per_head) {
        publish_segment_state<C>(
            h_o_ready_handle, head, chunk_end, vid);
      } else {
        paired_vec_store_final<D, C, StoreFinalStateCache>(
            FS_handle, final_state_cache, state_indices, head, H, vid,
            state_index_stride, state_cache_slots, output_final_state, s_ub,
            s_ub_half);
        ffts_cross_core_sync(
            PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
      }
    }
  } else if (use_split_head_pipeline && cid < paired_core_count) {
    const int64_t head0 = cid;
    const int64_t head1 = block_num + cid;
    const int64_t head_g0 = head0 / GROUP;
    const int64_t head_g1 = head1 / GROUP;
    const int64_t bos = static_cast<int64_t>(cu_seqlens[0]);
    const int64_t slen = static_cast<int64_t>(cu_seqlens[1]) - bos;
    const int32_t num_chunks = static_cast<int32_t>(slen / C);
    const int32_t split_chunk = num_chunks / 2;
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_slot_span = 4 * ws_field_span;
    const int64_t ws_ws_base0 = ws_core_offset;
    const int64_t ws_k_base0 = ws_field_span + ws_core_offset;
    const int64_t ws_s_base0 = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base0 = 3 * ws_field_span + ws_core_offset;
    const int64_t ws_ws_base1 = ws_slot_span + ws_ws_base0;
    const int64_t ws_k_base1 = ws_slot_span + ws_k_base0;
    const int64_t ws_s_base1 = ws_slot_span + ws_s_base0;
    const int64_t ws_kv_base1 = ws_slot_span + ws_kv_base0;
    const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;
    const int64_t v_head_base0 = head0 * static_cast<int64_t>(C) * D;
    const int64_t v_head_base1 = head1 * static_cast<int64_t>(C) * D;

    paired_vec_init_zero<D, C, 0, true>(
        S_handle, head0, 0, H, vid, s_ub, s_ub_half);
    paired_vec_init_zero<D, C, 4, true>(
        S_handle, head1, 0, H, vid, s_alt_ub, s_ub_half);
#ifdef MEGA_STOP_AFTER_H
    if constexpr (GDN_PAIR_DEBUG_STAGE == 1) return;
    if constexpr (GDN_PAIR_DEBUG_STAGE == 14 ||
                  GDN_PAIR_DEBUG_STAGE == 15) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 10) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 12) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 13) {
      wait_flag_dev(0);
      return;
    }
    if constexpr (GDN_PAIR_DEBUG_STAGE == 2 ||
                  GDN_PAIR_DEBUG_STAGE == 8 ||
                  GDN_PAIR_DEBUG_STAGE == 9 ||
                  GDN_PAIR_DEBUG_STAGE == 11) {
      wait_flag_dev(0);
      return;
    }
#endif

    for (int32_t ci = 0; ci < split_chunk; ++ci) {
      const int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      paired_vec_front<D, C, 0, G_UB, COEFF_UB, U_UB>(
          K_handle, U_handle, G_handle, V_handle, workspace_handle, head0,
          head_g0, chunk_start, C, ci, H, Hg, total_tokens, vid, D,
          v_head_base0, v_chunk_stride, ws_ws_base0, ws_k_base0, s_ub,
          k_ub_half, g_ub, u_ub_half, k_ub, g_last_tail_ub, coeff_ub,
          u_ub, ws_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 3) return;
      if constexpr (GDN_PAIR_DEBUG_STAGE == 4) {
        wait_flag_dev(4);
        return;
      }
#endif
      paired_vec_front<D, C, 4, G_UB, COEFF_UB, U_UB>(
          K_handle, U_handle, G_handle, V_handle, workspace_handle, head1,
          head_g1, chunk_start, C, ci, H, Hg, total_tokens, vid, D,
          v_head_base1, v_chunk_stride, ws_ws_base1, ws_k_base1, s_alt_ub,
          k_ub_half, g_ub, u_ub_half, k_ub, g_last_tail_ub, coeff_ub,
          u_ub, ws_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 5) return;
#endif
      paired_vec_finish<D, C, 0>(
          S_handle, workspace_handle, h_o_ready_handle, head0, 0, ci,
          num_chunks, H, vid, ws_kv_base0, true, s_ub,
          s_ub_half, kv_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 6) return;
#endif
      paired_vec_finish<D, C, 4>(
          S_handle, workspace_handle, h_o_ready_handle, head1, 0, ci,
          num_chunks, H, vid, ws_kv_base1, true, s_alt_ub,
          s_ub_half, kv_ub);
#ifdef MEGA_STOP_AFTER_H
      if constexpr (GDN_PAIR_DEBUG_STAGE == 7) return;
#endif
    }

    publish_segment_state<C>(
        h_o_ready_handle, head1, split_chunk, vid);

  } else {
  // Vec owns the running recurrent state S_i and updates it after every chunk.
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t head_g = head / GROUP;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
    const bool emit_precomputed_qs =
        !FuseResidentOutput && precompute_qs != 0 && C == D && H >= 8 &&
        cu_seqlens != nullptr;
    const bool use_resident_packed_v =
        FuseResidentOutput && precompute_qs != 0 && C == D && H >= 8 &&
        cu_seqlens != nullptr;
    const bool use_packed_v =
        (emit_precomputed_qs || use_resident_packed_v) &&
        batch_size == 1 && (slen % C) == 0;
#else
    const bool emit_precomputed_qs =
        precompute_qs != 0 && C == D && H >= 8 && cu_seqlens != nullptr;
    const bool use_packed_v =
        emit_precomputed_qs && batch_size == 1 && (slen % C) == 0;
#endif
    const int32_t v_stride =
        use_packed_v ? D : BSND_QKV_STRIDE;
    const int64_t v_head_base =
        use_packed_v
            ? (chunk_offset * H + head) *
                  static_cast<int64_t>(C) * D
            : (bos * H + head) * static_cast<int64_t>(D);
    const int64_t v_chunk_stride =
        static_cast<int64_t>(H) * C * D;
    const int64_t ws_core_offset =
        static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
    const int64_t ws_field_span =
        static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
    const int64_t ws_ws_base = ws_core_offset;
    const int64_t ws_k_base = ws_field_span + ws_core_offset;
    const int64_t ws_s_base = 2 * ws_field_span + ws_core_offset;
    const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;

    if (has_initial_state != 0) {
      if constexpr (LoadInitialStateCache) {
        const int32_t state_index = initial_state_indices[seq_idx];
        if (state_index >= 0 && state_index < state_cache_slots) {
          const int64_t cache_offset =
              (static_cast<int64_t>(state_index) * H + head) * DD +
              vid * HalfC * D;
          GmShape2D cache_shape(HalfC, D);
          GmStride2D cache_stride(D);
          GmTensor2D<float> cache_global(initial_state_cache + cache_offset,
                                         cache_shape, cache_stride);
          DynVecTile<float, HalfC, D> cache_load(HalfC, D);
          TASSIGN(cache_load, S_UB);
          TLOAD(cache_load, cache_global);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          // Match the standalone FP32 cache -> BF16 initial-state boundary.
          TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_RINT);
          TCVT(s_ub, s_ub_half, pto::RoundMode::CAST_NONE);
        } else {
          TEXPANDS(s_ub, 0.0f);
          TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
        }
      } else {
        int64_t h0_offset = (seq_idx * H + head) * DD + vid * HalfC * D;
#if defined(GDN_A5_KERNEL)
        // H0 GM lines may be stale in this Vector core's cache when the address
        // was read by a previous work item or launch.
        for (int32_t r = 0; r < HalfC * D; r += 16) {
          dcci(static_cast<__gm__ void *>(H0_handle + h0_offset + r),
               SINGLE_CACHE_LINE);
        }
#endif
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        GmShape2D h0_shape(HalfC, D);
        GmStride2D h0_stride(D);
        GmTensor2D<ComputeT> h0_global(H0_handle + h0_offset, h0_shape,
                                     h0_stride);
        DynVecTile<ComputeT, HalfC, D> h0_load(HalfC, D);
        TASSIGN(h0_load, S_UB_HALF);
        TLOAD(h0_load, h0_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(s_ub, s_ub_half, pto::RoundMode::CAST_NONE);
      }
    } else {
      // Start each sequence/head recurrence from S_0 = 0.
      TEXPANDS(s_ub, 0.0f);
      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      int64_t s_out_offset = (chunk_offset * H + head) * DD;
      GmShape2D s_out_shape(HalfC, D);
      GmStride2D s_out_stride(D);
      GmTensor2D<ComputeT> s_out_global(
          S_handle + s_out_offset + vid * HalfC * D, s_out_shape,
          s_out_stride);
      DynVecTile<ComputeT, HalfC, D> s_out_store(HalfC, D);
      TASSIGN(s_out_store, S_UB_HALF);
      TSTORE(s_out_global, s_out_store);
    }
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

    int64_t chunk_start_0 = bos;
    int64_t valid0 = slen;
    if (valid0 > C) valid0 = C;
    // Vec work is split by row stripe, not by individual token.  For the first
    // chunk we compute exactly how many live rows belong to this sub-block's
    // HalfC stripe so short tails do not overrun the packed BSND input.
    int32_t valid_rows_0 =
        static_cast<int32_t>(valid0 - static_cast<int64_t>(vid) * HalfC);
    if (valid_rows_0 < 0) valid_rows_0 = 0;
    if (valid_rows_0 > HalfC) valid_rows_0 = HalfC;

    int64_t k_offset_0 =
        (chunk_start_0 * Hg + head_g) * D + vid * HalfC * BSND_K_STRIDE;
    if (valid_rows_0 > 0) {
#if defined(GDN_A5_KERNEL)
      // K only feeds the state update (final_state); a stale line here does
      // not show up in the main output, so it must be invalidated explicitly.
      for (int32_t row = 0; row < valid_rows_0; ++row) {
        for (int32_t r = 0; r < D; r += 16) {
          dcci(static_cast<__gm__ void *>(
                   K_handle + k_offset_0 + row * BSND_K_STRIDE + r),
               SINGLE_CACHE_LINE);
        }
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      GmShape2D k_shape(valid_rows_0, D);
      GmStride2D k_stride(BSND_K_STRIDE);
      GmTensor2D<ComputeT> k_global(K_handle + k_offset_0, k_shape, k_stride);
      DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> k_load(valid_rows_0, D);
      TASSIGN(k_load, K_UB_HALF);
      TLOAD(k_load, k_global);
      if (valid_rows_0 != HalfC) {
        TFILLPAD_INPLACE(k_ub_half, k_load);
      }
    } else {
      // Empty stripe (typically vid=1 on a very short tail chunk): synthesize
      // a zero tile so later full-width vector math and workspace stores still
      // observe proper padding semantics.
      TEXPANDS(k_ub, 0.0f);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
    }

    {
#if defined(GDN_A5_KERNEL)
      // g (log-sigmoid gate) only feeds the state recurrence; same
      // consumer-side stale-line hazard as K.
      for (int32_t r = 0; r < valid0 * 4; r += 32) {
        dcci(static_cast<__gm__ void *>(
                 reinterpret_cast<__gm__ char *>(
                     G_handle + head * total_tokens + chunk_start_0) + r),
             SINGLE_CACHE_LINE);
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      GmShape2D g_shape(1, static_cast<int32_t>(valid0));
      GmStride2D g_stride(1);
      GmTensor2D<float> g_global(G_handle + head * total_tokens + chunk_start_0,
                                 g_shape, g_stride);
      DynVecTile<float, 1, C, pto::PadValue::Zero> g_load(
          1, static_cast<int32_t>(valid0));
      TASSIGN(g_load, G_UB);
      TLOAD(g_load, g_global);
      if (valid0 != C) {
        TFILLPAD_INPLACE(g_ub, g_load);
      }
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;
      int32_t valid_rows =
          static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
      if (valid_rows < 0) valid_rows = 0;
      if (valid_rows > HalfC) valid_rows = HalfC;
      // Each Vec subblock owns one contiguous HalfC-row stripe of the chunk.
      // For short tail chunks, `valid_rows` may be smaller or even zero.  This
      // is the key fix that keeps ragged tails and dense varlen boundary mixes
      // from reading or writing beyond the live rows in this stripe.

      int64_t u_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
      if (valid_rows > 0) {
        GmShape2D u_shape(valid_rows, D);
        GmStride2D u_stride(BSND_QKV_STRIDE);
        GmTensor2D<ComputeT> u_global(U_handle + u_offset, u_shape, u_stride);
        DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> u_load(valid_rows, D);
        TASSIGN(u_load, U_UB_HALF);
        TLOAD(u_load, u_global);
        if (valid_rows != HalfC) {
          TFILLPAD_INPLACE(u_ub_half, u_load);
        }
      } else {
        // No live rows for this stripe in the current chunk; keep the tile
        // explicitly zero-padded so the remainder of the recurrence logic can
        // run in full-tile form without special-casing every later step.
        TEXPANDS(u_ub, 0.0f);
        TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
      }

      TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

      TileUbDataND<float, 1, 64, 1, 64> g_ub_temp;
      TASSIGN(g_ub_temp, G_UB + vid * 64 * sizeof(float));

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      // Rebase the chunk gate around g_last so the intra-chunk decay stays numerically local.
      // Torch-like:
      //   coeff = exp(g_last - g_rows_owned_by_this_subblock)
      TADDS(coeff_ub, g_ub_temp, -g_last);
      pipe_barrier(PIPE_V);
      TNEG(coeff_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);

      if (valid == C) {
        // Only exp(g_last) is consumed below. Keep the tail path unchanged,
        // while avoiding two full-vector EXP repeats for dense chunks.
        TEXP(g_last_tail_ub, g_last_tail_ub);
      } else {
        TEXP(g_ub, g_ub);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      TileUbDataDN<float, HalfC, 1, HalfC, 1> coeff_col_ub;
      TASSIGN(coeff_col_ub, COEFF_UB);
      TileUbDataND<float, HalfC, D, HalfC, D> coeff_2d_ub;
      TASSIGN(coeff_2d_ub, U_UB);
      // Broadcast one decay scalar per token row across the D feature columns:
      //   coeff_2d[row, :] = coeff[row]
      TROWEXPAND(coeff_2d_ub, coeff_col_ub);
      pipe_barrier(PIPE_V);
      // `k_ub` now holds k_tilde = exp(g_last - g_i) * K_i.
      TMUL(k_ub, k_ub, coeff_2d_ub);
      pipe_barrier(PIPE_V);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
      TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

      wait_flag_dev(0);
#if defined(GDN_A5_KERNEL)
      // A5 Vector MTE2 loads can return clean stale lines when the workspace
      // addresses were read by a previous launch.  Invalidate the Cube
      // product rows on the consumer immediately before loading them.
      for (int32_t r = 0; r < HalfC * D; r += 16) {
        dcci(static_cast<__gm__ void *>(
                 workspace_handle + ws_ws_base + vid * HalfC * D + r),
             SINGLE_CACHE_LINE);
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      {
        GmShape2D ws_shape(HalfC, D);
        GmStride2D ws_stride(D);
        GmTensor2D<ComputeT> ws_global(
            workspace_handle + ws_ws_base + vid * HalfC * D,
            ws_shape, ws_stride);
        DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> ws_load(HalfC, D);
        TASSIGN(ws_load, U_UB_HALF);
        TLOAD(ws_load, ws_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      // v_i_new = U_i - W_i @ S_i.
      // In PyTorch notation:
      //   u_ub = u_ub - ws_ub
      TSUB(u_ub, u_ub, ws_ub);
      pipe_barrier(PIPE_V);
      // W@S is formed as a full C-row Cube tile. On a ragged tail, rows
      // outside valid_rows are not part of the sequence and must not feed
      // the later K^T V state update, especially when S starts non-zero.
      if (valid_rows == 0) {
        TEXPANDS(u_ub, 0.0f);
      } else if (valid_rows != HalfC) {
        DynVecTile<float, HalfC, D, pto::PadValue::Zero> u_live(
            valid_rows, D);
        TASSIGN(u_live, U_UB);
        TileUbDataND<float, HalfC, D, HalfC, D,
                     pto::PadValue::Zero> u_ub_padded;
        TASSIGN(u_ub_padded, U_UB);
        TFILLPAD_INPLACE(u_ub_padded, u_live);
      }
      pipe_barrier(PIPE_V);
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      const int64_t v_offset =
          v_head_base + static_cast<int64_t>(ci) * v_chunk_stride +
          static_cast<int64_t>(vid) * HalfC * v_stride;
      if (valid_rows > 0) {
        GmShape2D v_shape(valid_rows, D);
        GmStride2D v_gm_stride(v_stride);
        GmTensor2D<ComputeT> v_global(
            V_handle + v_offset, v_shape, v_gm_stride);
        DynVecTile<ComputeT, HalfC, D> v_store(valid_rows, D);
        TASSIGN(v_store, U_UB_HALF);
        TSTORE(v_global, v_store);
      }

      // Spill both V_i_new and k_i_tilde so the Cube stage can form
      // k_i_tilde^T @ V_i_new for this chunk.
      {
        GmShape2D k_shape(HalfC, D);
        GmStride2D k_stride(D);
        GmTensor2D<ComputeT> k_global(
            workspace_handle + ws_k_base + vid * HalfC * D,
            k_shape, k_stride);
        DynVecTile<ComputeT, HalfC, D> k_store(HalfC, D);
        TASSIGN(k_store, K_UB_HALF);
        TSTORE(k_global, k_store);
      }

      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      float exp_g_last =
          valid == C
              ? g_last_tail_ub.GetValue(ExpTailElements - 1)
              : g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      // Carry the recurrence across chunks: S_{i+1} = exp(g_last) * S_i + K_i^T V_i.
#if defined(GDN_A5_KERNEL)
      for (int32_t row = 0; row < HalfC; ++row) {
        TileUbDataND<float, 1, D> s_row;
        TASSIGN(s_row, S_UB + row * D * static_cast<int32_t>(sizeof(float)));
        TMULS(s_row, s_row, exp_g_last);
      }
#else
      TMULS(s_ub, s_ub, exp_g_last);
#endif

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        int64_t next_start = bos + static_cast<int64_t>(ci + 1) * C;
        int64_t next_valid = slen - static_cast<int64_t>(ci + 1) * C;
        if (next_valid > C) next_valid = C;
        int32_t next_valid_rows = static_cast<int32_t>(
            next_valid - static_cast<int64_t>(vid) * HalfC);
        if (next_valid_rows < 0) next_valid_rows = 0;
        if (next_valid_rows > HalfC) next_valid_rows = HalfC;

        int64_t nk_off =
            (next_start * Hg + head_g) * D + vid * HalfC * BSND_K_STRIDE;
        if (next_valid_rows > 0) {
#if defined(GDN_A5_KERNEL)
          // Same stale-line invalidation for the prefetched K rows.
          for (int32_t row = 0; row < next_valid_rows; ++row) {
            for (int32_t r = 0; r < D; r += 16) {
              dcci(static_cast<__gm__ void *>(
                       K_handle + nk_off + row * BSND_K_STRIDE + r),
                   SINGLE_CACHE_LINE);
            }
          }
#endif
          set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          GmShape2D k_shape(next_valid_rows, D);
          GmStride2D k_stride(BSND_K_STRIDE);
          GmTensor2D<ComputeT> k_global(K_handle + nk_off, k_shape, k_stride);
          DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> k_load(
              next_valid_rows, D);
          TASSIGN(k_load, K_UB_HALF);
          TLOAD(k_load, k_global);
          if (next_valid_rows != HalfC) {
            TFILLPAD_INPLACE(k_ub_half, k_load);
          }
        } else {
          // Same tail-safe zero materialization for the prefetch path: the next
          // chunk may have no rows in this stripe even though the other stripe
          // is still active.
          TEXPANDS(k_ub, 0.0f);
          TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
        }

        {
#if defined(GDN_A5_KERNEL)
          // Same stale-line invalidation for the prefetched gate row.
          for (int32_t r = 0; r < next_valid * 4; r += 32) {
            dcci(static_cast<__gm__ void *>(
                     reinterpret_cast<__gm__ char *>(
                         G_handle + head * total_tokens + next_start) + r),
                 SINGLE_CACHE_LINE);
          }
#endif
          set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
          GmShape2D g_shape(1, static_cast<int32_t>(next_valid));
          GmStride2D g_stride(1);
          GmTensor2D<float> g_global(G_handle + head * total_tokens + next_start,
                                     g_shape, g_stride);
          DynVecTile<float, 1, C, pto::PadValue::Zero> g_load(
              1, static_cast<int32_t>(next_valid));
          TASSIGN(g_load, G_UB);
          TLOAD(g_load, g_global);
          if (next_valid != C) {
            TFILLPAD_INPLACE(g_ub, g_load);
          }
        }
      }

      wait_flag_dev(2);
      if (emit_precomputed_qs && vid == 0) {
        const int64_t ready_offset =
            (seq_idx * H + head) * H_O_READY_STRIDE;
        h_o_ready_gm.SetValue(ready_offset, static_cast<int32_t>(ci + 1));
        __asm__ __volatile__("");
        AscendC::DataCacheCleanAndInvalid<
            int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_ALL>(
            h_o_ready_gm[ready_offset]);
        __asm__ __volatile__("");
      }
#if defined(GDN_A5_KERNEL)
      // Same consumer-side invalidation for the k_tilde^T @ v_new product
      // published by the Cube stage; without it the recurrence can fold in a
      // stale line from a previous launch and corrupt the final state.
      for (int32_t r = 0; r < HalfC * D; r += 16) {
        dcci(static_cast<__gm__ void *>(
                 workspace_handle + ws_kv_base + vid * HalfC * D + r),
             SINGLE_CACHE_LINE);
      }
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      {
        GmShape2D kv_shape(HalfC, D);
        GmStride2D kv_stride(D);
        GmTensor2D<ComputeT> kv_global(
            workspace_handle + ws_kv_base + vid * HalfC * D,
            kv_shape, kv_stride);
        DynVecTile<ComputeT, HalfC, D, pto::PadValue::Zero> kv_load(HalfC, D);
        TASSIGN(kv_load, S_UB_HALF);
        TLOAD(kv_load, kv_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#if defined(GDN_A5_KERNEL)
      // A5 hardware quirk (also noted in the non-tiled A5 correctness path):
      // vector ops on 64x128 UB tiles can leave stale rows on a physical
      // Vector subblock.  Do the kv conversion and the state update row by
      // row so no tile row can go stale.
      for (int32_t row = 0; row < HalfC; ++row) {
        TileUbDataND<float, 1, D> kv_row_f;
        TileUbDataND<ComputeT, 1, D> kv_row_h;
        TASSIGN(kv_row_f, KV_UB + row * D * static_cast<int32_t>(sizeof(float)));
        TASSIGN(kv_row_h, S_UB_HALF + row * D * static_cast<int32_t>(sizeof(ComputeT)));
        TCVT(kv_row_f, kv_row_h, pto::RoundMode::CAST_NONE);
      }
#else
      TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
#endif
      pipe_barrier(PIPE_ALL);
      // Finish S_{i+1} = exp(g_last) * S_i + k_i_tilde^T @ v_i_new.
      // Torch-like:
      //   s_ub = s_ub + kv_ub
#if defined(GDN_A5_KERNEL)
      for (int32_t row = 0; row < HalfC; ++row) {
        TileUbDataND<float, 1, D> s_row;
        TileUbDataND<float, 1, D> kv_row;
        TileUbDataND<ComputeT, 1, D> out_row;
        TASSIGN(s_row, S_UB + row * D * static_cast<int32_t>(sizeof(float)));
        TASSIGN(kv_row, KV_UB + row * D * static_cast<int32_t>(sizeof(float)));
        TASSIGN(out_row, S_UB_HALF + row * D * static_cast<int32_t>(sizeof(ComputeT)));
        TADD(s_row, s_row, kv_row);
        TCVT(out_row, s_row, pto::RoundMode::CAST_NONE);
      }
#else
      TADD(s_ub, s_ub, kv_ub);
      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
#endif

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // Expose the post-chunk state so the next chunk and output snapshot
        // can see S_{i+1}. Conceptually:
        //   S_handle[chunk_idx + 1, head] = S_{i+1}
        int64_t s_out_offset =
            ((chunk_offset + ci + 1) * H + head) * DD;
        {
          GmShape2D s_out_shape(HalfC, D);
          GmStride2D s_out_stride(D);
          GmTensor2D<ComputeT> s_out_global(
              S_handle + s_out_offset + vid * HalfC * D, s_out_shape,
              s_out_stride);
          DynVecTile<ComputeT, HalfC, D> s_out_store(HalfC, D);
          TASSIGN(s_out_store, S_UB_HALF);
          TSTORE(s_out_global, s_out_store);
        }
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
      }

#ifdef MEGA_CHUNK_GDN_RESIDENT_HO
      if constexpr (FuseResidentOutput) {
        ResidentHoGateQkAndScaleQ<D, C>(
            Q_handle, G_handle, resident_raw_qk_mailbox,
            resident_combined_mailbox, resident_gated_qk_mailbox,
            static_cast<int64_t>(cid), chunk_start, total_tokens,
            static_cast<int32_t>(head), static_cast<int32_t>(head_g), Hg,
            static_cast<int32_t>(vid));
        ResidentHoConsumeCombined<
            D, C, FuseResidentGatedRmsNorm>(
            resident_combined_mailbox, resident_output_handle,
            resident_z_handle, static_cast<int64_t>(cid), chunk_start,
            static_cast<int32_t>(head), H, static_cast<int32_t>(vid),
            ci + 1 == static_cast<int32_t>(num_chunks));
      }
#endif

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      }
    }

    if (output_final_state != 0) {
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      int64_t fs_offset = (seq_idx * H + head) * DD;
      if constexpr (StoreFinalStateCache) {
        TCVT(s_ub, s_ub_half, pto::RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);
        const int32_t state_index =
            state_indices[seq_idx] * state_index_stride;
        if (state_index >= 0 && state_index < state_cache_slots) {
          const int64_t cache_offset =
              (static_cast<int64_t>(state_index) * H + head) * DD +
              vid * HalfC * D;
          GmShape2D cache_shape(HalfC, D);
          GmStride2D cache_stride(D);
          GmTensor2D<float> cache_global(final_state_cache + cache_offset,
                                        cache_shape, cache_stride);
          DynVecTile<float, HalfC, D> cache_store(HalfC, D);
          TASSIGN(cache_store, S_UB);
          TSTORE(cache_global, cache_store);
        }
      } else {
        GmShape2D fs_shape(HalfC, D);
        GmStride2D fs_stride(D);
        GmTensor2D<ComputeT> fs_global(
            FS_handle + fs_offset + vid * HalfC * D, fs_shape, fs_stride);
        DynVecTile<ComputeT, HalfC, D> fs_store(HalfC, D);
        TASSIGN(fs_store, S_UB_HALF);
        TSTORE(fs_global, fs_store);
      }
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
#if defined(GDN_A5_KERNEL)
      // The FS TSTORE reads S_UB_HALF asynchronously on the MTE3 pipe.  The
      // next work item on this block immediately reuses that UB region (H0
      // TLOAD, TCVT, TEXPANDS), which can overwrite rows the store has not
      // read yet and corrupt final_state with the next item's input data.
      // Order the following V-pipe writes after the store completes.
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      // A5 MIX Vector GM writes are cached per Vector core and are not
      // snooped by the next kernel.  Clean the address that was actually
      // written above.  Prefill stores directly into final_state_cache,
      // whereas the standalone operator stores into FS_handle; cleaning
      // FS_handle unconditionally leaves Prefill's cache slot stale.
      if constexpr (StoreFinalStateCache) {
        const int32_t state_index =
            state_indices[seq_idx] * state_index_stride;
        if (state_index >= 0 && state_index < state_cache_slots) {
          const int64_t cache_offset =
              (static_cast<int64_t>(state_index) * H + head) * DD +
              vid * HalfC * D;
          for (int32_t r = 0; r < HalfC * D; r += 16) {
            dcci(static_cast<__gm__ void *>(final_state_cache + cache_offset + r),
                 SINGLE_CACHE_LINE);
          }
        }
      } else {
        for (int32_t r = 0; r < HalfC * D; r += 16) {
          dcci(static_cast<__gm__ void *>(FS_handle + fs_offset +
                                          vid * HalfC * D + r),
               SINGLE_CACHE_LINE);
        }
      }
      dsb(DSB_ALL);
#endif
      set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
  }
  }

  if (use_split_head_pipeline) {
    if (cid < 2 * paired_core_count) {
      const bool helper_core = cid >= paired_core_count;
      const int64_t head =
          helper_core ? block_num + cid - paired_core_count : cid;
      const int64_t head_g = head / GROUP;
      const int32_t num_chunks = static_cast<int32_t>(dense_chunk_count);
      const int32_t split_chunk = num_chunks / 2;
      const int64_t ws_core_offset =
          static_cast<int64_t>(cid) * WS_FIELD_STRIDE;
      const int64_t ws_field_span =
          static_cast<int64_t>(block_num) * WS_FIELD_STRIDE;
      const int64_t ws_ws_base = ws_core_offset;
      const int64_t ws_k_base = ws_field_span + ws_core_offset;
      const int64_t ws_s_base = 2 * ws_field_span + ws_core_offset;
      const int64_t ws_kv_base = 3 * ws_field_span + ws_core_offset;
      const int64_t v_head_base = head * static_cast<int64_t>(C) * D;
      const int64_t v_chunk_stride = static_cast<int64_t>(H) * C * D;

      if (helper_core) {
        wait_segment_state<C>(
            h_o_ready_handle, head, split_chunk, vid);
        paired_vec_load_state<D, C, 0>(
            S_handle, head, 0, split_chunk, H, vid, s_ub, s_ub_half);
      } else {
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
      }

      for (int32_t ci = split_chunk; ci < num_chunks; ++ci) {
        const int64_t chunk_start = static_cast<int64_t>(ci) * C;
        paired_vec_front<D, C, 0, G_UB, COEFF_UB, U_UB>(
            K_handle, U_handle, G_handle, V_handle, workspace_handle, head,
            head_g, chunk_start, C, ci, H, Hg, total_tokens, vid, D,
            v_head_base, v_chunk_stride, ws_ws_base, ws_k_base, s_ub,
            k_ub_half, g_ub, u_ub_half, k_ub, g_last_tail_ub, coeff_ub,
            u_ub, ws_ub);
        paired_vec_finish<D, C, 0>(
            S_handle, workspace_handle, h_o_ready_handle, head, 0, ci,
            num_chunks, H, vid, ws_kv_base, true, s_ub,
            s_ub_half, kv_ub);
      }

      paired_vec_store_final<D, C, StoreFinalStateCache>(
          FS_handle, final_state_cache, state_indices, head, H, vid,
          state_index_stride, state_cache_slots, output_final_state, s_ub,
          s_ub_half);
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  }
#endif
}

#undef GDN_CHUNK_H_KERNEL
