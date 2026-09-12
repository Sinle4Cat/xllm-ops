// Qwen3.5 prefill Conv-to-gated-RMSNorm single-launch PTO kernel.

#include "../gdn_prefill_arch.h"

struct MegaGdnPrefillOpKernelTilingData {
    uint32_t block_dim;
    uint32_t vector_task_count;
    uint32_t target_arch;
    uint32_t num_matrices;
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t num_key_heads;
    uint32_t token_block_size;
    uint32_t token_block_count;
    uint32_t base_dim;
    uint32_t base_dim_count;
    uint32_t conv_dim;
    uint32_t conv_state_slots;
    uint32_t conv_state_len;
    uint32_t ssm_state_slots;
    uint32_t checkpoint_stride;
    int64_t total_tokens;
    uint64_t ffts_addr;
};

#define GDN_PREFILL_COMPUTE_DTYPE half

#define GDN_COMPUTE_DTYPE GDN_PREFILL_COMPUTE_DTYPE
#define GDN_PUBLIC_DTYPE DTYPE_MIXED_QKV
#define MEGA_CHUNK_GDN_HELPERS_ONLY
#define MEGA_CHUNK_GDN_HELPER_NAMESPACE qwen35_e2e_pto
#define MEGA_GDN_BUILD_REV 2026082515

// A5 solve experiments.  The default uses the validated fp32 hybrid together
// with the A5 WY sliding pipeline.
// Override this definition for candidate builds only:
//   0: Cube diagonals + fp32 AIV off-diagonal recurrence (baseline)
//   1: Cube diagonals with fp16 recurrent state + fp32 AIV off-diagonal
//   2: blocked Cube sums with fp16 Cube/AIV block handoffs
//   3: resident full-Cube solve with one final fp16 handoff
//   4: full-matrix Cube-resident Neumann solve with one final packed handoff
//   5: megagdn-pto recursive Cube solve with direct final GM store
//   6: megagdn-pto recursive Cube solve with packed AIV layout conversion
//   7: variant 6 with a true fp32 final Cube-to-AIV handoff
//   8: variant 7 identity-publication diagnostic
//   9: variant 8 with an AIV BSND-scatter bypass diagnostic
//  10: direct recursive Cube solve with a fused MIX completion handoff
//  11: direct BSND recursive Cube solve with fp32 final handoff
//  12: variant 11 with the solve buffer copied to the public output
//  13: variant 11 compatibility alias (Newton refinement removed)
//  14: variant 13 with the final BSND scatter striped across both AIVs
//  15: variant 14 with A5 vector/cache optimizations
//  16: variant 15 with head-major packed W/U/V_new internal data flow
//  17: variant 16 with A5 multi-batch group-QK reuse
//  18: variant 17 with directed A5 group-QK intra-block synchronization
//  19: variant 18 without redundant per-chunk H-to-O ready counters
//  20: variant 19 with one-round A5 group-QK double-mailbox lookahead
//  21: variant 16 with A5 per-chunk H/O producer-consumer overlap
//  22: variant 16 with a full-chunk 2x64 recursive solve
//  23: variant 16 with A5 group-KK score/cumsum overlap
//  30: variant 16 with a four-mailbox sliding WY pipeline
//  31: variant 30 with width-4 rolling Conv partial updates in a RegBase VF
//  32: variant 30 with packed Q/K normalization and FP16 packing in a RegBase VF
//  33: variant 30 with both Conv RegBase C1 and packed normalization RegBase C2
//  36: variant 33 with a direct FP32 L0C-to-paired-AIV Solve handoff
//  37: variant 36 with an aligned multi-batch O-stage ordered two-bank pipeline
//  38: variant 36 with a RegBase FP32 H-state update and ComputeT snapshot
//  39: variant 38 with a RegBase post-RMSNorm SiLU/output finalization
//  41: variant 38 with fused RegBase RMSNorm, norm weight and SiLU output
//  42: variant 41 with RegBase gate-coefficient generation
//  44: variant 41 with owner-local group-QK and a persistent raw-QK UB tile
//  45: variant 44 restricted to high-parallelism H48/K16 mappings
//  46: variant 41 with fused RegBase QS conversion and row scaling
//  47: variant 45 with fused RegBase QS conversion and row scaling
//  48: variant 47 with one-lane-ahead BF16 QS prefetch in the group path
//  49: variant 48 with cross-group lane0 QS prefetch for full chunks
//  50: variant 47 with a packed head-major Solve-to-WY handoff
//  51: variant 47 with KKT-produced -M to remove Solve's (-I)@M GEMM
//  54: variant 51 with a four-register fixed-mask H-state update
//  56: variant 54 with direct scalar-one generation in O-stage RMSNorm
//  63: variant 56 with staggered A2/V and A1/K WY mailbox release
//  65: variant 63 with fused single-read A1/A2 vector preparation
//  68: variant 65 with published frontend K and direct full-chunk K-to-L1
//  69: variant 68 with direct full-chunk BSND V-to-L1 loads in WY
//  70: variant 69 with direct BF16 L0C-to-AIV QKV handoff in group O
//  71: variant 70 with two 64-row QK GEMMs and direct BF16 raw-QK handoff
//  72: variant 71 with block-local 16->32 and 32->64 Solve recursion
//  73: variant 72 with a three-GEMM final 64->128 Solve recursion
//  74: variant 72 with two 64-row final results handed directly to AIV
//  75: variant 73 with the optimized group-QK O path enabled for H24
//  80: variant 75 with fused O compute/public rounding boundaries
//  81: variant 80 with QKV+QS folded into the same O boundary pass
//  82: variant 81 with generic Cube2/Cube3 L0C-to-UB handoff and UB ping-pong
//  83: variant 81 with generic Cube1 QK L0C-to-UB handoff
#ifndef MEGA_GDN_A5_SOLVE_VARIANT
#if defined(GDN_PREFILL_ARCH_A5) && defined(__CCE_AICORE__) && \
    __CCE_AICORE__ == 310
#define MEGA_GDN_A5_SOLVE_VARIANT 82
#else
#define MEGA_GDN_A5_SOLVE_VARIANT 30
#endif
#endif

#if defined(GDN_PREFILL_ARCH_A5)
#define MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
// Qwen GQA shares one K head across several value heads.  Build each K*K^T
// tile once per key head for packed multi-sequence prefill instead of
// repeating the identical Cube GEMM for every value head.
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
#if MEGA_GDN_A5_SOLVE_VARIANT != 4 && MEGA_GDN_A5_SOLVE_VARIANT != 5 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 6 && MEGA_GDN_A5_SOLVE_VARIANT != 7 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 8 && MEGA_GDN_A5_SOLVE_VARIANT != 9 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 10 && MEGA_GDN_A5_SOLVE_VARIANT != 11 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 12 && MEGA_GDN_A5_SOLVE_VARIANT != 13 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 14 && MEGA_GDN_A5_SOLVE_VARIANT != 15 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 16 && MEGA_GDN_A5_SOLVE_VARIANT != 17 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 18 && MEGA_GDN_A5_SOLVE_VARIANT != 19 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 20 && MEGA_GDN_A5_SOLVE_VARIANT != 21 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 22 && MEGA_GDN_A5_SOLVE_VARIANT != 23 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 30 && MEGA_GDN_A5_SOLVE_VARIANT != 31 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 32 && MEGA_GDN_A5_SOLVE_VARIANT != 33 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 36 && MEGA_GDN_A5_SOLVE_VARIANT != 37 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 38 && MEGA_GDN_A5_SOLVE_VARIANT != 39 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 41 && MEGA_GDN_A5_SOLVE_VARIANT != 42 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 44 && MEGA_GDN_A5_SOLVE_VARIANT != 45 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 46 && MEGA_GDN_A5_SOLVE_VARIANT != 47 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 48 && MEGA_GDN_A5_SOLVE_VARIANT != 49 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 50 && MEGA_GDN_A5_SOLVE_VARIANT != 51 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 54 && MEGA_GDN_A5_SOLVE_VARIANT != 56 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 63 && MEGA_GDN_A5_SOLVE_VARIANT != 65 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 68 && MEGA_GDN_A5_SOLVE_VARIANT != 69 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 70 && MEGA_GDN_A5_SOLVE_VARIANT != 71 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 72 && MEGA_GDN_A5_SOLVE_VARIANT != 73 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 74 && MEGA_GDN_A5_SOLVE_VARIANT != 75 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 80 && MEGA_GDN_A5_SOLVE_VARIANT != 81 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 82 && MEGA_GDN_A5_SOLVE_VARIANT != 83 && \
    MEGA_GDN_A5_SOLVE_VARIANT != 84
#define MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT < 2
#define MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 1
#define MEGA_CHUNK_GDN_A5_FP16_INTERMEDIATE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 0
#define MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
#define MEGA_CHUNK_GDN_A5_SKIP_DIAGONAL_REFINEMENT
#define MEGA_CHUNK_GDN_A5_VECTOR_OFFDIAG
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 3
#define MEGA_CHUNK_GDN_A5_RESIDENT_FULL_CUBE_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 4
#define MEGA_CHUNK_GDN_A5_FULL_MATRIX_CUBE_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 5
#define MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 10
#define MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SYNC_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 11 || MEGA_GDN_A5_SOLVE_VARIANT == 12 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 13 || MEGA_GDN_A5_SOLVE_VARIANT == 14 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 15 || MEGA_GDN_A5_SOLVE_VARIANT == 16 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 17 || MEGA_GDN_A5_SOLVE_VARIANT == 18 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 19 || MEGA_GDN_A5_SOLVE_VARIANT == 20 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 21 || MEGA_GDN_A5_SOLVE_VARIANT == 22 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 23 || MEGA_GDN_A5_SOLVE_VARIANT == 30 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 31 || MEGA_GDN_A5_SOLVE_VARIANT == 32 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 33 || MEGA_GDN_A5_SOLVE_VARIANT == 36 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 37 || MEGA_GDN_A5_SOLVE_VARIANT == 38 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 39 || MEGA_GDN_A5_SOLVE_VARIANT == 41 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 14 || MEGA_GDN_A5_SOLVE_VARIANT == 15 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 16 || MEGA_GDN_A5_SOLVE_VARIANT == 17 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 18 || MEGA_GDN_A5_SOLVE_VARIANT == 19 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 20 || MEGA_GDN_A5_SOLVE_VARIANT == 21 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 22 || MEGA_GDN_A5_SOLVE_VARIANT == 23 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 30 || MEGA_GDN_A5_SOLVE_VARIANT == 31 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 32 || MEGA_GDN_A5_SOLVE_VARIANT == 33 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 36 || MEGA_GDN_A5_SOLVE_VARIANT == 37 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 38 || MEGA_GDN_A5_SOLVE_VARIANT == 39 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 15 || MEGA_GDN_A5_SOLVE_VARIANT == 16 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 17 || MEGA_GDN_A5_SOLVE_VARIANT == 18 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 19 || MEGA_GDN_A5_SOLVE_VARIANT == 20 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 21 || MEGA_GDN_A5_SOLVE_VARIANT == 22 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 23 || MEGA_GDN_A5_SOLVE_VARIANT == 30 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 31 || MEGA_GDN_A5_SOLVE_VARIANT == 32 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 33 || MEGA_GDN_A5_SOLVE_VARIANT == 36 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 37 || MEGA_GDN_A5_SOLVE_VARIANT == 38 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 39 || MEGA_GDN_A5_SOLVE_VARIANT == 41 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
#define MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#define MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 16 || MEGA_GDN_A5_SOLVE_VARIANT == 17 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 18 || MEGA_GDN_A5_SOLVE_VARIANT == 19 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 20 || MEGA_GDN_A5_SOLVE_VARIANT == 21 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 22 || MEGA_GDN_A5_SOLVE_VARIANT == 23 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 30 || MEGA_GDN_A5_SOLVE_VARIANT == 31 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 32 || MEGA_GDN_A5_SOLVE_VARIANT == 33 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 36 || MEGA_GDN_A5_SOLVE_VARIANT == 37 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 38 || MEGA_GDN_A5_SOLVE_VARIANT == 39 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_PACKED_WUV
#define MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#define MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 23
#define MEGA_CHUNK_GDN_A5_CUMSUM_GROUP_KK_OVERLAP
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 30 || MEGA_GDN_A5_SOLVE_VARIANT == 31 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 32 || MEGA_GDN_A5_SOLVE_VARIANT == 33 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 36 || MEGA_GDN_A5_SOLVE_VARIANT == 37 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 38 || MEGA_GDN_A5_SOLVE_VARIANT == 39 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 31 || MEGA_GDN_A5_SOLVE_VARIANT == 33 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 36 || MEGA_GDN_A5_SOLVE_VARIANT == 37 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 38 || MEGA_GDN_A5_SOLVE_VARIANT == 39 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_GDN_PREFILL_A5_CONV_REGBASE_C1
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 32 || MEGA_GDN_A5_SOLVE_VARIANT == 33 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 36 || MEGA_GDN_A5_SOLVE_VARIANT == 37 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 38 || MEGA_GDN_A5_SOLVE_VARIANT == 39 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_GDN_PREFILL_A5_PACKED_NORM_REGBASE_C2
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 36 || MEGA_GDN_A5_SOLVE_VARIANT == 37 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 38 || MEGA_GDN_A5_SOLVE_VARIANT == 39 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_SOLVE_FP32_DIRECT_HANDOFF
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 37
// Keep the validated full H-to-O MIX barrier and activate only O's banked
// raw-QK/gated-QK/QKV protocol. Variant21's per-chunk overlap stays disabled.
#define MEGA_CHUNK_GDN_A5_O_ORDERED_TWO_BANK
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 38 || MEGA_GDN_A5_SOLVE_VARIANT == 39 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 39
#define MEGA_CHUNK_GDN_A5_O_FINAL_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 41 || MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_O_RMS_FINAL_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 42
#define MEGA_CHUNK_GDN_A5_GATE_COEFF_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 44 || MEGA_GDN_A5_SOLVE_VARIANT == 45 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 47 || MEGA_GDN_A5_SOLVE_VARIANT == 48 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 49 || MEGA_GDN_A5_SOLVE_VARIANT == 50
#define MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
#define MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
#define MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
#define MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#define MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI
#define MEGA_CHUNK_GDN_A5_PACKED_WUV
#define MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#define MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#define MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#define MEGA_GDN_PREFILL_A5_CONV_REGBASE_C1
#define MEGA_GDN_PREFILL_A5_PACKED_NORM_REGBASE_C2
#define MEGA_CHUNK_GDN_A5_SOLVE_FP32_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE
#define MEGA_CHUNK_GDN_A5_O_RMS_FINAL_REGBASE
#define MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#define MEGA_CHUNK_GDN_A5_GROUP_QK_PERSISTENT_RAW
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 45 || MEGA_GDN_A5_SOLVE_VARIANT == 47 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 48 || MEGA_GDN_A5_SOLVE_VARIANT == 49 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 50
#define MEGA_CHUNK_GDN_A5_GROUP_QK_MIN_H48
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 46
#define MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
#define MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
#define MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
#define MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#define MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI
#define MEGA_CHUNK_GDN_A5_PACKED_WUV
#define MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#define MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#define MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#define MEGA_GDN_PREFILL_A5_CONV_REGBASE_C1
#define MEGA_GDN_PREFILL_A5_PACKED_NORM_REGBASE_C2
#define MEGA_CHUNK_GDN_A5_SOLVE_FP32_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE
#define MEGA_CHUNK_GDN_A5_O_RMS_FINAL_REGBASE
#define MEGA_CHUNK_GDN_A5_QS_SCALE_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 47 || MEGA_GDN_A5_SOLVE_VARIANT == 48 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 49 || MEGA_GDN_A5_SOLVE_VARIANT == 50
#define MEGA_CHUNK_GDN_A5_QS_SCALE_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 51 || MEGA_GDN_A5_SOLVE_VARIANT == 54 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 56 || MEGA_GDN_A5_SOLVE_VARIANT == 63 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 65 || MEGA_GDN_A5_SOLVE_VARIANT == 68 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 69 || MEGA_GDN_A5_SOLVE_VARIANT == 70 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 71 || MEGA_GDN_A5_SOLVE_VARIANT == 72 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
#define MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
#define MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
#define MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#define MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI
#define MEGA_CHUNK_GDN_A5_PACKED_WUV
#define MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#define MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#define MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#define MEGA_GDN_PREFILL_A5_CONV_REGBASE_C1
#define MEGA_GDN_PREFILL_A5_PACKED_NORM_REGBASE_C2
#define MEGA_CHUNK_GDN_A5_SOLVE_FP32_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE
#define MEGA_CHUNK_GDN_A5_O_RMS_FINAL_REGBASE
#define MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#define MEGA_CHUNK_GDN_A5_GROUP_QK_PERSISTENT_RAW
#define MEGA_CHUNK_GDN_A5_GROUP_QK_MIN_H48
#define MEGA_CHUNK_GDN_A5_QS_SCALE_REGBASE
#define MEGA_CHUNK_GDN_A5_PRECOMPUTED_M_NEG
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 54 || MEGA_GDN_A5_SOLVE_VARIANT == 56 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 63 || MEGA_GDN_A5_SOLVE_VARIANT == 65 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 68 || MEGA_GDN_A5_SOLVE_VARIANT == 69 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 70 || MEGA_GDN_A5_SOLVE_VARIANT == 71 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 72 || MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE4
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 56 || MEGA_GDN_A5_SOLVE_VARIANT == 63 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 65 || MEGA_GDN_A5_SOLVE_VARIANT == 68 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 69 || MEGA_GDN_A5_SOLVE_VARIANT == 70 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 71 || MEGA_GDN_A5_SOLVE_VARIANT == 72 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_O_RMS_DIRECT_ONE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 63 || MEGA_GDN_A5_SOLVE_VARIANT == 65 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 68 || MEGA_GDN_A5_SOLVE_VARIANT == 69 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 70 || MEGA_GDN_A5_SOLVE_VARIANT == 71 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 72 || MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 65 || MEGA_GDN_A5_SOLVE_VARIANT == 68 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 69 || MEGA_GDN_A5_SOLVE_VARIANT == 70 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 71 || MEGA_GDN_A5_SOLVE_VARIANT == 72 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 68 || MEGA_GDN_A5_SOLVE_VARIANT == 69 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 70 || MEGA_GDN_A5_SOLVE_VARIANT == 71 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 72 || MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_PUBLISH_INPUTS_BEFORE_WY
#define MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 69 || MEGA_GDN_A5_SOLVE_VARIANT == 70 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 71 || MEGA_GDN_A5_SOLVE_VARIANT == 72 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 70 || MEGA_GDN_A5_SOLVE_VARIANT == 71 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 72 || MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_GROUP_QKV_DIRECT_HANDOFF
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 71 || MEGA_GDN_A5_SOLVE_VARIANT == 72 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECT_HANDOFF
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 72 || MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_BLOCK_LOCAL_EARLY_RECURSION
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 73
#define MEGA_CHUNK_GDN_A5_THREE_GEMM_FINAL_RECURSION
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 74
// Keep v73's validated frontend/WY/H/O chain, but replace only the final
// full-chunk Solve assembly.  This standalone block intentionally avoids
// widening every historical variant predicate.
#define MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
#define MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
#define MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
#define MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#define MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI
#define MEGA_CHUNK_GDN_A5_PACKED_WUV
#define MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#define MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#define MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#define MEGA_GDN_PREFILL_A5_CONV_REGBASE_C1
#define MEGA_GDN_PREFILL_A5_PACKED_NORM_REGBASE_C2
#define MEGA_CHUNK_GDN_A5_SOLVE_FP32_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE
#define MEGA_CHUNK_GDN_A5_O_RMS_FINAL_REGBASE
#define MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#define MEGA_CHUNK_GDN_A5_GROUP_QK_PERSISTENT_RAW
#define MEGA_CHUNK_GDN_A5_GROUP_QK_MIN_H48
#define MEGA_CHUNK_GDN_A5_QS_SCALE_REGBASE
#define MEGA_CHUNK_GDN_A5_PRECOMPUTED_M_NEG
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE4
#define MEGA_CHUNK_GDN_A5_O_RMS_DIRECT_ONE
#define MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
#define MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
#define MEGA_CHUNK_GDN_A5_PUBLISH_INPUTS_BEFORE_WY
#define MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
#define MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
#define MEGA_CHUNK_GDN_A5_GROUP_QKV_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_BLOCK_LOCAL_EARLY_RECURSION
#define MEGA_CHUNK_GDN_A5_HALF_ROW_FINAL_HANDOFF
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 75 || MEGA_GDN_A5_SOLVE_VARIANT == 80 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 81 || MEGA_GDN_A5_SOLVE_VARIANT == 82 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 83 || MEGA_GDN_A5_SOLVE_VARIANT == 84
// Keep v73 bit-for-bit for frontend, Solve, WY, and H.  The only difference
// is that the already validated group-QK/direct-QK/direct-QKV O schedule may
// also serve H24/K8 (GROUP=3), instead of being restricted to H>=48.
#define MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
#define MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
#define MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
#define MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#define MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI
#define MEGA_CHUNK_GDN_A5_PACKED_WUV
#define MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#define MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#define MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#define MEGA_GDN_PREFILL_A5_CONV_REGBASE_C1
#define MEGA_GDN_PREFILL_A5_PACKED_NORM_REGBASE_C2
#define MEGA_CHUNK_GDN_A5_SOLVE_FP32_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE
#define MEGA_CHUNK_GDN_A5_O_RMS_FINAL_REGBASE
#define MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#define MEGA_CHUNK_GDN_A5_GROUP_QK_PERSISTENT_RAW
#define MEGA_CHUNK_GDN_A5_QS_SCALE_REGBASE
#define MEGA_CHUNK_GDN_A5_PRECOMPUTED_M_NEG
#define MEGA_CHUNK_GDN_A5_H_STATE_REGBASE4
#define MEGA_CHUNK_GDN_A5_O_RMS_DIRECT_ONE
#define MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
#define MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
#define MEGA_CHUNK_GDN_A5_PUBLISH_INPUTS_BEFORE_WY
#define MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
#define MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
#define MEGA_CHUNK_GDN_A5_GROUP_QKV_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECT_HANDOFF
#define MEGA_CHUNK_GDN_A5_BLOCK_LOCAL_EARLY_RECURSION
#define MEGA_CHUNK_GDN_A5_THREE_GEMM_FINAL_RECURSION
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 80 || MEGA_GDN_A5_SOLVE_VARIANT == 81 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 82 || MEGA_GDN_A5_SOLVE_VARIANT == 83 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 84
#define MEGA_CHUNK_GDN_A5_O_BOUNDARY_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 81 || MEGA_GDN_A5_SOLVE_VARIANT == 82 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 83 || MEGA_GDN_A5_SOLVE_VARIANT == 84
#define MEGA_CHUNK_GDN_A5_O_COMBINE_BOUNDARY_REGBASE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 84
// v81 base schedule with only the generic-varlen V DMA issued before the
// gated-QK wait; retain the validated L1 layout and GEMM3 fences.
#define MEGA_CHUNK_GDN_A5_GENERIC_V_PREFETCH
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 82
// For full C128 tiles in the non-group O path, publish Cube2/QS and Cube3/QKV
// directly into the paired AIV UBs. The two QKV slots mirror the existing GM
// mailbox ping-pong; per-half ready events and the existing free protocol
// protect slot reuse.
#define MEGA_CHUNK_GDN_A5_GENERIC_L0C2UB_PINGPONG
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 83
#define MEGA_CHUNK_GDN_A5_GENERIC_QK_L0C2UB
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 48 || MEGA_GDN_A5_SOLVE_VARIANT == 49
// Keep one extra BF16 QS half-tile above the persistent raw-QK cache.  The
// current lane consumes it before the same address is refilled for the next
// value head, hiding two of the three per-group QS GM reads behind QKV work.
#define MEGA_CHUNK_GDN_A5_GROUP_QS_PREFETCH
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 49
#define MEGA_CHUNK_GDN_A5_GROUP_QS_CROSS_PREFETCH
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 50
// The inverse is private to WY.  For workloads made entirely of full C=128
// chunks, publish it as [H,T,C] so both Solve's MTE3 and WY's MTE2 move
// contiguous rows.  Ragged workloads retain the validated BSND layout.
#define MEGA_CHUNK_GDN_A5_SOLVE_WY_PACKED_HANDOFF
#endif
// The experimental 64x64 L0C handoff is intentionally not selectable on
// CANN 9.1 post5: strided submatrix stores corrupt data and direct TINSERT
// raises 507015. Keep variant 22 on the validated full-128 fallback.
#if MEGA_GDN_A5_SOLVE_VARIANT == 17 || MEGA_GDN_A5_SOLVE_VARIANT == 18 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 19 || MEGA_GDN_A5_SOLVE_VARIANT == 20
#define MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 18 || MEGA_GDN_A5_SOLVE_VARIANT == 19 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 20
#define MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 19 || MEGA_GDN_A5_SOLVE_VARIANT == 20
#define MEGA_CHUNK_GDN_A5_GROUP_QK_SKIP_HO_READY
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 20
#define MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 21
#define MEGA_CHUNK_GDN_A5_HO_OVERLAP
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 12 || \
    defined(MEGA_GDN_PREFILL_DUMP_SOLVE_BUILD)
#define MEGA_CHUNK_GDN_A5_DUMP_SOLVE_OUTPUT
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 6
#define MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 7 || MEGA_GDN_A5_SOLVE_VARIANT == 8 || \
    MEGA_GDN_A5_SOLVE_VARIANT == 9
#define MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_FP32_SOLVE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 8 || MEGA_GDN_A5_SOLVE_VARIANT == 9
#define MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_IDENTITY_PROBE
#endif
#if MEGA_GDN_A5_SOLVE_VARIANT == 9
#define MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_BSND_SCATTER_PROBE
#endif
#endif
#if defined(GDN_PREFILL_ARCH_A2A3)
#define MEGA_CHUNK_GDN_PRECOMPUTED_SOLVE_AUX
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
#define MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#endif
// The included helpers keep the public BF16 boundary and cast into FP16 on
// A2/A3.
#include "../../../mega_chunk_gdn/op_kernel/mega_chunk_gdn.cpp"
#if defined(GDN_PREFILL_ARCH_A2A3)
#undef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#undef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
#undef MEGA_CHUNK_GDN_PRECOMPUTED_SOLVE_AUX
#endif
#undef MEGA_CHUNK_GDN_HELPER_NAMESPACE
#undef MEGA_CHUNK_GDN_HELPERS_ONLY
#if defined(GDN_PREFILL_ARCH_A5)
#undef MEGA_CHUNK_GDN_A5_O_COMBINE_BOUNDARY_REGBASE
#undef MEGA_CHUNK_GDN_A5_O_BOUNDARY_REGBASE
#undef MEGA_CHUNK_GDN_A5_GENERIC_V_PREFETCH
#undef MEGA_CHUNK_GDN_A5_GENERIC_L0C2UB_PINGPONG
#undef MEGA_CHUNK_GDN_A5_GENERIC_QK_L0C2UB
#undef MEGA_CHUNK_GDN_A5_THREE_GEMM_FINAL_RECURSION
#undef MEGA_CHUNK_GDN_A5_BLOCK_LOCAL_EARLY_RECURSION
#undef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECT_HANDOFF
#undef MEGA_CHUNK_GDN_A5_GROUP_QKV_DIRECT_HANDOFF
#undef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_V_L1
#undef MEGA_CHUNK_GDN_A5_WY_DIRECT_INPUT_K_L1
#undef MEGA_CHUNK_GDN_A5_PUBLISH_INPUTS_BEFORE_WY
#undef MEGA_CHUNK_GDN_A5_WY_FUSED_A1_A2_PREP
#undef MEGA_CHUNK_GDN_A5_WY_STAGGERED_MAILBOX
#undef MEGA_CHUNK_GDN_A5_O_RMS_DIRECT_ONE
#undef MEGA_CHUNK_GDN_A5_H_STATE_REGBASE4
#undef MEGA_CHUNK_GDN_A5_PRECOMPUTED_M_NEG
#undef MEGA_CHUNK_GDN_A5_SOLVE_WY_PACKED_HANDOFF
#undef MEGA_CHUNK_GDN_A5_GROUP_QS_CROSS_PREFETCH
#undef MEGA_CHUNK_GDN_A5_GROUP_QS_PREFETCH
#undef MEGA_CHUNK_GDN_A5_QS_SCALE_REGBASE
#undef MEGA_CHUNK_GDN_A5_GROUP_QK_MIN_H48
#undef MEGA_CHUNK_GDN_A5_GROUP_QK_PERSISTENT_RAW
#undef MEGA_CHUNK_GDN_A5_GATE_COEFF_REGBASE
#undef MEGA_CHUNK_GDN_A5_O_RMS_FINAL_REGBASE
#undef MEGA_CHUNK_GDN_A5_O_FINAL_REGBASE
#undef MEGA_CHUNK_GDN_A5_H_STATE_REGBASE
#undef MEGA_CHUNK_GDN_A5_O_ORDERED_TWO_BANK
#undef MEGA_CHUNK_GDN_A5_SOLVE_FP32_DIRECT_HANDOFF
#undef MEGA_CHUNK_GDN_A5_WY_SLIDING_PIPELINE
#undef MEGA_CHUNK_GDN_A5_CUMSUM_GROUP_KK_OVERLAP
#undef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_KK
#undef MEGA_CHUNK_GDN_A5_SPLIT64_SOLVE
#undef MEGA_CHUNK_GDN_A5_HO_OVERLAP
#undef MEGA_CHUNK_GDN_A5_GROUP_QK_DOUBLE_MAILBOX
#undef MEGA_CHUNK_GDN_A5_GROUP_QK_SKIP_HO_READY
#undef MEGA_CHUNK_GDN_A5_GROUP_QK_DIRECTED_SYNC
#undef MEGA_CHUNK_GDN_MULTI_BATCH_GROUP_QK
#undef MEGA_CHUNK_GDN_A5_GROUP_QK_REUSE
#undef MEGA_CHUNK_GDN_A5_WY_GROUP_K_REUSE
#undef MEGA_CHUNK_GDN_A5_WY_SINGLE_A_READ
#undef MEGA_CHUNK_GDN_A5_PACKED_WUV
#undef MEGA_CHUNK_GDN_A5_ENTIRE_CACHE_DCCI
#undef MEGA_CHUNK_GDN_A5_DUAL_AIV_WY
#undef MEGA_CHUNK_GDN_A5_SINGLE_POST_SOLVE_SYNC
#undef MEGA_CHUNK_GDN_A5_DUAL_FP32_SCATTER
#undef MEGA_CHUNK_GDN_A5_DUMP_SOLVE_OUTPUT
#undef MEGA_CHUNK_GDN_A5_DIRECT_RECURSIVE_CUBE_FP32_SOLVE
#undef MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SYNC_SOLVE
#undef MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_BSND_SCATTER_PROBE
#undef MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_IDENTITY_PROBE
#undef MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_FP32_SOLVE
#undef MEGA_CHUNK_GDN_A5_PACKED_RECURSIVE_CUBE_SOLVE
#undef MEGA_CHUNK_GDN_A5_REFERENCE_RECURSIVE_CUBE_SOLVE
#undef MEGA_CHUNK_GDN_A5_FULL_MATRIX_CUBE_SOLVE
#undef MEGA_CHUNK_GDN_A5_RESIDENT_FULL_CUBE_SOLVE
#undef MEGA_CHUNK_GDN_A5_SKIP_DIAGONAL_REFINEMENT
#undef MEGA_CHUNK_GDN_A5_VECTOR_OFFDIAG
#undef MEGA_CHUNK_GDN_A5_FP16_INTERMEDIATE
#undef MEGA_CHUNK_GDN_A5_CUBE_FP32_HANDOFF
#undef MEGA_CHUNK_GDN_A5_CUBE_DIAG_AIV_OFFDIAG
#undef MEGA_CHUNK_GDN_A5_BLOCKED_CUBE_SOLVE
#undef MEGA_CHUNK_GDN_A5_DUAL_AIV_SOLVE
#endif
#undef MEGA_GDN_A5_SOLVE_VARIANT
#undef GDN_PUBLIC_DTYPE
#undef GDN_COMPUTE_DTYPE

#define GDN_PREFILL_PACKED_QKV_DTYPE GDN_PREFILL_COMPUTE_DTYPE
#include "gdn_prefill_frontend.h"
#undef GDN_PREFILL_PACKED_QKV_DTYPE
#undef MEGA_GDN_PREFILL_A5_PACKED_NORM_REGBASE_C2
#undef MEGA_GDN_PREFILL_A5_CONV_REGBASE_C1

namespace {
constexpr uint64_t kAlignBytes = 512;
constexpr uint64_t kDtypeBytes = 2;
constexpr uint64_t kFloatBytes = 4;
constexpr uint64_t kHeadDim = 128;
constexpr uint64_t kChunkSize = 128;

AICORE inline uint64_t AlignWorkspace(uint64_t bytes)
{
    return (bytes + kAlignBytes - 1) / kAlignBytes * kAlignBytes;
}
}  // namespace

extern "C" __global__ __aicore__ void GDN_KERNEL_NAME(
    GM_ADDR mixed_qkv_ptr, GM_ADDR b_ptr, GM_ADDR a_ptr, GM_ADDR z_ptr,
    GM_ADDR conv_weight_ptr, GM_ADDR conv_state_ptr, GM_ADDR a_log_ptr,
    GM_ADDR dt_bias_ptr, GM_ADDR conv_state_read_indices_ptr,
    GM_ADDR conv_state_write_indices_ptr, GM_ADDR ssm_state_read_indices_ptr,
    GM_ADDR ssm_state_write_indices_ptr, GM_ADDR ssm_cache_ptr,
    GM_ADDR mask_lower_ptr, GM_ADDR mask_full_ptr, GM_ADDR minus_identity_ptr,
    GM_ADDR cu_seqlens_ptr, GM_ADDR norm_weight_ptr, GM_ADDR norm_output_ptr,
    GM_ADDR conv_state_out_ptr, GM_ADDR ssm_cache_out_ptr,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaGdnPrefillOpKernelTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaGdnPrefillOpKernelTilingData,
                                tiling_data, tiling);

#ifdef __CCE_AICORE__
    if (tiling_data.target_arch != GDN_PREFILL_ARCH_ID) {
        return;
    }
    GM_ADDR user_workspace = AscendC::GetUserWorkspace(workspace);
    const uint32_t block_dim = tiling_data.block_dim;
    const uint32_t batch_size = tiling_data.batch_size;
    const uint32_t num_heads = tiling_data.num_heads;
    const uint32_t num_key_heads = tiling_data.num_key_heads;
    const int64_t total_tokens = tiling_data.total_tokens;
    const uint64_t tokens = static_cast<uint64_t>(total_tokens);
    const uint64_t heads = num_heads;
    const uint64_t key_heads = num_key_heads;
    const uint64_t matrices = tiling_data.num_matrices;
    uint64_t offset = 0;

    GM_ADDR compact_conv_state_snapshot_ptr = user_workspace + offset;
    offset += AlignWorkspace(
        static_cast<uint64_t>(batch_size) * tiling_data.conv_state_len *
        tiling_data.conv_dim * kDtypeBytes);
    GM_ADDR packed_qkv_compute_ptr = user_workspace + offset;
    offset += AlignWorkspace(
        tokens * tiling_data.conv_dim * kDtypeBytes);
    GM_ADDR g_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kFloatBytes);
    GM_ADDR beta_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kDtypeBytes);
    GM_ADDR beta_compute_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kDtypeBytes);
    GM_ADDR minus_identity_compute_ptr = user_workspace + offset;
    offset += AlignWorkspace(3 * kChunkSize * kChunkSize * kDtypeBytes);
    GM_ADDR g_sum_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kFloatBytes);
    GM_ADDR g_t_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kFloatBytes);
    GM_ADDR beta_t_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kDtypeBytes);
    GM_ADDR a_matrix_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kChunkSize * kDtypeBytes);
    GM_ADDR a_inv_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kChunkSize * kDtypeBytes);
    GM_ADDR w_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kHeadDim * kDtypeBytes);
    GM_ADDR u_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kHeadDim * kDtypeBytes);
    GM_ADDR s_ptr = user_workspace + offset;
    offset += AlignWorkspace(matrices * kHeadDim * kHeadDim * kDtypeBytes);
    GM_ADDR v_new_ptr = user_workspace + offset;
    offset += AlignWorkspace(tokens * heads * kHeadDim * kDtypeBytes);

    const uint64_t tile_bytes = kChunkSize * kChunkSize * kDtypeBytes;
    GM_ADDR kkt_workspace_ptr = user_workspace + offset;
    offset += static_cast<uint64_t>(block_dim) * 2 * tile_bytes;
    GM_ADDR wy_workspace_a1_ptr = user_workspace + offset;
    offset += static_cast<uint64_t>(block_dim) * tile_bytes;
    GM_ADDR wy_workspace_a2_ptr = user_workspace + offset;
    offset += static_cast<uint64_t>(block_dim) * tile_bytes;
    GM_ADDR h_workspace_unaligned = user_workspace + offset;
    const uint64_t h_workspace_address =
        (reinterpret_cast<uint64_t>(h_workspace_unaligned) +
         GDN_H_WORKSPACE_ALIGNMENT_BYTES - 1) &
        ~(static_cast<uint64_t>(GDN_H_WORKSPACE_ALIGNMENT_BYTES) - 1);
    const uint64_t h_workspace_phased_address =
        h_workspace_address + GDN_H_WORKSPACE_PHASE_BYTES;
    GM_ADDR h_workspace_ptr =
        reinterpret_cast<GM_ADDR>(h_workspace_phased_address);
    GM_ADDR o_workspace_qk_ptr =
        h_workspace_ptr + static_cast<uint64_t>(block_dim) *
                              8 * (tile_bytes +
                                   GDN_H_WORKSPACE_PAD_BYTES);
    GM_ADDR o_workspace_qs_ptr =
        o_workspace_qk_ptr + static_cast<uint64_t>(block_dim) * tile_bytes;
    GM_ADDR o_workspace_gated_ptr =
        o_workspace_qs_ptr + static_cast<uint64_t>(block_dim) * tile_bytes;

    GdnPrefillFrontendTilingData frontend_tiling{};
    frontend_tiling.num_heads = num_heads;
    frontend_tiling.num_key_heads = num_key_heads;
    frontend_tiling.token_block_size = tiling_data.token_block_size;
    frontend_tiling.token_block_count = tiling_data.token_block_count;
    frontend_tiling.base_dim = tiling_data.base_dim;
    frontend_tiling.base_dim_count = tiling_data.base_dim_count;
    frontend_tiling.conv_dim = tiling_data.conv_dim;
    frontend_tiling.conv_state_slots = tiling_data.conv_state_slots;
    frontend_tiling.total_tokens = total_tokens;

    gdn_prefill_frontend::RunConv(
        mixed_qkv_ptr, conv_weight_ptr, conv_state_ptr, conv_state_out_ptr,
        conv_state_read_indices_ptr, conv_state_write_indices_ptr,
        cu_seqlens_ptr, packed_qkv_compute_ptr,
        compact_conv_state_snapshot_ptr, frontend_tiling,
        batch_size, tiling_data.conv_state_len);

#if defined(GDN_PREFILL_ARCH_A5)
    // RunConv owns a temporary TPipe and publishes both packed QKV and the
    // final convolution state through MTE3.  Drain those stores before the
    // same AIV initializes the gate stage and reuses pipe/event resources.
    // PR #41's intra-block flags order peer consumers, but they do not by
    // themselves protect this local TPipe lifetime boundary.
    pipe_barrier(PIPE_ALL);
#endif

#ifdef E2E_STOP_AFTER_CONV
    qwen35_e2e_pto::SyncAllImpl<false>();
    return;
#endif

    const bool round_g_to_bf16 =
#if defined(GDN_PREFILL_ARCH_A5)
        true;
#else
        tiling_data.checkpoint_stride > 1;
#endif
    gdn_prefill_frontend::PrepareGate<GDN_PREFILL_COMPUTE_DTYPE>(
        a_ptr, b_ptr, a_log_ptr, dt_bias_ptr, g_ptr, beta_compute_ptr,
        total_tokens, static_cast<int32_t>(num_heads),
        static_cast<int32_t>(tiling_data.vector_task_count),
        round_g_to_bf16);
    qwen35_e2e_pto::mega_prepare_solve_constants<
        bfloat16_t, GDN_PREFILL_COMPUTE_DTYPE>(
        reinterpret_cast<__gm__ bfloat16_t *>(minus_identity_ptr),
        reinterpret_cast<__gm__ GDN_PREFILL_COMPUTE_DTYPE *>(
            minus_identity_compute_ptr),
        static_cast<int64_t>(kChunkSize * kChunkSize));

    qwen35_e2e_pto::SyncAllImpl<false>();

#ifdef E2E_STOP_AFTER_FRONTEND
    return;
#endif

#ifdef E2E_STOP_AFTER_CAST
    return;
#endif

    const uint64_t q_bytes = tokens * key_heads * kHeadDim * kDtypeBytes;
    GM_ADDR q_ptr = packed_qkv_compute_ptr;
    GM_ADDR k_ptr = q_ptr + q_bytes;
    GM_ADDR v_ptr = k_ptr + q_bytes;
#if defined(GDN_PREFILL_ARCH_A5)
    // Recycling one of four KKT-to-Solve slots while the fifth producer wave
    // is still live can deadlock at 16 chunks. A5 uses the existing full
    // stage rendezvous instead of the pipelined slot protocol.
#endif
    qwen35_e2e_pto::mega_kernel_impl<true, true, true, true, true, true>(
        q_ptr, k_ptr, v_ptr, g_ptr, beta_compute_ptr, mask_lower_ptr,
        mask_full_ptr, minus_identity_compute_ptr, cu_seqlens_ptr,
        norm_output_ptr, g_sum_ptr,
        g_t_ptr, beta_t_ptr, a_matrix_ptr, a_matrix_ptr, a_inv_ptr, w_ptr,
        u_ptr, s_ptr, v_new_ptr, s_ptr, s_ptr,
        1, kkt_workspace_ptr,
        wy_workspace_a1_ptr, wy_workspace_a2_ptr, h_workspace_ptr,
        o_workspace_qk_ptr, o_workspace_qs_ptr, o_workspace_gated_ptr,
        static_cast<int32_t>(heads), num_key_heads, batch_size, total_tokens,
        total_tokens, static_cast<uint32_t>(matrices), tiling_data.ffts_addr,
        z_ptr, norm_weight_ptr, ssm_cache_out_ptr,
        ssm_state_write_indices_ptr, 1, ssm_cache_ptr,
        ssm_state_read_indices_ptr,
        static_cast<int64_t>(tiling_data.ssm_state_slots));
#if defined(GDN_PREFILL_ARCH_A2A3) && defined(__DAV_C220_VEC__)
    // A PIPE_ALL barrier orders issued work but does not acknowledge the final
    // GM stores. Drain MTE3 once before kernel completion so output and state
    // cache writes are visible to the following op and to host-side checks.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

#undef GDN_PREFILL_COMPUTE_DTYPE
