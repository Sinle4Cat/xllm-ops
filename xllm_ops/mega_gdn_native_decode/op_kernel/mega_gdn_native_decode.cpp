/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "mega_gdn_native_decode_pto_kernel.h"
struct MegaGdnNativeDecodeTilingData {
  int64_t batch_size;
  int64_t sequence_length;
  int64_t num_k_heads;
  int64_t num_v_heads;
  int64_t conv_stride;
  int64_t ssm_stride;
  int64_t total_tokens;
  int64_t state_slots;
  int64_t conv_rows;
  int64_t qkv_stride;
  int64_t z_stride;
};
extern "C" __global__ __aicore__ void mega_gdn_native_decode(
    GM_ADDR qkv,
    GM_ADDR z,
    GM_ADDR b,
    GM_ADDR a,
    GM_ADDR conv_weight,
    GM_ADDR conv_state,
    GM_ADDR a_log,
    GM_ADDR dt_bias,
    GM_ADDR ssm_state,
    GM_ADDR state_indices,
    GM_ADDR query_start_loc,
    GM_ADDR num_accepted_tokens,
    GM_ADDR norm_weight,
    GM_ADDR conv_out,
    GM_ADDR conv_state_out,
    GM_ADDR ssm_state_out,
    GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  REGISTER_TILING_DEFAULT(MegaGdnNativeDecodeTilingData);
  GET_TILING_DATA_WITH_STRUCT(MegaGdnNativeDecodeTilingData, t, tiling);
  (void)workspace;
  if constexpr (TILING_KEY_IS(1)) {
    mega_gdn_native_decode_pto::Run<0, false, false, true>(
        reinterpret_cast<__gm__ bfloat16_t*>(qkv),
        reinterpret_cast<__gm__ bfloat16_t*>(z),
        reinterpret_cast<__gm__ bfloat16_t*>(b),
        reinterpret_cast<__gm__ bfloat16_t*>(a),
        reinterpret_cast<__gm__ bfloat16_t*>(conv_weight),
        reinterpret_cast<__gm__ bfloat16_t*>(conv_state),
        reinterpret_cast<__gm__ float*>(a_log),
        reinterpret_cast<__gm__ float*>(dt_bias),
        reinterpret_cast<__gm__ float*>(ssm_state),
        reinterpret_cast<__gm__ int*>(state_indices),
        reinterpret_cast<__gm__ int*>(query_start_loc),
        reinterpret_cast<__gm__ int*>(num_accepted_tokens),
        reinterpret_cast<__gm__ bfloat16_t*>(norm_weight),
        reinterpret_cast<__gm__ bfloat16_t*>(conv_out),
        reinterpret_cast<__gm__ bfloat16_t*>(conv_state_out),
        reinterpret_cast<__gm__ float*>(ssm_state_out),
        reinterpret_cast<__gm__ bfloat16_t*>(out),
        t.num_k_heads, t.num_v_heads, t.batch_size, t.sequence_length,
        t.conv_stride, t.ssm_stride, t.total_tokens, t.state_slots, t.conv_rows, t.qkv_stride, t.z_stride);
  }
}
#include "lib/matmul_intf.h"
