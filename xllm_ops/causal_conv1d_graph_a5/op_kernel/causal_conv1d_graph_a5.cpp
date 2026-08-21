/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#define CAUSAL_CONV1D_SKIP_TPL_REGISTRATION
#include "causal_conv1d_update.h"
#undef CAUSAL_CONV1D_SKIP_TPL_REGISTRATION

extern "C" __global__ __aicore__ void causal_conv1d_graph_a5(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR conv_states,
    GM_ADDR cache_indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
  GET_TILING_DATA_WITH_STRUCT(CausalConv1dTilingData, graph_tiling, tiling);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
#if defined(__DAV_VEC__)
  // Prefill publishes persistent Conv/SSM state from a different AIV kernel.
  // ACL graph replay can reuse any AIV with a stale private data-cache line,
  // so host tiling launches every AIV and each one invalidates locally before
  // the first state load.  The producer-side clean makes DDR authoritative;
  // this is the matching acquire required by the A5 cross-core GM contract.
  dcci(static_cast<__gm__ void*>(0), ENTIRE_DATA_CACHE);
  dsb(DSB_ALL);
#endif
  GM_ADDR user_workspace = workspace;
  if (workspace != nullptr) {
    user_workspace = AscendC::GetUserWorkspace(workspace);
  }
  NsCausalConv1d::RunCausalConv1dUpdate<DTYPE_X>(
      x, weight, bias, conv_states, nullptr, cache_indices, nullptr, nullptr, y,
      user_workspace, &graph_tiling);
}
