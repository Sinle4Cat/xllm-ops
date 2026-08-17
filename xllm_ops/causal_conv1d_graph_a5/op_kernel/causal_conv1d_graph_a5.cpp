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
  GM_ADDR user_workspace = workspace;
  if (workspace != nullptr) {
    user_workspace = AscendC::GetUserWorkspace(workspace);
  }
  NsCausalConv1d::RunCausalConv1dUpdate<DTYPE_X>(
      x, weight, bias, conv_states, nullptr, cache_indices, nullptr, nullptr, y,
      user_workspace, &graph_tiling);
}
