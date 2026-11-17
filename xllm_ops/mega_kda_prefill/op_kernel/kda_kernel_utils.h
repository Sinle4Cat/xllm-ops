/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <pto/pto-inst.hpp>

// A5 links AIC and AIV into one image; force architecture-dependent
// template bodies inline so weak symbols cannot select another core.
#if __CCE_AICORE__ == 310
#define KDA_ARCH_INLINE PTO_INLINE
#else
#define KDA_ARCH_INLINE
#endif

namespace mega_kda_prefill_utils {
/**
 * @brief Do a sync step (set-wait flag) between two pipes.
 *
 * @tparam SrcPipe The pipe that sets the flag.
 * @tparam DstPipe The pipe that waits for the flag.
 * @param [in] id The event id to sync for.
 */
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetWaitFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

template <pipe_t Pipe, uint8_t VEC_NUM = 2>
AICORE inline void SetCrossFlag(int32_t flag) {
  ffts_cross_core_sync(Pipe, 1 | (VEC_NUM << 4) | (flag << 8));
}

template <pipe_t Pipe>
AICORE inline void SignalBothVecOnA5(uint16_t flag) {
  // A5: the flag offset is 16 on new core.
  constexpr uint16_t VEC_FLAG_OFFSET = 16;

  set_intra_block(Pipe, flag);
  set_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

template <pipe_t Pipe>
AICORE inline void WaitBothVecOnA5(uint16_t flag) {
  // A5: the flag offset is 16 on new core.
  constexpr uint16_t VEC_FLAG_OFFSET = 16;

  wait_intra_block(Pipe, flag);
  wait_intra_block(Pipe, flag + VEC_FLAG_OFFSET);
}

/**
 * @brief Returns the outer matrix layout based on the target architecture and
 * matrix orientation.
 *
 * On DAV C310 targets, the layout depends on whether the matrix is "left-sided"
 * (L0A). DAV C310: L0A is NZ, L0B is ZN. Older: L0A is ZZ, L0B is ZN.
 *
 * Link:
 * https://pto-isa.github.io/docs/isa/cube/nz-fractal-layout/#per-buffer-nz-layouts
 *
 * @param is_left Whether the matrix is on the left side (L0A) or not (L0B).
 * @return The appropriate @c BLayout for the target architecture.
 */
constexpr inline pto::BLayout GetOuterLayout(bool is_left) {
#ifdef __DAV_C310__
  return is_left ? pto::BLayout::ColMajor : pto::BLayout::RowMajor;
#else
  return pto::BLayout::RowMajor;
#endif
}

/**
 * @brief Pipe in-core barrier for vector core that is a no-op for A5.
 *
 */
AICORE inline void PipeBarrierVec() {
#if __CCE_AICORE__ == 220
  pipe_barrier(PIPE_V);
#endif
}

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId) {
  constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
  constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;
  return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
          ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

}  // namespace mega_kda_prefill_utils
