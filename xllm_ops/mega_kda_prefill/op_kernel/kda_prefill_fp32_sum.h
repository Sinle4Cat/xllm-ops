/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

#include <pto/pto-inst.hpp>
#include <cstdint>

namespace kda_prefill {

// Shared byte layout for the bulk Conv frontend. R16/R32 retain their original
// offsets. R64 reuses the inactive window and narrow-conversion storage only
// after their consumers have drained at the existing vector barriers.
template <int R>
struct ConvRowsLayout {
  static_assert(R == 16 || R == 32 || R == 64);
  static constexpr int kD = 128;
  static constexpr int kFloatTile = R * kD * sizeof(float);
  static constexpr int kTemporary = R == 64 ? kFloatTile : 2 * kFloatTile;
  static constexpr int kBf = kTemporary + kFloatTile;
  static constexpr int kHalf = kBf + R * kD * 2;
  static constexpr int kWeight = kHalf + R * kD * 2;
  static constexpr int kWeightBf = kWeight + kD * sizeof(float);
  static constexpr int kScalar = kWeightBf + kD * 2;
  static constexpr int kReduce = kScalar + R * 8 * sizeof(float);
  static constexpr int kNorm = kReduce + R * 64 * sizeof(float);
  static constexpr int kCachedWeights = kNorm + kFloatTile;
  static constexpr int kWindow = kCachedWeights + 4 * kD * sizeof(float);
  static constexpr int kRawWindow = R == 64 ? kBf : kWindow + (R + 3) * kD * sizeof(float);
  static constexpr int kInput = R == 64 ? kWindow : kFloatTile;
  static constexpr int kFrontendLimit = R == 16 ? 65536 : R == 32 ? 131072 : 192 * 1024;
  static_assert(kWindow + (R + 3) * kD * sizeof(float) <= kFrontendLimit);
  static_assert(kRawWindow + (R + 3) * kD * 2 <= kFrontendLimit);
};

#if defined(__DAV_VEC__)
AICORE inline int compare_sqrt_midpoint(uint32_t input, uint32_t lower) {
  const int input_exp = static_cast<int>((input >> 23) & 0xff);
  const int lower_exp = static_cast<int>((lower >> 23) & 0xff);
  const uint64_t input_sig = (input & 0x7fffff) | 0x800000;
  const uint64_t twice_midpoint = 2 * ((lower & 0x7fffff) | 0x800000) + 1;
  // Compare x with ((lower + nextafter(lower, +inf)) / 2)^2 exactly.
  // The sqrt estimate is within a few ULPs, so the shift is 25--27.
  const uint64_t scaled_input = input_sig << (input_exp - 2 * lower_exp + 152);
  const uint64_t squared_midpoint = twice_midpoint * twice_midpoint;
  return (scaled_input > squared_midpoint) - (scaled_input < squared_midpoint);
}

AICORE inline uint32_t nearest_fp32_sqrt(uint32_t input, uint32_t estimate) {
  // Normalization adds a positive epsilon. Preserve the existing exceptional
  // behavior for NaN/Inf rather than applying integer midpoint comparisons.
  if (input < 0x00800000 || input >= 0x7f800000 ||
      estimate < 0x00800000 || estimate >= 0x7f800000) {
    return estimate;
  }
  for (int step = 0; step < 4; ++step) {
    const int upper = compare_sqrt_midpoint(input, estimate);
    if (upper > 0 || (upper == 0 && (estimate & 1))) {
      ++estimate;
      continue;
    }
    const int lower = compare_sqrt_midpoint(input, estimate - 1);
    if (lower < 0 || (lower == 0 && (estimate & 1))) {
      --estimate;
      continue;
    }
    break;
  }
  return estimate;
}

template <typename Out, typename In>
__tf__ PTO_INTERNAL void round_fp32_sqrt_impl(
    typename Out::TileDType __out__ output_data,
    typename Out::TileDType __in__ estimate_data,
    typename In::TileDType __in__ input_data) {
  auto* output = (__ubuf__ float*)__cce_get_tile_ptr(output_data);
  auto* estimate = (__ubuf__ float*)__cce_get_tile_ptr(estimate_data);
  auto* input = (__ubuf__ float*)__cce_get_tile_ptr(input_data);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int row = 0; row < Out::ValidRow; ++row) {
    union Bits { float value; uint32_t bits; } x, y;
    x.value = input[row * In::RowStride];
    y.value = estimate[row * Out::RowStride];
    y.bits = nearest_fp32_sqrt(x.bits, y.bits);
    output[row * Out::RowStride] = y.value;
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

// The frozen FP32 reference sums 128 features with eight lanes and four
// interleaved accumulators, then adds the eight lane totals in order. A
// reassociated reduction can cross a BF16 midpoint after normalization.
template <typename Out, typename In, typename Scratch>
__tf__ PTO_INTERNAL void sum_fp32_features_impl(
    typename Out::TileDType __out__ out_data,
    typename In::TileDType __in__ in_data,
    typename Scratch::TileDType __out__ scratch_data) {
  static_assert(In::Cols == 128 && In::ValidCol == 128);
  static_assert(Scratch::Cols >= 32);
  auto* out = (__ubuf__ float*)__cce_get_tile_ptr(out_data);
  auto* input = (__ubuf__ float*)__cce_get_tile_ptr(in_data);
  auto* scratch = (__ubuf__ float*)__cce_get_tile_ptr(scratch_data);
  constexpr uint8_t rows = In::ValidRow;
  constexpr uint8_t src_stride = In::RowStride / 8;
  constexpr uint8_t tmp_stride = Scratch::RowStride / 8;

  // Four independent eight-lane accumulators share a single32-lane issue.
  // Each lane retains the original four-addend FP32 accumulation order.
  set_mask_norm();
  set_vector_mask(0, 0xffffffffULL);
  vadd(scratch, input, input + 32,
       rows, 1, 1, 1, tmp_stride, src_stride, src_stride);
  pipe_barrier(PIPE_V);
  for (int step = 2; step < 4; ++step) {
    vadd(scratch, scratch, input + step * 32,
         rows, 1, 1, 1, tmp_stride, tmp_stride, src_stride);
    pipe_barrier(PIPE_V);
  }
  set_vector_mask(0, 0xff);
  for (int group = 1; group < 4; ++group) {
    vadd(scratch, scratch, scratch + group * 8,
         rows, 1, 1, 1, tmp_stride, tmp_stride, tmp_stride);
    pipe_barrier(PIPE_V);
  }
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int row = 0; row < rows; ++row) {
    float total = 0.0f;
    for (int lane = 0; lane < 8; ++lane) {
      total += scratch[row * Scratch::RowStride + lane];
    }
    out[row * Out::RowStride] = total;
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}
AICORE inline int compare_div_midpoint(uint32_t numerator,
                                      uint32_t denominator,
                                      uint32_t lower) {
  const int numerator_exp = static_cast<int>((numerator >> 23) & 0xff);
  const int denominator_exp = static_cast<int>((denominator >> 23) & 0xff);
  const int lower_exp = static_cast<int>((lower >> 23) & 0xff);
  const uint64_t numerator_sig = (numerator & 0x7fffff) | 0x800000;
  const uint64_t denominator_sig = (denominator & 0x7fffff) | 0x800000;
  const uint64_t midpoint_sig = 2 * ((lower & 0x7fffff) | 0x800000) + 1;
  // Compare |a| with b * (lower + nextafter(lower, +inf)) / 2.
  // For adjacent quotient estimates the shift is 23--25 bits.
  const uint64_t left = numerator_sig <<
      (numerator_exp - denominator_exp - lower_exp + 151);
  const uint64_t right = denominator_sig * midpoint_sig;
  return (left > right) - (left < right);
}

AICORE inline uint32_t nearest_fp32_quotient(uint32_t numerator,
                                           uint32_t denominator,
                                           uint32_t estimate) {
  const uint32_t sign = estimate & 0x80000000;
  numerator &= 0x7fffffff;
  estimate &= 0x7fffffff;
  // Leave exceptional and underflow behavior to the device operation.
  if (numerator < 0x00800000 || numerator >= 0x7f800000 ||
      denominator < 0x00800000 || denominator >= 0x7f800000 ||
      estimate <= 0x00800000 || estimate >= 0x7f7fffff) {
    return estimate | sign;
  }
  for (int step = 0; step < 32; ++step) {
    const int upper = compare_div_midpoint(numerator, denominator, estimate);
    if (upper > 0 || (upper == 0 && (estimate & 1))) {
      ++estimate;
      continue;
    }
    const int lower = compare_div_midpoint(numerator, denominator, estimate - 1);
    if (lower < 0 || (lower == 0 && (estimate & 1))) {
      --estimate;
      continue;
    }
    break;
  }
  return estimate | sign;
}

// Return the first set position without a scalar scan through all128 values.
// The argument is nonzero; the six binary tests use only scalar registers.
AICORE inline int first_set_bit(uint64_t bits) {
  int index = 0;
  if ((bits & 0xffffffffULL) == 0) { index += 32; bits >>= 32; }
  if ((bits & 0xffffULL) == 0) { index += 16; bits >>= 16; }
  if ((bits & 0xffULL) == 0) { index += 8; bits >>= 8; }
  if ((bits & 0xfULL) == 0) { index += 4; bits >>= 4; }
  if ((bits & 0x3ULL) == 0) { index += 2; bits >>= 2; }
  return index + ((bits & 1) == 0);
}

template <typename Data, typename Denominator, typename Mask>
__tf__ PTO_INTERNAL void round_bf16_midpoints_impl(
    typename Data::TileDType __out__ output_data,
    typename Data::TileDType __in__ estimate_data,
    typename Data::TileDType __in__ numerator_data,
    typename Denominator::TileDType __in__ denominator_data,
    typename Denominator::TileDType __in__ variance_data,
    typename Mask::TileDType __in__ mask_data) {
  auto* output = (__ubuf__ float*)__cce_get_tile_ptr(output_data);
  auto* estimate = (__ubuf__ float*)__cce_get_tile_ptr(estimate_data);
  auto* numerator = (__ubuf__ float*)__cce_get_tile_ptr(numerator_data);
  auto* denominator = (__ubuf__ float*)__cce_get_tile_ptr(denominator_data);
  auto* masks = (__ubuf__ uint64_t*)__cce_get_tile_ptr(mask_data);
  auto* variance = (__ubuf__ float*)__cce_get_tile_ptr(variance_data);
  static_assert(Data::ValidCol == 128 && Data::RowStride == 128);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int row = 0; row < Data::ValidRow; ++row) {
    // TCMPS packs128 comparisons into two64-bit words for each row.
    const uint64_t lower = masks[row * 2];
    const uint64_t upper = masks[row * 2 + 1];
    if ((lower | upper) == 0) {
      continue;
    }
    union Bits { float value; uint32_t bits; } den;
    den.value = denominator[row * Denominator::RowStride];
    Bits sum;
    sum.value = variance[row * Denominator::RowStride];
    den.bits = nearest_fp32_sqrt(sum.bits, den.bits);
    for (int half = 0; half < 2; ++half) {
      uint64_t active = half == 0 ? lower : upper;
      while (active != 0) {
        const int col = half * 64 + first_set_bit(active);
        active &= active - 1;
        const int offset = row * Data::RowStride + col;
        Bits q, a;
        q.value = estimate[offset];
        a.value = numerator[offset];
        q.bits = nearest_fp32_quotient(a.bits, den.bits, q.bits);
        output[offset] = q.value;
      }
    }
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

#endif

template <typename Out, typename In, typename Scratch>
AICORE inline void sum_fp32_features(Out& output, In& input, Scratch& scratch) {
#if defined(__DAV_VEC__)
  sum_fp32_features_impl<Out, In, Scratch>(output.data(), input.data(), scratch.data());
#endif
}

template <typename Out, typename In>
AICORE inline void round_fp32_sqrt(Out& output, In& input) {
#if defined(__DAV_VEC__)
  round_fp32_sqrt_impl<Out, In>(output.data(), output.data(), input.data());
#endif
}

// Normalize already rounded BF16 activations. Most quotients remain on the
// vector path. Only rows near a BF16 midpoint need exact FP32 RNE correction;
// rounding the exact quotient directly to BF16 would change FP32 double rounding.
template <typename Data, typename Denominator>
AICORE inline void round_bf16_midpoints(Data& output,
                                       Data& numerator,
                                       Denominator& denominator,
                                       Denominator& variance) {
#if defined(__DAV_VEC__)
  constexpr int R = Data::ValidRow;
  constexpr int MaskCols = R * 16 < 32 ? 32 : R * 16;
  constexpr int Arena = R <= 16 ? 65536 : 131072;
  constexpr int VectorBytes = R <= 16 ? 8192 : R * 128 * sizeof(float);
  // After TDIV, BF16/FP16 conversion storage and the norm broadcast are dead.
  // R64 reuses them for the two integer masks; cached weights and root/variance
  // remain disjoint. PIPE_V drains before reuse, PIPE_S drains before BF16 output.
  constexpr int Remainder = R == 64 ? ConvRowsLayout<64>::kBf : Arena;
  constexpr int Distance = R == 64 ? ConvRowsLayout<64>::kNorm : Arena + VectorBytes;
  constexpr int MaskOffset = R == 64 ? ConvRowsLayout<64>::kWindow : Arena + 2 * VectorBytes;
  static_assert(R <= 64 && MaskOffset + MaskCols <= 192 * 1024);

  using I = pto::Tile<pto::TileType::Vec, int32_t, R, 128,
                      pto::BLayout::RowMajor, R, 128>;
  using F = pto::Tile<pto::TileType::Vec, float, R, 128,
                      pto::BLayout::RowMajor, R, 128>;
  using Flat = pto::Tile<pto::TileType::Vec, float, 1, R * 128,
                         pto::BLayout::RowMajor, 1, R * 128>;
  using M = pto::Tile<pto::TileType::Vec, uint8_t, 1, MaskCols,
                      pto::BLayout::RowMajor, 1, R * 16>;
  I bits, remainder, distance;
  F distance_float;
  M mask;
  Flat flat_distance;
  // R16/R32 arenas start beyond their frontend's disjoint live buffers.
  pto::TASSIGN(bits, reinterpret_cast<uintptr_t>(output.data()));
  pto::TASSIGN(remainder, Remainder);
  pto::TASSIGN(distance, Distance);
  pto::TASSIGN(distance_float, Remainder);
  pto::TASSIGN(mask, MaskOffset);
  pto::TASSIGN(flat_distance, Remainder);
  pto::TANDS(remainder, bits, int32_t{0xffff});
  pipe_barrier(PIPE_V);
  pto::TADDS(distance, remainder, int32_t{-32768});
  pipe_barrier(PIPE_V);
  pto::TCVT(distance_float, distance, pto::RoundMode::CAST_NONE);
  pipe_barrier(PIPE_V);
  pto::TABS(distance_float, distance_float);
  pipe_barrier(PIPE_V);
  pto::TCMPS(mask, flat_distance, 64.0f, pto::CmpMode::LE);
  pipe_barrier(PIPE_V);
  round_bf16_midpoints_impl<Data, Denominator, M>(
      output.data(), output.data(), numerator.data(), denominator.data(),
      variance.data(), mask.data());
#endif
}

}  // namespace kda_prefill
