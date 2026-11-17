# Copyright 2026 The xLLM Authors. All Rights Reserved.
import torch

from mega_kda_test_utils import round_bf16, silu_bf16


def test_all_finite_bf16_midpoints_and_neighbors():
    # Enumerate positive adjacent BF16 bit patterns. The midpoint and its
    # adjacent binary64 numbers distinguish direct RNE from double rounding.
    bits = torch.arange(0, 0x7f7f, dtype=torch.int32)
    lower = bits.to(torch.int16).view(torch.bfloat16).double()
    upper = (bits + 1).to(torch.int16).view(torch.bfloat16).double()
    midpoint = (lower + upper) / 2
    for sign in (1., -1.):
        middle = midpoint * sign
        below = torch.nextafter(middle, torch.full_like(middle, -float('inf')))
        above = torch.nextafter(middle, torch.full_like(middle, float('inf')))
        even = torch.where(bits % 2 == 0, lower, upper) * sign
        torch.testing.assert_close(round_bf16(middle).double(), even, rtol=0, atol=0)
        torch.testing.assert_close(
            round_bf16(below).double(), lower if sign > 0 else -upper, rtol=0, atol=0
        )
        torch.testing.assert_close(
            round_bf16(above).double(), upper if sign > 0 else -lower, rtol=0, atol=0
        )


def test_silu_double_rounding_regressions():
    raw = torch.tensor([2.19281005859375, 1.6744384765625], dtype=torch.float64)
    expected = torch.tensor([1.9765625, 1.4140625], dtype=torch.bfloat16)
    actual = round_bf16(torch.nn.functional.silu(raw))
    assert torch.equal(actual, expected)


def test_silu_uses_declared_fp64_compute_precision():
    raw = torch.tensor([37.375, 80.75], dtype=torch.float64)
    # FP64 SiLU has already rounded to these midpoints. The reference must
    # preserve FP64 semantics rather than introduce extended-precision math.
    expected = torch.tensor([37.5, 81.], dtype=torch.bfloat16)
    assert torch.equal(silu_bf16(raw), expected)
