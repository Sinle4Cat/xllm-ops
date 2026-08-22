#!/usr/bin/env python3
"""Comprehensive A2 precision evaluation for MegaGdnDraftDecode."""

from dataclasses import dataclass
from pathlib import Path
import sys

import pytest
import torch


PYTHON_TEST_DIR = Path(__file__).resolve().parents[3] / "test" / "python_test"
sys.path.insert(0, str(PYTHON_TEST_DIR))

import test_mega_gdn_draft_decode as base  # noqa: E402


# This fused operator has a fixed mixed profile: BF16 activations and gate
# hand-offs with an FP32 resident SSM State. The BF16 ecosystem threshold is
# therefore the end-to-end contract for all four observable outputs.
PROFILE_NAME = "BF16 activations/Conv State + FP32 SSM State/gate parameters"
THRESHOLD = 2**-7


@dataclass(frozen=True)
class PrecisionCase:
    category: str
    name: str
    sequence_lengths: tuple[int, ...]
    num_k_heads: int = 1
    num_v_heads: int = 2
    prefix_fork: bool = False
    fla_ssm_state_layout: bool = True
    state_validity: tuple[bool, ...] | None = None
    boundary_mode: str | None = None


TYPICAL_CASES = (
    PrecisionCase("Typical", "b1_s1_same_fla", (1,)),
    PrecisionCase("Typical", "b2_s1_same_fla", (1, 1)),
    PrecisionCase("Typical", "b4_s1_same_fla", (1, 1, 1, 1)),
    PrecisionCase("Typical", "b8_s1_same_fla", (1,) * 8),
    PrecisionCase("Typical", "b1_s2_same_fla", (2,)),
    PrecisionCase("Typical", "b2_s2_same_fla", (2, 2)),
    PrecisionCase("Typical", "b4_mixed_same_fla", (1, 2, 1, 2)),
    PrecisionCase(
        "Typical", "b8_mixed_same_fla", (2, 1, 2, 1, 1, 2, 1, 2)
    ),
    PrecisionCase("Typical", "b1_s1_prefix_fla", (1,), prefix_fork=True),
    PrecisionCase("Typical", "b2_s2_prefix_fla", (2, 2), prefix_fork=True),
    PrecisionCase(
        "Typical", "b4_mixed_prefix_fla", (1, 2, 1, 2), prefix_fork=True
    ),
    PrecisionCase(
        "Typical",
        "b8_mixed_prefix_fla",
        (2, 1, 2, 1, 1, 2, 1, 2),
        prefix_fork=True,
    ),
    PrecisionCase(
        "Typical", "b1_s1_same_nonfla", (1,), fla_ssm_state_layout=False
    ),
    PrecisionCase(
        "Typical", "b2_s2_same_nonfla", (2, 2), fla_ssm_state_layout=False
    ),
    PrecisionCase(
        "Typical",
        "b4_mixed_prefix_nonfla",
        (1, 2, 1, 2),
        prefix_fork=True,
        fla_ssm_state_layout=False,
    ),
    PrecisionCase(
        "Typical",
        "b8_mixed_prefix_nonfla",
        (2, 1, 2, 1, 1, 2, 1, 2),
        prefix_fork=True,
        fla_ssm_state_layout=False,
    ),
)


GENERAL_CASES = (
    PrecisionCase("General", "heads_1x1", (1,), 1, 1),
    PrecisionCase("General", "heads_1x4", (2,), 1, 4),
    PrecisionCase("General", "heads_2x2", (1, 2), 2, 2),
    PrecisionCase("General", "heads_2x8", (2, 1), 2, 8),
    PrecisionCase("General", "heads_4x8", (1,), 4, 8),
    PrecisionCase("General", "heads_4x16", (2,), 4, 16),
    PrecisionCase("General", "heads_8x8", (1,), 8, 8),
    PrecisionCase("General", "heads_8x32", (2,), 8, 32),
    PrecisionCase("General", "heads_16x16", (1,), 16, 16),
    PrecisionCase("General", "heads_16x64", (2,), 16, 64),
    PrecisionCase("General", "b16_alternating", (1, 2) * 8),
    PrecisionCase("General", "b32_s1", (1,) * 32),
    PrecisionCase(
        "General", "b32_alternating", (1, 2) * 16, prefix_fork=True
    ),
    PrecisionCase(
        "General", "fresh_all", (2, 1), state_validity=(False, False)
    ),
    PrecisionCase(
        "General",
        "fresh_mixed",
        (1, 2, 2, 1),
        state_validity=(True, False, True, False),
    ),
    PrecisionCase(
        "General",
        "fresh_prefix_nonfla",
        (2, 1),
        prefix_fork=True,
        fla_ssm_state_layout=False,
        state_validity=(False, True),
    ),
)


BOUNDARY_CASES = (
    PrecisionCase(
        "Boundary",
        "all_zero_fresh_state",
        (2, 1),
        state_validity=(False, False),
        boundary_mode="zero",
    ),
    PrecisionCase(
        "Boundary", "near_zero_qk_norm", (2,), boundary_mode="near_zero"
    ),
    PrecisionCase(
        "Boundary", "beta_approaches_one", (2,), boundary_mode="beta_one"
    ),
    PrecisionCase(
        "Boundary", "beta_approaches_zero", (2,), boundary_mode="beta_zero"
    ),
    PrecisionCase(
        "Boundary", "softplus_positive_region", (2,), boundary_mode="softplus_pos"
    ),
    PrecisionCase(
        "Boundary", "softplus_negative_region", (2,), boundary_mode="softplus_neg"
    ),
    PrecisionCase("Boundary", "same_read_write_slot", (2, 2)),
    PrecisionCase(
        "Boundary", "shared_prefix_private_writes", (2, 1, 2), prefix_fork=True
    ),
)


ALL_CASES = TYPICAL_CASES + GENERAL_CASES + BOUNDARY_CASES


def _prepare_inputs(case: PrecisionCase) -> dict[str, torch.Tensor]:
    inputs = base._make_inputs(
        case.sequence_lengths,
        num_k_heads=case.num_k_heads,
        num_v_heads=case.num_v_heads,
        prefix_fork=case.prefix_fork,
        state_validity=case.state_validity,
    )

    if case.boundary_mode == "zero":
        for name in ("qkv", "z", "a", "b", "conv_state", "ssm_state"):
            inputs[name].zero_()
    elif case.boundary_mode == "near_zero":
        inputs["qkv"].fill_(1e-5)
        inputs["conv_state"].zero_()
        inputs["ssm_state"].zero_()
    elif case.boundary_mode == "beta_one":
        inputs["b"].fill_(12.0)
    elif case.boundary_mode == "beta_zero":
        inputs["b"].fill_(-12.0)
    elif case.boundary_mode == "softplus_pos":
        inputs["a"].fill_(8.0)
        inputs["dt_bias"].fill_(2.0)
    elif case.boundary_mode == "softplus_neg":
        inputs["a"].fill_(-8.0)
        inputs["dt_bias"].fill_(-2.0)

    return inputs


def _compute_metrics(
    actual: tuple[torch.Tensor, ...], expected: base.DraftResult
) -> dict[str, float]:
    expected_tensors = (
        expected.conv_out,
        expected.conv_state,
        expected.ssm_state,
        expected.out,
    )
    total_numel = 0
    abs_sum = 0.0
    rel_sum = 0.0
    dot = 0.0
    actual_sq = 0.0
    expected_sq = 0.0
    max_abs = 0.0
    max_rel = 0.0

    for actual_tensor, expected_tensor in zip(actual, expected_tensors):
        actual_f = actual_tensor.float()
        expected_f = expected_tensor.float()
        abs_err = (actual_f - expected_f).abs()
        rel_err = abs_err / (expected_f.abs() + 1e-7)
        total_numel += actual_f.numel()
        abs_sum += abs_err.sum().item()
        rel_sum += rel_err.sum().item()
        max_abs = max(max_abs, abs_err.max().item())
        max_rel = max(max_rel, rel_err.max().item())
        dot += (actual_f * expected_f).sum().item()
        actual_sq += actual_f.square().sum().item()
        expected_sq += expected_f.square().sum().item()

    if actual_sq == 0.0 and expected_sq == 0.0:
        cosine = 1.0
    elif actual_sq == 0.0 or expected_sq == 0.0:
        cosine = 0.0
    else:
        cosine = max(-1.0, min(1.0, dot / (actual_sq * expected_sq) ** 0.5))

    return {
        "max_abs_err": max_abs,
        "mean_abs_err": abs_sum / total_numel,
        "MARE": max_rel,
        "MERE": rel_sum / total_numel,
        "cosine_sim": cosine,
    }


def run_precision_case(case: PrecisionCase) -> dict[str, object]:
    inputs = _prepare_inputs(case)
    expected = base._reference(
        inputs, fla_ssm_state_layout=case.fla_ssm_state_layout
    )
    npu_inputs = {name: tensor.to("npu:0") for name, tensor in inputs.items()}
    actual_npu = base.custom_ops.mega_gdn_draft_decode(
        **npu_inputs,
        fla_ssm_state_layout=case.fla_ssm_state_layout,
    )
    torch.npu.synchronize()
    actual = tuple(tensor.cpu() for tensor in actual_npu)

    metrics = _compute_metrics(actual, expected)
    contract_passed = (
        torch.allclose(actual[0], expected.conv_out, rtol=0.03, atol=0.003)
        and torch.equal(actual[1], expected.conv_state)
        and torch.allclose(actual[2], expected.ssm_state, rtol=0.003, atol=0.0003)
        and torch.allclose(actual[3], expected.out, rtol=0.05, atol=0.005)
    )
    if case.prefix_fork:
        contract_passed = (
            contract_passed
            and torch.equal(actual[1][0], inputs["conv_state"][0])
            and torch.equal(actual[2][0], inputs["ssm_state"][0])
        )
    passed = (
        contract_passed
        and metrics["MERE"] < THRESHOLD
        and metrics["MARE"] < 10 * THRESHOLD
    )
    return {
        "category": case.category,
        "name": case.name,
        "sequence_lengths": list(case.sequence_lengths),
        "batch_size": len(case.sequence_lengths),
        "total_tokens": sum(case.sequence_lengths),
        "num_k_heads": case.num_k_heads,
        "num_v_heads": case.num_v_heads,
        "prefix_fork": case.prefix_fork,
        "fla_ssm_state_layout": case.fla_ssm_state_layout,
        "dtype_profile": PROFILE_NAME,
        "threshold": THRESHOLD,
        "contract_passed": contract_passed,
        **metrics,
        "passed": passed,
    }


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.name)
def test_precision(case: PrecisionCase) -> None:
    result = run_precision_case(case)
    assert result["passed"], (
        f"{case.name}: MERE={result['MERE']:.3e} (limit {THRESHOLD:.3e}), "
        f"MARE={result['MARE']:.3e} (limit {10 * THRESHOLD:.3e}), "
        f"MaxAbsErr={result['max_abs_err']:.3e}"
    )
