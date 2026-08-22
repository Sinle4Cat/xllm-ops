# Copyright 2026 The xLLM Authors. All Rights Reserved.

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops_lib")


HEAD_DIM = 128


@dataclass(frozen=True)
class DraftResult:
    conv_out: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    out: torch.Tensor


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _make_inputs(
    sequence_lengths: tuple[int, ...],
    *,
    num_k_heads: int = 1,
    num_v_heads: int = 2,
    prefix_fork: bool = False,
    state_validity: tuple[bool, ...] | None = None,
) -> dict[str, torch.Tensor]:
    batch_size = len(sequence_lengths)
    total_tokens = sum(sequence_lengths)
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    num_state_slots = batch_size + 1 if prefix_fork else batch_size
    generator = torch.Generator().manual_seed(
        20260822 + 10 * batch_size + total_tokens
    )

    def rand_bfloat16(*shape: int, scale: float = 0.1) -> torch.Tensor:
        return (
            torch.randn(*shape, generator=generator, dtype=torch.float32)
            .mul_(scale)
            .to(torch.bfloat16)
        )

    if prefix_fork:
        read_state_indices = torch.zeros(batch_size, dtype=torch.int32)
        write_state_indices = torch.arange(
            1, batch_size + 1, dtype=torch.int32
        )
    else:
        read_state_indices = torch.arange(batch_size, dtype=torch.int32)
        write_state_indices = read_state_indices.clone()
    q_cu_seq_lens = torch.tensor(
        (0, *torch.tensor(sequence_lengths).cumsum(0).tolist()),
        dtype=torch.int32,
    )
    if state_validity is None:
        state_validity = (True,) * batch_size

    return {
        "qkv": rand_bfloat16(total_tokens, conv_dim),
        "z": rand_bfloat16(total_tokens, num_v_heads, HEAD_DIM),
        "b": rand_bfloat16(total_tokens, num_v_heads, scale=0.3),
        "a": rand_bfloat16(total_tokens, num_v_heads, scale=0.2),
        "conv_weight": rand_bfloat16(4, conv_dim, scale=0.15),
        "conv_state": rand_bfloat16(num_state_slots, 3, conv_dim),
        "a_log": torch.full((num_v_heads,), -1.0, dtype=torch.float32),
        "dt_bias": torch.zeros(num_v_heads, dtype=torch.float32),
        "ssm_state": torch.randn(
            num_state_slots,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.02),
        "read_state_indices": read_state_indices,
        "write_state_indices": write_state_indices,
        "q_cu_seq_lens": q_cu_seq_lens,
        "state_validity_mask": torch.tensor(state_validity, dtype=torch.bool),
        "norm_weight": 1.0 + rand_bfloat16(HEAD_DIM, scale=0.05),
    }


def _reference(
    inputs: dict[str, torch.Tensor],
    *,
    fla_ssm_state_layout: bool,
) -> DraftResult:
    qkv = inputs["qkv"]
    z = inputs["z"]
    b = inputs["b"]
    a = inputs["a"]
    conv_weight = inputs["conv_weight"]
    conv_state = inputs["conv_state"]
    a_log = inputs["a_log"]
    dt_bias = inputs["dt_bias"]
    ssm_state = inputs["ssm_state"]
    read_state_indices = inputs["read_state_indices"]
    write_state_indices = inputs["write_state_indices"]
    q_cu_seq_lens = inputs["q_cu_seq_lens"]
    state_validity_mask = inputs["state_validity_mask"]
    norm_weight = inputs["norm_weight"]

    total_tokens, conv_dim = qkv.shape
    num_v_heads = z.size(1)
    num_k_heads = (
        conv_dim - num_v_heads * HEAD_DIM
    ) // (2 * HEAD_DIM)
    repeats = num_v_heads // num_k_heads
    conv_out = torch.empty_like(qkv)
    out = torch.empty_like(z)
    conv_state_out = conv_state.clone()
    ssm_state_out = ssm_state.clone()
    conv_snapshots = conv_state.index_select(
        0, read_state_indices.to(torch.int64)
    ).clone()
    ssm_snapshots = ssm_state.index_select(
        0, read_state_indices.to(torch.int64)
    ).clone()

    for batch_idx in range(read_state_indices.numel()):
        token_begin = int(q_cu_seq_lens[batch_idx])
        token_end = int(q_cu_seq_lens[batch_idx + 1])
        write_slot = int(write_state_indices[batch_idx])
        has_initial_state = bool(state_validity_mask[batch_idx])
        history = (
            conv_snapshots[batch_idx].float()
            if has_initial_state
            else torch.zeros_like(conv_snapshots[batch_idx].float())
        )
        state_stored = (
            ssm_snapshots[batch_idx].float()
            if has_initial_state
            else torch.zeros_like(ssm_snapshots[batch_idx].float())
        )
        state = (
            state_stored
            if fla_ssm_state_layout
            else state_stored.transpose(-1, -2)
        )

        for token_idx in range(token_begin, token_end):
            token = qkv[token_idx].float()
            conv_acc = (
                (history * conv_weight[:3].float()).sum(dim=0)
                + token * conv_weight[3].float()
            )
            # Match CausalConvSilu and the BF16 tensor hand-off.
            conv_token = (
                conv_acc * torch.reciprocal(torch.exp(-conv_acc) + 1.0)
            ).to(torch.bfloat16)
            conv_out[token_idx] = conv_token
            history = torch.cat((history[1:], token.unsqueeze(0)), dim=0)

            q, k, v = torch.split(
                conv_token,
                [
                    num_k_heads * HEAD_DIM,
                    num_k_heads * HEAD_DIM,
                    num_v_heads * HEAD_DIM,
                ],
            )
            q = q.reshape(num_k_heads, HEAD_DIM).float()
            k = k.reshape(num_k_heads, HEAD_DIM).float()
            v = v.reshape(num_v_heads, HEAD_DIM).float()
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            q = q.repeat_interleave(repeats, dim=0) / HEAD_DIM**0.5
            k = k.repeat_interleave(repeats, dim=0)

            # Match the BF16 g and beta tensors produced by the small-op path.
            g = (
                -torch.exp(a_log)
                * F.softplus(a[token_idx].float() + dt_bias)
            ).to(torch.bfloat16).float()
            beta = torch.sigmoid(b[token_idx].float()).to(torch.bfloat16).float()
            state = state * torch.exp(g)[:, None, None]
            prediction = torch.einsum("hkv,hk->hv", state, k)
            delta = (v - prediction) * beta[:, None]
            state = state + torch.einsum("hk,hv->hkv", k, delta)
            readout = torch.einsum("hkv,hk->hv", state, q)

            # Match the BF16 recurrent-to-Norm tensor boundary.
            norm_input = readout.to(torch.bfloat16).float()
            rms_inv = torch.rsqrt(
                norm_input.square().mean(-1, keepdim=True) + 1e-6
            )
            norm_output = norm_input * rms_inv * norm_weight.float()
            norm_output = norm_output * F.silu(z[token_idx].float())
            out[token_idx] = norm_output.to(torch.bfloat16)

        if token_end > token_begin:
            conv_state_out[write_slot] = history.to(torch.bfloat16)
            ssm_state_out[write_slot] = (
                state
                if fla_ssm_state_layout
                else state.transpose(-1, -2)
            )

    return DraftResult(conv_out, conv_state_out, ssm_state_out, out)


def _run_case(
    sequence_lengths: tuple[int, ...],
    *,
    prefix_fork: bool,
    fla_ssm_state_layout: bool,
    state_validity: tuple[bool, ...] | None = None,
    num_k_heads: int = 1,
    num_v_heads: int = 2,
) -> None:
    inputs = _make_inputs(
        sequence_lengths,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        prefix_fork=prefix_fork,
        state_validity=state_validity,
    )
    expected = _reference(
        inputs,
        fla_ssm_state_layout=fla_ssm_state_layout,
    )
    npu_inputs = {
        name: tensor.to("npu:0") for name, tensor in inputs.items()
    }
    actual = custom_ops.mega_gdn_draft_decode(
        **npu_inputs,
        fla_ssm_state_layout=fla_ssm_state_layout,
    )
    torch.npu.synchronize()
    actual = tuple(tensor.cpu() for tensor in actual)

    torch.testing.assert_close(
        actual[0], expected.conv_out, rtol=0.03, atol=0.003
    )
    torch.testing.assert_close(
        actual[1], expected.conv_state, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual[2], expected.ssm_state, rtol=0.003, atol=0.0003
    )
    torch.testing.assert_close(actual[3], expected.out, rtol=0.05, atol=0.005)

    if prefix_fork:
        torch.testing.assert_close(
            actual[1][0], inputs["conv_state"][0], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            actual[2][0], inputs["ssm_state"][0], rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_one_token_sequences(batch_size: int) -> None:
    _run_case(
        (1,) * batch_size,
        prefix_fork=False,
        fla_ssm_state_layout=True,
    )


@pytest.mark.parametrize(
    ("sequence_lengths", "state_validity"),
    [
        ((2,), (True,)),
        ((2, 2), (True, False)),
        ((1, 2, 1, 2), (True, False, True, True)),
        ((2, 1, 2, 1, 1, 2, 1, 2), None),
    ],
)
@pytest.mark.parametrize("prefix_fork", [False, True])
@pytest.mark.parametrize("fla_ssm_state_layout", [True, False])
def test_mixed_draft_sequences(
    sequence_lengths: tuple[int, ...],
    state_validity: tuple[bool, ...] | None,
    prefix_fork: bool,
    fla_ssm_state_layout: bool,
) -> None:
    _run_case(
        sequence_lengths,
        prefix_fork=prefix_fork,
        fla_ssm_state_layout=fla_ssm_state_layout,
        state_validity=state_validity,
    )


def test_multiple_recurrent_tasks_per_aiv() -> None:
    """The next head must not overwrite State while MTE3 stores this head."""
    _run_case(
        (2,),
        prefix_fork=False,
        fla_ssm_state_layout=True,
        num_k_heads=16,
        num_v_heads=64,
    )
