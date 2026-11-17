# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Independent FP64 formula oracle and FP32 activation-stage checks.

The mathematical oracle preserves explicit FP16/BF16 activation boundaries, but
does not emulate device FP32 accumulation or reduction trees bit for bit.
"""
from typing import Any

import torch
from mega_kda_test_utils import round_bf16, silu_bf16


OUTPUTS = ("output", "conv_state_out", "ssm_state_out")
NATIVE, MTP, PUBLIC_COMPAT = 0, 1, 3


def reference(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return independent output pools, preserving padding and discarded writes."""
    if any(x.device.type != "cpu" for x in data.values() if isinstance(x, torch.Tensor)):
        raise ValueError("the oracle must run on CPU")
    mode, capacity = data["mode"], data["max_query_tokens"]
    accepted = data["num_accepted_tokens"]
    if not ((mode in (NATIVE, PUBLIC_COMPAT) and capacity == 1 and accepted is None) or
            (mode == MTP and 1 <= capacity <= 17 and accepted is not None)):
        raise ValueError("invalid route")
    qkv = data["qkv"]
    heads = data["gate"].shape[1]
    conv_in, ssm_in = data["conv_state_in"], data["ssm_state_in"]
    out, conv_out, ssm_out = (data[name].clone() for name in OUTPUTS)
    offsets = data["cu_seqlens"].tolist()
    if (offsets[0] != 0 or offsets[-1] != qkv.shape[0]
            or any(b < a for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("invalid offsets")
    if mode != MTP and offsets != list(range(len(offsets))):
        raise ValueError("ordinary decode requires unit offsets")
    expected_history = capacity + 2 if mode == MTP else 3
    if conv_in.shape[1] != expected_history or conv_out.shape[1] != expected_history:
        raise ValueError("invalid Conv history")
    conv_destinations, ssm_destinations = set(), set()
    for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        count = end - begin
        if count == 0:
            continue
        if count > capacity or (mode != MTP and (count != 1 or begin != row)):
            raise ValueError("invalid query length")
        naccepted = int(accepted[row]) if mode == MTP else 1
        if not 1 <= naccepted <= capacity:
            raise ValueError("invalid accepted count")
        cr, cw = int(data["conv_read_indices"][row]), int(data["conv_write_indices"][row])
        sr = int(data["ssm_read_indices"][row, naccepted - 1])
        if cr == -1 or sr == -1:
            continue
        if not (0 <= cr < len(conv_in) and 0 <= sr < len(ssm_in) and -1 <= cw < len(conv_out)):
            raise ValueError("invalid state slot")
        if cw >= 0:
            if cw in conv_destinations:
                raise ValueError("duplicate Conv destination")
            conv_destinations.add(cw)
        writes = data["ssm_write_indices"][row, :count].tolist()
        for sw in writes:
            if not -1 <= sw < len(ssm_out):
                raise ValueError("invalid SSM destination")
            if sw >= 0:
                if sw in ssm_destinations:
                    raise ValueError("duplicate SSM destination")
                ssm_destinations.add(sw)
        history = conv_in[cr, naccepted - 1:naccepted + 2]
        sequence = torch.cat((history, qkv[begin:end]))
        state = ssm_in[sr].double().clone()
        for i, token in enumerate(range(begin, end)):
            x = sequence[i:i + 4]
            x = (x if mode == PUBLIC_COMPAT else x.half()).double()
            acc = torch.zeros_like(x[0]) if data["conv_bias"] is None else data["conv_bias"].double().clone()
            for tap in range(4):
                acc = acc + x[tap] * data["conv_weight"][tap].double()
            # All modes round Conv accumulation before SiLU, and SiLU before KDA.
            conv = silu_bf16(round_bf16(acc).double()).double().view(3, heads, 128)
            q, k, v = conv.unbind(0)
            if mode == PUBLIC_COMPAT:
                # Replacing a zero denominator before division avoids masked 0/0.
                q_norm = torch.sqrt((q * q).sum(-1, keepdim=True))
                k_norm = torch.sqrt((k * k).sum(-1, keepdim=True))
                q = q / torch.where(q_norm > 0, q_norm, torch.ones_like(q_norm)) * 128 ** -0.5
                k = k / torch.where(k_norm > 0, k_norm, torch.ones_like(k_norm))
            else:
                q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
                k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            raw_g = data["gate"][token].double() + data["gate_bias"].double()
            decay = (-5.0 * torch.sigmoid(raw_g * data["a_log"].double().exp()[:, None])).exp()
            beta = round_bf16(torch.sigmoid(data["beta"][token].double())).double()
            state = state * decay[:, None, :]
            delta = (v - (state * k[:, None, :]).sum(-1)) * beta[:, None]
            state = state + delta[:, :, None] * k[:, None, :]
            out[token] = round_bf16((state * q[:, None, :]).sum(-1))
            if writes[i] >= 0:
                ssm_out[writes[i]] = state
        if cw >= 0:
            conv_out[cw, :count + 2] = sequence[1:]
    return out, conv_out, ssm_out


def public_compat_stages(data: dict[str, Any], sequence: torch.Tensor,
                         token: int = 0) -> dict[str, torch.Tensor]:
    """Evaluate FP32 frontend boundaries, independently of the FP64 recurrence.

    This checks operation order and zero handling; Torch's CPU reductions and exp
    are not an instruction-level model of the Ascend implementation.
    """
    heads = data["gate"].shape[1]
    accumulation = (torch.zeros_like(sequence[0], dtype=torch.float32)
                    if data["conv_bias"] is None else data["conv_bias"].float().clone())
    for tap, weight in zip(sequence.float().unbind(0), data["conv_weight"].float().unbind(0)):
        accumulation = accumulation + tap * weight
    pre_silu = accumulation.bfloat16().float()
    activation = (pre_silu / (1.0 + torch.exp(-pre_silu))).bfloat16().float()
    q, k, v = activation.reshape(3, heads, 128).unbind(0)
    normalized = []
    for vector in (q, k):
        length = torch.sqrt(vector.square().sum(dim=-1, keepdim=True))
        reciprocal = 1.0 / torch.where(length > 0, length, torch.ones_like(length))
        normalized.append(vector * reciprocal)
    scaled_gate = (data["gate"][token].float() + data["gate_bias"].float()) * data["a_log"].float().exp()[:, None]
    sigmoid = 1.0 / (1.0 + torch.exp(-scaled_gate))
    gate = sigmoid * -5.0
    beta = torch.sigmoid(data["beta"][token].float()).bfloat16().float()
    return {"pre_silu": pre_silu, "conv": activation, "q": normalized[0] * 128 ** -0.5,
            "k": normalized[1], "v": v, "gate": gate, "decay": gate.exp(), "beta": beta}


def make_case(batch: int = 2, heads: int = 2, capacity: int = 1, mode: int = NATIVE,
              *, bias: bool = False, ragged: bool = False,
              padding: bool = False) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(7301)
    lengths = [capacity] * batch
    if ragged and capacity > 1:
        lengths[0] = 1
    if ragged and mode == MTP and batch > 2:
        lengths[-1] = 0
    offsets = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
    tokens, channels = int(offsets[-1]), heads * 384
    history = capacity + 2 if mode == MTP else 3
    slots = batch * capacity + 3

    def rand(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.1).to(dtype)

    data = dict(
        qkv=rand((tokens, channels)), gate=rand((tokens, heads, 128)), beta=rand((tokens, heads)),
        a_log=torch.full((heads,), -2.0), gate_bias=rand((heads, 128), torch.float32),
        conv_weight=rand((4, channels)), conv_bias=rand((channels,)) if bias else None,
        conv_state_in=rand((batch + 3, history, channels)),
        ssm_state_in=rand((slots, heads, 128, 128), torch.float32),
        cu_seqlens=offsets, conv_read_indices=torch.arange(batch, dtype=torch.int32),
        conv_write_indices=torch.arange(batch - 1, -1, -1, dtype=torch.int32) + 1,
        ssm_read_indices=torch.arange(batch * capacity, dtype=torch.int32).view(batch, capacity),
        ssm_write_indices=torch.arange(batch * capacity, dtype=torch.int32).view(batch, capacity) + 2,
        num_accepted_tokens=torch.tensor([capacity] + [1] * (batch - 1), dtype=torch.int32) if mode == MTP else None,
        mode=mode, max_query_tokens=capacity,
        output=torch.full((tokens, heads, 128), 37.0, dtype=torch.bfloat16),
        conv_state_out=torch.full((batch + 5, history, channels), 37.0, dtype=torch.bfloat16),
        ssm_state_out=torch.full((slots + 2, heads, 128, 128), 37.0),
    )
    if padding:
        data["conv_read_indices"][-1] = -1
    return data
