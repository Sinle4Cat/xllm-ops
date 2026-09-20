"""Native KV bank contract: arbitrary checkpoints, strides and graph replay."""

from types import SimpleNamespace

import custom_ops_lib as compact
import native_ops_lib as native
import pytest
import torch
import torch_npu  # noqa: F401


def make_case(lengths=(4, 2, 1, 0), heads=(4, 12), width=4, strided=True):
    torch.manual_seed(20260919)
    hk, hv = heads
    batch, slots, d = len(lengths), len(lengths) * (width + 3) + 9, 128
    c, tokens = (2 * hk + hv) * d, max(sum(lengths), 4)

    def rand(shape, scale=0.1):
        return (torch.randn(shape) * scale).bfloat16().npu()

    combined = rand((tokens, c + hv * d))
    qkv, flat_z = combined.split((c, hv * d), -1)
    z = flat_z.reshape(tokens, hv, d)
    if not strided:
        qkv, z = qkv.contiguous(), z.contiguous()
    conv_back = rand((slots, (width + 2) * c + (64 if strided else 0)))
    conv = conv_back[:, : (width + 2) * c].view(slots, width + 2, c)
    ssm_back = torch.randn(slots, hv * d * d + (64 if strided else 0)).npu() * 0.02
    ssm = ssm_back[:, : hv * d * d].view(slots, hv, d, d)
    perm = torch.randperm(slots - 1)[: batch * width] + 1
    ids = perm.reshape(batch, width).int().npu()
    accepted = (torch.arange(batch) % width + 1).int().npu()
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    ).npu()
    return SimpleNamespace(
        qkv=qkv,
        z=z,
        b=rand((tokens, hv), 1),
        a=rand((tokens, hv), 1),
        weights=rand((4, c)),
        conv=conv,
        ssm=ssm,
        conv_back=conv_back,
        ssm_back=ssm_back,
        ids=ids,
        starts=starts,
        accepted=accepted,
        a_log=torch.zeros(hv).npu(),
        dt_bias=torch.zeros(hv).npu(),
        norm=torch.ones(d, dtype=torch.bfloat16).npu(),
    )


def call(x):
    return native.mega_gdn_native_decode(
        x.qkv,
        x.z,
        x.b,
        x.a,
        x.weights,
        x.conv,
        x.a_log,
        x.dt_bias,
        x.ssm,
        x.ids,
        x.starts,
        x.accepted,
        x.norm,
    )


def model_reference(x):
    """Independent PyTorch model math: BF16 normalized Q/K and FP32 g.

    The legacy MegaMTP API has different rounding boundaries and cannot
    serve as an oracle for the framework-compatible native API.
    """
    import torch.nn.functional as F

    c = {k: v.cpu() for k, v in vars(x).items() if isinstance(v, torch.Tensor)}
    state, conv = c["ssm"].clone(), c["conv"].clone()
    out = torch.zeros_like(c["z"])
    hv, d = c["z"].shape[1:]
    hk = (c["qkv"].size(1) // d - hv) // 2
    for row in range(c["ids"].size(0)):
        lo, hi = map(int, c["starts"][row : row + 2])
        if hi == lo or int(c["ids"][row, 0]) <= 0:
            continue
        accepted = int(c["accepted"][row])
        base = int(c["ids"][row, 0])
        history = conv[base, accepted - 1 : accepted + 2].float().clone()
        state_row = state[int(c["ids"][row, accepted - 1])].clone()
        conv[base, :2] = history[1:].bfloat16()
        for token in range(lo, hi):
            inp = c["qkv"][token].float()
            convolved = F.silu(
                (history * c["weights"][:3].float()).sum(0)
                + inp * c["weights"][3].float()
            ).bfloat16()
            history = torch.cat((history[1:], inp.unsqueeze(0)))
            conv[base, token - lo + 2] = inp.bfloat16()
            q, k, v = convolved.split((hk * d, hk * d, hv * d))
            q, k, v = (
                q.view(hk, d).float(),
                k.view(hk, d).float(),
                v.view(hv, d).float(),
            )
            q = (
                q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            ).bfloat16().float().repeat_interleave(hv // hk, 0) * d**-0.5
            k = (
                (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6))
                .bfloat16()
                .float()
                .repeat_interleave(hv // hk, 0)
            )
            g = -c["a_log"].exp() * F.softplus(c["a"][token].float() + c["dt_bias"])
            beta = c["b"][token].float().sigmoid().bfloat16().float()
            state_row *= g.exp()[:, None, None]
            prediction = (state_row * k[:, :, None]).sum(1)
            state_row += k[:, :, None] * ((v - prediction) * beta[:, None])[:, None, :]
            state[int(c["ids"][row, token - lo])] = state_row
            value = (state_row * q[:, :, None]).sum(1).bfloat16().float()
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            out[token] = (
                value * c["norm"].float() * F.silu(c["z"][token].float())
            ).bfloat16()
    return out.to(x.z.device), conv.to(x.conv.device), state.to(x.ssm.device)


@pytest.mark.parametrize(
    "lengths,width", [((1, 1, 1, 0), 1), ((4, 4, 4, 4), 4), ((4, 2, 1, 0), 4)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_native_matches_model_math(lengths, width, strided):
    x = make_case(lengths, width=width, strided=strided)
    before_c, before_s = x.conv_back.clone(), x.ssm_back.clone()
    expected, conv, ssm = model_reference(x)
    actual = call(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-4)
    torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)
    if strided:
        torch.testing.assert_close(
            x.conv_back[:, -64:], before_c[:, -64:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            x.ssm_back[:, -64:], before_s[:, -64:], rtol=0, atol=0
        )


def test_native_graph_replays_ragged_ids_accepted_and_request_counts():
    x = make_case((4, 0, 0, 0))
    original_c, original_s = x.conv_back.clone(), x.ssm_back.clone()
    for _ in range(3):
        call(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = call(x)
    for lengths in [(1, 1, 1, 1), (2, 1, 1, 0), (4, 0, 0, 0), (0, 0, 0, 0)]:
        x.starts.copy_(
            torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
            ).npu()
        )
        x.ids.copy_(x.ids.roll(1, 0).roll(1, 1))
        x.accepted.copy_(x.accepted.roll(1))
        x.conv_back.copy_(original_c)
        x.ssm_back.copy_(original_s)
        expected, conv, ssm = model_reference(x)
        for _ in range(10):
            x.conv_back.copy_(original_c)
            x.ssm_back.copy_(original_s)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(result, expected, rtol=0.01, atol=1e-4)
            torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
            torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)


def test_native_invalid_rows_preserve_banks():
    x = make_case((4, 2, 1, 0))
    x.ids[0, 0] = 0
    x.accepted[1] = 0
    x.ids[2, 2] = x.ssm.size(0)  # accepted=3 selects an invalid checkpoint
    conv, ssm = x.conv_back.clone(), x.ssm_back.clone()
    out = call(x)
    torch.npu.synchronize()
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(x.conv_back, conv, rtol=0, atol=0)
    torch.testing.assert_close(x.ssm_back, ssm, rtol=0, atol=0)


def prefill_case(length=128, initial=True, inplace=True):
    x = make_case((length, length + 3), width=4)
    write = x.ids[:, 0].contiguous()
    read = write if inplace else x.ids[:, 1].contiguous()
    if not initial:
        read = torch.full_like(write, -1)
    lower = torch.tril(torch.ones(128, 128, device=x.qkv.device), diagonal=-1)
    full = torch.tril(torch.ones_like(lower))
    minus = -torch.eye(128, dtype=torch.bfloat16, device=x.qkv.device)
    matrices = ((length + 127) // 128 + (length + 3 + 127) // 128) * x.z.size(1)
    return x, read, write, lower, full, minus, matrices


def prefill_call(case):
    x, read, write, lower, full, minus, matrices = case
    return native.mega_gdn_native_prefill(
        x.qkv,
        x.b,
        x.a,
        x.z,
        x.weights,
        x.conv,
        x.a_log,
        x.dt_bias,
        read,
        write,
        read,
        write,
        x.ssm,
        lower,
        full,
        minus,
        x.starts,
        x.norm,
        matrices,
    )


@pytest.mark.parametrize("length", [3, 128, 512, 801, 8192])
@pytest.mark.parametrize(
    "initial,inplace", [(False, True), (True, True), (True, False)]
)
def test_native_prefill_minimal_history_and_strided_banks(length, initial, inplace):
    case = prefill_case(length, initial, inplace)
    x, read, write, lower, full, minus, matrices = case
    original_c, original_s = x.conv_back.clone(), x.ssm_back.clone()
    expected_c, expected_s = x.conv.clone(), x.ssm.clone()
    # Materialize only the oracle's dense/minimal-history bank.
    compact_c, compact_s = x.conv[:, :3].contiguous(), x.ssm.contiguous()
    expected = compact.mega_gdn_prefill_op(
        x.qkv,
        x.b,
        x.a,
        x.z,
        x.weights,
        compact_c,
        x.a_log,
        x.dt_bias,
        read,
        write,
        read,
        write,
        compact_s,
        lower,
        full,
        minus,
        x.starts,
        x.norm,
        matrices,
    )
    expected_c[write.long(), :3] = compact_c[write.long()]
    expected_s[write.long()] = compact_s[write.long()]
    actual = prefill_call(case)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The public cache stays FP32. The validated ComputeT path writes a
    # ComputeT result promoted to FP32, so its FP16 projection must remain
    # exactly compatible with the compact reference boundary.
    assert x.ssm.dtype == torch.float32
    assert torch.isfinite(x.ssm).all()
    torch.testing.assert_close(x.conv, expected_c, rtol=0, atol=0)
    written_slots = torch.zeros(
        x.ssm.size(0), dtype=torch.bool, device=x.ssm.device
    )
    written_slots[write.long()] = True
    torch.testing.assert_close(
        x.ssm[~written_slots], expected_s[~written_slots], rtol=0, atol=0
    )
    torch.testing.assert_close(
        x.ssm[written_slots].to(torch.float16),
        expected_s[written_slots].to(torch.float16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        x.conv_back[:, -64:], original_c[:, -64:], rtol=0, atol=0
    )
    torch.testing.assert_close(x.ssm_back[:, -64:], original_s[:, -64:], rtol=0, atol=0)


def test_native_prefill_then_mtp_accepted_checkpoint():
    case = prefill_case(128)
    x = case[0]
    prefill_call(case)
    torch.npu.synchronize()
    # Each row's first checkpoint now holds the prefill final state. Follow it
    # with a 4-token verification, then replay using an intermediate acceptance.
    x.qkv = x.qkv[:8]
    x.z = x.z[:8]
    x.a = x.a[:8]
    x.b = x.b[:8]
    x.starts.copy_(torch.tensor([0, 4, 8], dtype=torch.int32).npu())
    x.accepted.fill_(1)
    for accepted in (1, 2, 4, 3):
        x.accepted.fill_(accepted)
        expected, conv, ssm = model_reference(x)
        result = call(x)
        torch.npu.synchronize()
        torch.testing.assert_close(result, expected, rtol=0.01, atol=1e-4)
        torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
        torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "lengths,width", [((2, 1, 0, 2), 2), ((17, 8, 1, 0), 17), ((4,) * 32, 4)]
)
def test_native_capacity_and_sequence_boundaries(lengths, width):
    x = make_case(lengths, width=width)
    expected, conv, ssm = model_reference(x)
    actual = call(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-4)
    torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)


def test_native_cpu_golden_all_checkpoints():
    import torch.nn.functional as F

    x = make_case()
    c = {k: v.cpu() for k, v in vars(x).items() if isinstance(v, torch.Tensor)}
    state, conv = c["ssm"].clone(), c["conv"].clone()
    out = torch.zeros_like(c["z"])
    hv, d = c["z"].shape[1:]
    hk = (c["qkv"].size(1) // d - hv) // 2
    for row in range(c["ids"].size(0)):
        lo, hi = map(int, c["starts"][row : row + 2])
        if hi == lo:
            continue
        accepted = int(c["accepted"][row])
        base = int(c["ids"][row, 0])
        history = conv[base, accepted - 1 : accepted + 2].float().clone()
        state_row = state[int(c["ids"][row, accepted - 1])].clone()
        conv[base, :2] = history[1:].bfloat16()
        for token in range(lo, hi):
            inp = c["qkv"][token].float()
            convolved = F.silu(
                (history * c["weights"][:3].float()).sum(0)
                + inp * c["weights"][3].float()
            ).bfloat16()
            history = torch.cat((history[1:], inp.unsqueeze(0)))
            conv[base, token - lo + 2] = inp.bfloat16()
            q, k, v = convolved.split((hk * d, hk * d, hv * d))
            q, k, v = (
                q.view(hk, d).float(),
                k.view(hk, d).float(),
                v.view(hv, d).float(),
            )
            q = (
                q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            ).bfloat16().float().repeat_interleave(hv // hk, 0) * d**-0.5
            k = (
                (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6))
                .bfloat16()
                .float()
                .repeat_interleave(hv // hk, 0)
            )
            g = -c["a_log"].exp() * F.softplus(c["a"][token].float() + c["dt_bias"])
            beta = c["b"][token].float().sigmoid().bfloat16().float()
            state_row *= g.exp()[:, None, None]
            prediction = (state_row * k[:, :, None]).sum(1)
            state_row += k[:, :, None] * ((v - prediction) * beta[:, None])[:, None, :]
            state[int(c["ids"][row, token - lo])] = state_row
            value = (state_row * q[:, :, None]).sum(1).bfloat16().float()
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            out[token] = (
                value * c["norm"].float() * F.silu(c["z"][token].float())
            ).bfloat16()
    actual = call(x)
    torch.npu.synchronize()
    # Same independent CPU-golden tolerances as the existing SDK MTP tests.
    # Exact bank-addressing/stress comparisons use the tighter SDK tolerance.
    torch.testing.assert_close(actual.cpu(), out, rtol=5e-3, atol=2e-2)
    torch.testing.assert_close(x.ssm.cpu(), state, rtol=5e-3, atol=2.5e-5)
    torch.testing.assert_close(x.conv.cpu(), conv, rtol=0, atol=0)


def test_native_400_replays_preserve_every_checkpoint():
    x = make_case((4, 2, 1, 0))
    original_c, original_s = x.conv_back.clone(), x.ssm_back.clone()
    expected, conv, ssm = model_reference(x)
    for _ in range(3):
        call(x)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = call(x)
    for _ in range(400):
        x.conv_back.copy_(original_c)
        x.ssm_back.copy_(original_s)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result, expected, rtol=0.01, atol=1e-4)
        torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
        torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("length", [4, 22, 128, 801])
def test_native_prefill_null_slots_compute_finite_output_without_cache_writes(length):
    case = list(prefill_case(length))
    x, _, _, lower, full, minus, matrices = case
    case[1] = torch.full((2,), -1, dtype=torch.int32).npu()
    case[2] = torch.zeros(2, dtype=torch.int32).npu()
    before_c, before_s = x.conv_back.clone(), x.ssm_back.clone()
    # Model warmup uses nonempty sequences with null cache slots. A private
    # compact bank is an independent oracle for stateless output on each row.
    scratch_c = x.conv.new_zeros((2, 3, x.conv.size(-1)))
    scratch_s = x.ssm.new_zeros((2, *x.ssm.shape[1:]))
    slots = torch.arange(2, dtype=torch.int32).npu()
    expected = compact.mega_gdn_prefill_op(
        x.qkv,
        x.b,
        x.a,
        x.z,
        x.weights,
        scratch_c,
        x.a_log,
        x.dt_bias,
        case[1],
        slots,
        case[1],
        slots,
        scratch_s,
        lower,
        full,
        minus,
        x.starts,
        x.norm,
        matrices,
    )
    actual = prefill_call(case)
    torch.npu.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.conv_back, before_c, rtol=0, atol=0)
    torch.testing.assert_close(x.ssm_back, before_s, rtol=0, atol=0)


def test_native_single_token_nonzero_storage_offset():
    x = make_case((1,), width=4)
    # A graph bucket may contain just one token even though the checkpoint
    # table retains four columns from MTP. Use nonzero projection offsets.
    x.qkv = x.qkv[1:2]
    x.z = x.z[1:2]
    x.a = x.a[1:2]
    x.b = x.b[1:2]
    x.accepted.fill_(4)
    expected, conv, ssm = model_reference(x)
    actual = call(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-4)
    torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)


def test_native_first_two_graphs_replay_in_reverse_order():
    warm = make_case()
    for _ in range(3):
        call(warm)
    x = make_case()
    original_c, original_s = x.conv_back.clone(), x.ssm_back.clone()
    first, second = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(first):
        first_result = call(x)
    with torch.npu.graph(second):
        second_result = call(x)
    expected, conv, ssm = model_reference(x)
    for graph, result in ((second, second_result), (first, first_result)):
        x.conv_back.copy_(original_c)
        x.ssm_back.copy_(original_s)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result, expected, rtol=0.01, atol=1e-4)
        torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
        torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)


def test_native_realistic_head_gates_and_norm_weights():
    x = make_case((4, 2, 1, 0))
    x.a_log.copy_(torch.linspace(-2, 0.25, x.z.size(1)).npu())
    x.dt_bias.copy_(torch.linspace(-0.75, 0.5, x.z.size(1)).npu())
    x.norm.copy_(torch.linspace(0.75, 1.25, 128).bfloat16().npu())
    expected, conv, ssm = model_reference(x)
    actual = call(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-4)
    torch.testing.assert_close(x.conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(x.ssm, ssm, rtol=2e-5, atol=2e-6)


def test_native_prefill_null_slot_replay_recomputes_workspace():
    case = list(prefill_case(22, initial=False))
    x, read, write, lower, full, minus, matrices = case
    for _ in range(2):
        prefill_call(case)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = prefill_call(case)
    graph.replay()
    torch.npu.synchronize()
    # Keep the captured workspace but change both input activations and IDs.
    # An eager compact oracle alone can accidentally prime the allocator's
    # workspace with exactly the Q/K/V that a broken null-slot path fails to
    # compute, hiding its uninitialized read.
    x.qkv.mul_(-2)
    write.zero_()
    before_c, before_s = x.conv_back.clone(), x.ssm_back.clone()
    scratch_c = x.conv.new_zeros((2, 3, x.conv.size(-1)))
    scratch_s = x.ssm.new_zeros((2, *x.ssm.shape[1:]))
    slots = torch.arange(2, dtype=torch.int32).npu()
    expected = compact.mega_gdn_prefill_op(
        x.qkv,
        x.b,
        x.a,
        x.z,
        x.weights,
        scratch_c,
        x.a_log,
        x.dt_bias,
        read,
        slots,
        read,
        slots,
        scratch_s,
        lower,
        full,
        minus,
        x.starts,
        x.norm,
        matrices,
    )
    graph.replay()
    torch.npu.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.conv_back, before_c, rtol=0, atol=0)
    torch.testing.assert_close(x.ssm_back, before_s, rtol=0, atol=0)


@pytest.mark.parametrize("write_kind", ["null", "negative", "past_end"])
def test_native_single_prefill_invalid_slot_preserves_guards(write_kind):
    # The smallest MTP3 graph warmup has one four-token sequence and slot 0.
    # Put a complete guard slot before and after both banks so an invalid
    # write is detected without corrupting another live NPU allocation.
    x = make_case((4,), strided=False)
    c_guard = torch.full(
        (x.conv.size(0) + 2, *x.conv.shape[1:]),
        7,
        dtype=x.conv.dtype,
        device=x.conv.device,
    )
    s_guard = torch.full(
        (x.ssm.size(0) + 2, *x.ssm.shape[1:]),
        11,
        dtype=x.ssm.dtype,
        device=x.ssm.device,
    )
    x.conv, x.ssm = c_guard[1:-1], s_guard[1:-1]
    before_c, before_s = c_guard.clone(), s_guard.clone()
    read = torch.full((1,), -1, dtype=torch.int32, device=x.qkv.device)
    value = {"null": 0, "negative": -1, "past_end": x.conv.size(0)}[write_kind]
    write = torch.full_like(read, value)
    lower = torch.tril(torch.ones(128, 128, device=x.qkv.device), diagonal=-1)
    full = torch.tril(torch.ones_like(lower))
    minus = -torch.eye(128, dtype=torch.bfloat16, device=x.qkv.device)
    actual = prefill_call((x, read, write, lower, full, minus, x.z.size(1)))
    torch.npu.synchronize()
    torch.testing.assert_close(c_guard, before_c, rtol=0, atol=0)
    torch.testing.assert_close(s_guard, before_s, rtol=0, atol=0)
    # Invalid cache writes must still produce a defined stateless output.
    scratch_c = x.conv.new_zeros((1, 3, x.conv.size(-1)))
    scratch_s = x.ssm.new_zeros((1, *x.ssm.shape[1:]))
    slots = torch.zeros_like(read)
    expected = compact.mega_gdn_prefill_op(
        x.qkv,
        x.b,
        x.a,
        x.z,
        x.weights,
        scratch_c,
        x.a_log,
        x.dt_bias,
        read,
        slots,
        read,
        slots,
        scratch_s,
        lower,
        full,
        minus,
        x.starts,
        x.norm,
        x.z.size(1),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
