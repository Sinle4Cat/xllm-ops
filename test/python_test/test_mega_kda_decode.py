# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""CPU contract checks and opt-in real ACLNN precision/replay tests."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from mega_kda_decode_reference import (
    MTP, NATIVE, OUTPUTS, PUBLIC_COMPAT, make_case, public_compat_stages, reference,
)
from mega_kda_test_utils import GLM_KDA_HEADS, TP_SIZES, bitwise_equal, error_metrics


ROOT = Path(__file__).resolve().parents[2]
DECODE_MODES = [(NATIVE, 1), (PUBLIC_COMPAT, 1)] + [(MTP, capacity) for capacity in range(1, 18)]
NPU_TEST = pytest.mark.skipif(os.getenv("MEGA_KDA_TEST_NPU") is None,
                              reason="native validation requires an explicitly selected exclusive NPU")


def test_native_host_policy(tmp_path):
    source = tmp_path / "policy.cpp"
    source.write_text(r'''
#include <cassert>
#include "mega_kda_decode_policy.h"
int main() {
    using namespace mega_kda;
    TilingData t{2, 2, 8, 1, 3, 4, 7, 5, 9};
    assert(select_key(0, false, t) == 1);
    assert(select_key(1, true, t) == 2);
    assert(select_key(3, false, t) == 3);
    assert(select_key(3, true, t) == 0);
    assert(select_key(0, true, t) == 0);
    assert(select_key(1, false, t) == 0);
    assert(select_key(2, false, t) == 0);
    for (int tp : {1, 2, 4, 8, 16, 32, 64}) {
        t.heads = 64 / tp;
        assert(select_key(0, false, t) == 1);
        assert(select_key(1, true, t) == 2);
        assert(select_key(3, false, t) == 3);
    }
    t.heads = 8;
    t.max_query_tokens = 4; t.conv_history = 6; t.tokens = 7;
    assert(select_key(1, true, t) == 2);
    assert(select_key(0, false, t) == 0);
    assert(select_key(3, false, t) == 0);
    t.tokens = 9;
    assert(select_key(1, true, t) == 0);
    t.tokens = 7; t.conv_history = 3;
    assert(select_key(1, true, t) == 0);
    t.conv_history = 6; t.heads = 129;
    assert(select_key(1, true, t) == 0);
    t.heads = 8;
    for (int capacity = 1; capacity <= 17; ++capacity) {
        t.max_query_tokens = capacity;
        t.conv_history = capacity + 2;
        t.tokens = t.batch * capacity;
        assert(select_key(1, true, t) == 2);
    }
    t.max_query_tokens = 18; t.conv_history = 20; t.tokens = 36;
    assert(select_key(1, true, t) == 0);
    assert(!fits_bytes({9223372036854775807LL, 2}));
    assert(!fits_bytes({-1, 2}));
}
''')
    exe = tmp_path / "policy"
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source),
                    "-I", str(ROOT / "xllm_ops/mega_kda_decode/op_host"), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_single_registered_entry():
    op = ROOT / "xllm_ops/mega_kda_decode"
    definitions = "\n".join(path.read_text() for path in op.rglob("*.cpp"))
    assert definitions.count("OP_ADD(MegaKdaDecode)") == 1
    assert definitions.count('__global__ __aicore__ void mega_kda_decode(') == 1
    assert all(f"TILING_KEY_IS({key})" in definitions for key in (1, 2, 3))
    assert "GetRequiredInputShape" in definitions and "GetOptionalInputShape" in definitions
    assert "mega_kda_decode" in (ROOT / "xllm_ops/build_aclnn.sh").read_text()


@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
@pytest.mark.parametrize("bias", [False, True])
def test_reference_state_contract(mode, capacity, bias):
    data = make_case(batch=3, capacity=capacity, mode=mode, bias=bias, ragged=True, padding=True)
    snapshots = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
    result = reference(data)
    for key, before in snapshots.items():
        assert torch.equal(before, data[key]), key
    out, conv, ssm = result
    assert [x.dtype for x in result] == [torch.bfloat16, torch.bfloat16, torch.float32]
    assert all(torch.isfinite(x).all() for x in result)
    assert torch.equal(conv[0], data["conv_state_out"][0])
    assert torch.equal(ssm[0], data["ssm_state_out"][0])
    for row in range(2):
        begin, end = data["cu_seqlens"][row:row + 2].tolist()
        accepted = int(data["num_accepted_tokens"][row]) if mode == MTP else 1
        slot = int(data["conv_write_indices"][row])
        assert torch.equal(conv[slot, :2], data["conv_state_in"][row, accepted:accepted + 2])
        assert torch.equal(conv[slot, 2:2 + end - begin], data["qkv"][begin:end])
        assert torch.equal(conv[slot, 2 + end - begin:], data["conv_state_out"][slot, 2 + end - begin:])
    for _ in range(20):
        assert all(bitwise_equal(a, b) for a, b in zip(result, reference(data)))


def test_reference_checkpoint_restore_and_discard():
    data = make_case(mode=1, capacity=4)
    first = reference(data)
    data["num_accepted_tokens"][0] = 1
    restored = reference(data)
    assert not torch.equal(first[0][0], restored[0][0])
    data["ssm_write_indices"].fill_(-1)
    data["conv_write_indices"].fill_(-1)
    discarded = reference(data)
    assert torch.equal(restored[0], discarded[0])
    assert torch.equal(discarded[1], data["conv_state_out"])
    assert torch.equal(discarded[2], data["ssm_state_out"])


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
def test_tp_reference_shape(tp, mode, capacity):
    heads = GLM_KDA_HEADS // tp
    data = make_case(heads=heads, mode=mode, capacity=capacity, bias=True)
    result = reference(data)
    assert data["qkv"].shape == (2 * capacity, heads * 384)
    assert data["gate"].shape == (2 * capacity, heads, 128)
    for name, value in zip(OUTPUTS, result):
        assert value.shape == data[name].shape and value.dtype == data[name].dtype
        assert torch.isfinite(value).all(), name
    assert result[2].shape[1:] == (heads, 128, 128)


@pytest.mark.parametrize("batch", [3, 4, 128])
@pytest.mark.parametrize("mode", [NATIVE, PUBLIC_COMPAT])
def test_plain_padding_keeps_one_token_per_row(batch, mode):
    data = make_case(batch=batch, heads=1, mode=mode, ragged=True, padding=True)
    assert data["qkv"].shape[0] == batch
    assert torch.equal(data["cu_seqlens"], torch.arange(batch + 1, dtype=torch.int32))
    assert int(data["conv_read_indices"][-1]) == -1
    out, _, _ = reference(data)
    assert torch.equal(out[-1], data["output"][-1])


@pytest.mark.parametrize("invalid", ["duplicate_conv", "duplicate_ssm", "accepted", "offsets", "slot"])
def test_reference_rejects_invalid_metadata(invalid):
    data = make_case(mode=1, capacity=4)
    if invalid == "duplicate_conv":
        data["conv_write_indices"].fill_(0)
    elif invalid == "duplicate_ssm":
        data["ssm_write_indices"].fill_(0)
    elif invalid == "accepted":
        data["num_accepted_tokens"][0] = 0
    elif invalid == "offsets":
        data["cu_seqlens"][-1] += 1
    else:
        data["conv_write_indices"][0] = 999
    with pytest.raises(ValueError):
        reference(data)


@pytest.mark.parametrize("invalid", ["accepted", "capacity", "offsets", "history"])
def test_public_compat_rejects_mtp_or_nonunit_contract(invalid: str) -> None:
    data = make_case(mode=PUBLIC_COMPAT)
    if invalid == "accepted":
        data["num_accepted_tokens"] = torch.ones(2, dtype=torch.int32)
    elif invalid == "capacity":
        data["max_query_tokens"] = 2
    elif invalid == "offsets":
        data["cu_seqlens"][:] = torch.tensor([0, 0, 2], dtype=torch.int32)
    else:
        data["conv_state_in"] = data["conv_state_in"][:, :2].contiguous()
    with pytest.raises(ValueError):
        reference(data)


def _public_discard_case() -> dict:
    data = make_case(batch=3, heads=8, mode=PUBLIC_COMPAT)
    data["conv_read_indices"][2] = -1
    data["ssm_read_indices"][1] = -1
    data["conv_write_indices"][0] = -1
    data["ssm_write_indices"][0] = -1
    return data


def test_public_compat_inactive_and_discard_contract() -> None:
    data = _public_discard_case()
    output, conv, state = reference(data)
    assert not bitwise_equal(output[0], data["output"][0])
    assert bitwise_equal(output[1:], data["output"][1:])
    assert bitwise_equal(conv, data["conv_state_out"])
    assert bitwise_equal(state, data["ssm_state_out"])


def _public_case(batch: int, heads: int, variant: str) -> dict:
    data = make_case(batch=batch, heads=heads, mode=PUBLIC_COMPAT)
    channels_per_part = heads * 128
    if variant == "zero":
        data["conv_weight"][:, :2 * channels_per_part].zero_()
    elif variant == "tiny":
        data["qkv"].fill_(1)
        data["conv_state_in"].fill_(1)
        data["ssm_state_in"].fill_(1)
        data["conv_weight"][:, :2 * channels_per_part].fill_(1e-7)
    elif variant == "wide":
        # BF16 values above FP16's finite range, with finite small Conv products.
        data["qkv"].fill_(70000)
        data["conv_state_in"].fill_(70000)
        data["conv_weight"].fill_(1e-6)
    elif variant == "gate_extremes":
        data["gate"][:, :, ::2] = -80
        data["gate"][:, :, 1::2] = 80
        data["gate_bias"].zero_()
        data["a_log"].zero_()
    else:
        raise ValueError(variant)
    return data


@pytest.mark.parametrize("variant", ["zero", "tiny", "wide", "gate_extremes"])
def test_public_compat_fp32_stages(variant: str) -> None:
    data = _public_case(1, 2, variant)
    sequence = torch.cat((data["conv_state_in"][0], data["qkv"][:1]))
    stages = public_compat_stages(data, sequence)
    assert all(torch.isfinite(value).all() for value in stages.values())
    assert all(torch.isfinite(value).all() for value in reference(data))
    if variant == "zero":
        assert torch.count_nonzero(stages["q"]) == 0
        assert torch.count_nonzero(stages["k"]) == 0
        assert torch.count_nonzero(reference(data)[0]) == 0
    elif variant == "tiny":
        torch.testing.assert_close(stages["k"].square().sum(-1), torch.ones(2), rtol=1e-6, atol=1e-6)
        vector = stages["conv"].reshape(3, 2, 128)[1]
        epsilon_norm = vector / torch.sqrt(vector.square().sum(-1, keepdim=True) + 1e-6)
        assert epsilon_norm.abs().max() < 1e-3
        assert stages["k"].abs().max() > 0.01
        public_output, _, public_state = reference(data)
        counterfactual = dict(data, mode=NATIVE)
        native_output, _, native_state = reference(counterfactual)
        # A mistaken epsilon policy must fail the unchanged NPU pointwise gates.
        assert ((public_output.float() - native_output.float()).abs()
                > 1e-3 + 1e-2 * public_output.float().abs()).any()
        assert ((public_state - native_state).abs()
                > 1e-4 + 1e-3 * public_state.abs()).any()
    elif variant == "wide":
        assert not torch.isfinite(sequence.half()).all()
        assert torch.isfinite(stages["conv"]).all()
    else:
        torch.testing.assert_close(stages["decay"][:, ::2], torch.ones((2, 64)), rtol=0, atol=0)
        torch.testing.assert_close(stages["decay"][:, 1::2], torch.full((2, 64), -5.0).exp(), rtol=0, atol=0)


@pytest.mark.parametrize("mode", [NATIVE, MTP, PUBLIC_COMPAT])
def test_reference_preserves_common_bf16_activation_boundaries(mode: int) -> None:
    data = make_case(batch=1, heads=1, mode=mode)
    data["qkv"].fill_(1)
    data["conv_state_in"].fill_(1)
    data["conv_weight"].zero_()
    data["conv_weight"][0].fill_(1)
    data["conv_weight"][1].fill_(1 / 256)
    data["ssm_state_in"].zero_()
    data["beta"].fill_(1 / 128)
    actual = reference(data)
    # 1+1/256 is the BF16 midpoint and rounds to 1 before SiLU.
    value = torch.nn.functional.silu(torch.tensor(1.0, dtype=torch.float64)).bfloat16().double()
    beta = torch.sigmoid(data["beta"][0, 0].double()).bfloat16().double()
    omitted_conv_round = torch.nn.functional.silu(torch.tensor(1 + 1 / 256, dtype=torch.float64)).bfloat16().double()
    omitted_beta_round = torch.sigmoid(data["beta"][0, 0].double())
    assert value != omitted_conv_round and beta != omitted_beta_round
    square_norm = 128 * value.square() + (0 if mode == PUBLIC_COMPAT else 1e-6)
    state_scalar = value * beta * value / square_norm.sqrt()
    torch.testing.assert_close(actual[2][2, 0], state_scalar.float().expand(128, 128), rtol=0, atol=0)
    expected_output = (state_scalar * (128 * value / square_norm.sqrt()) * 128 ** -0.5).bfloat16()
    torch.testing.assert_close(actual[0][0, 0], expected_output.expand(128), rtol=0, atol=0)


def test_public_compat_fp32_gate_reciprocal_then_multiply() -> None:
    data = make_case(batch=1, heads=1, mode=PUBLIC_COMPAT)
    data["gate"][0, 0] = torch.linspace(-8, 8, 128).bfloat16()
    data["gate_bias"].zero_()
    data["a_log"].zero_()
    sequence = torch.cat((data["conv_state_in"][0], data["qkv"]))
    stages = public_compat_stages(data, sequence)
    denominator = 1.0 + torch.exp(-data["gate"][0].float())
    want = (1.0 / denominator) * -5.0
    assert bitwise_equal(stages["gate"], want)
    # Tensor.__rtruediv__ may lower scalar/Tensor to reciprocal*scalar too.
    # Tensor/Tensor div supplies an independent direct-division counterfactual.
    direct_division = torch.div(torch.full_like(denominator, -5.0), denominator)
    assert not bitwise_equal(want, direct_division)
    assert bitwise_equal(stages["beta"], torch.sigmoid(data["beta"][0].float()).bfloat16().float())


def npu_case(data):
    device = os.getenv("MEGA_KDA_TEST_NPU")
    if device is None:
        pytest.skip("set MEGA_KDA_TEST_NPU to an exclusively available NPU for native tests")
    import torch_npu  # noqa: F401
    import custom_ops_lib
    torch.npu.set_device(int(device))
    tensors = {k: v.to(f"npu:{device}") if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    return custom_ops_lib, tensors


def assert_native(data, actual, input_snapshots=None, expected=None):
    if expected is None:
        expected = reference(data)
    records = {}
    for name, want in zip(OUTPUTS, expected):
        got = actual[name].cpu()
        assert got.shape == want.shape and got.dtype == want.dtype, name
        assert torch.isfinite(got).all() and torch.isfinite(want).all(), name
        if name == "conv_state_out":
            assert bitwise_equal(got, want), name
            continue
        rtol, atol = (1e-2, 1e-3) if name == "output" else (1e-3, 1e-4)
        active = want != 37
        stats = error_metrics(got[active], want[active], near=0.001, relative=rtol, absolute=atol)
        records[name] = stats
        assert stats["finite"], (name, stats)
        # Retain the existing pointwise gate; split metrics are additional diagnostics.
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
        # Pool padding and discarded writes are exact, even when active arithmetic is approximate.
        untouched = want == 37
        assert bitwise_equal(got[untouched], want[untouched]), name
    for key, value in (data if input_snapshots is None else input_snapshots).items():
        if isinstance(value, torch.Tensor) and key not in OUTPUTS:
            assert bitwise_equal(actual[key].cpu(), value), key
    print(json.dumps({"mode": data["mode"], "capacity": data["max_query_tokens"],
                      "shape": list(data["gate"].shape), "errors": records}), flush=True)
    return expected


def check_fixed20(data):
    ops, actual = npu_case(data)
    for _ in range(3):
        ops.mega_kda_decode(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = ops.mega_kda_decode(**actual)
    assert all(a.data_ptr() == actual[name].data_ptr() for a, name in zip(result, OUTPUTS))
    expected = reference(data)
    first = None
    for _ in range(20):
        graph.replay()
        assert_native(data, actual, expected=expected)
        current = [actual[name].cpu() for name in OUTPUTS]
        if first is None:
            first = current
        assert all(bitwise_equal(a, b) for a, b in zip(first, current))


@pytest.mark.parametrize("batch,heads", [(1, 1), (2, 2), (4, 2), (8, 8), (16, 8), (32, 16), (64, 16), (128, 32)])
@pytest.mark.parametrize("mode,capacity", [(NATIVE, 1), (PUBLIC_COMPAT, 1), (MTP, 1), (MTP, 4)])
@NPU_TEST
def test_native_fixed20(batch, heads, mode, capacity):
    data = make_case(batch=batch, heads=heads, capacity=capacity, mode=mode,
                     bias=True, ragged=True, padding=batch > 1)
    check_fixed20(data)


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
@NPU_TEST
def test_native_tp_fixed20(tp, mode, capacity):
    data = make_case(heads=GLM_KDA_HEADS // tp, capacity=capacity, mode=mode, bias=True)
    check_fixed20(data)


def check_changed20_carry20(data: dict) -> None:
    mode, capacity = data["mode"], data["max_query_tokens"]
    batch = data["conv_read_indices"].numel()
    ops, actual = npu_case(data)
    for _ in range(3):
        ops.mega_kda_decode(**actual)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        ops.mega_kda_decode(**actual)
    for carry in (False, True):
        for step in range(20):
            data["qkv"].add_(0.001)
            actual["qkv"].copy_(data["qkv"])
            if mode == MTP:
                data["num_accepted_tokens"].fill_(step % capacity + 1)
                actual["num_accepted_tokens"].copy_(data["num_accepted_tokens"])
            # Change destinations without replacing bindings or recapturing.
            destinations = torch.arange(1, batch + 1, dtype=torch.int32)
            data["conv_write_indices"].copy_(destinations if step % 2 else destinations.flip(0))
            actual["conv_write_indices"].copy_(data["conv_write_indices"])
            snapshots = {k: v.cpu() for k, v in actual.items()
                         if isinstance(v, torch.Tensor) and k not in OUTPUTS}
            graph.replay()
            expected = assert_native(data, actual, snapshots)
            for name, result in zip(OUTPUTS, expected):
                data[name].copy_(result)
            if carry:
                for row in range(batch):
                    cr, cw = int(data["conv_read_indices"][row]), int(data["conv_write_indices"][row])
                    data["conv_state_in"][cr].copy_(expected[1][cw])
                    actual["conv_state_in"][cr].copy_(actual["conv_state_out"][cw])
                    for token in range(capacity):
                        sr, sw = int(data["ssm_read_indices"][row, token]), int(data["ssm_write_indices"][row, token])
                        data["ssm_state_in"][sr].copy_(expected[2][sw])
                        actual["ssm_state_in"][sr].copy_(actual["ssm_state_out"][sw])


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
@NPU_TEST
def test_native_graph_changed20_carry20(tp, mode, capacity):
    data = make_case(heads=GLM_KDA_HEADS // tp, capacity=capacity, mode=mode)
    check_changed20_carry20(data)


@pytest.mark.parametrize("batch,heads", [(1, 8), (4, 8), (5, 8), (2, 3)])
@NPU_TEST
def test_public_compat_specialized_fixed20_changed20_carry20(batch: int, heads: int) -> None:
    data = make_case(batch=batch, heads=heads, mode=PUBLIC_COMPAT, bias=True)
    check_fixed20(data)
    check_changed20_carry20(data)


@pytest.mark.parametrize("batch,heads", [(1, 8), (4, 8), (5, 8), (2, 3)])
@pytest.mark.parametrize("variant", ["zero", "tiny", "wide", "gate_extremes"])
@NPU_TEST
def test_public_compat_numerical_edges(batch: int, heads: int, variant: str) -> None:
    data = _public_case(batch, heads, variant)
    ops, actual = npu_case(data)
    ops.mega_kda_decode(**actual)
    assert_native(data, actual)


@NPU_TEST
def test_public_compat_row_dispatch() -> None:
    # Set this from the validation device's PlatformAscendC GetCoreNumAiv value.
    # The launch uses min(AIV cores, 8*B*H); record the actual launch calculation.
    core_count = os.getenv("MEGA_KDA_TEST_AIV_CORES")
    if core_count is None:
        pytest.skip("set MEGA_KDA_TEST_AIV_CORES from the selected device for dispatch coverage")
    cores = int(core_count)
    assert cores > 2
    rows_seen = set()
    for batch, heads in ((1, 1), ((cores + 1) // 2, 1)):
        blocks = min(cores, 8 * batch * heads)
        rows = 64 if 2 * batch * heads >= blocks else 32
        data = make_case(batch=batch, heads=heads, mode=PUBLIC_COMPAT)
        ops, actual = npu_case(data)
        ops.mega_kda_decode(**actual)
        assert_native(data, actual)
        rows_seen.add(rows)
        print(json.dumps({"aiv_cores": cores, "launched_blocks": blocks, "batch": batch,
                          "heads": heads, "v_rows": rows, "mode": PUBLIC_COMPAT}), flush=True)
    assert rows_seen == {32, 64}


@pytest.mark.parametrize("invalid", ["accepted", "capacity"])
@NPU_TEST
def test_public_compat_binding_rejects_mtp_contract(invalid: str) -> None:
    data = make_case(mode=PUBLIC_COMPAT)
    if invalid == "accepted":
        data["num_accepted_tokens"] = torch.ones(2, dtype=torch.int32)
    else:
        data["max_query_tokens"] = 2
    ops, actual = npu_case(data)
    with pytest.raises(RuntimeError, match="mode/capacity/accepted-count"):
        ops.mega_kda_decode(**actual)


@NPU_TEST
def test_public_compat_nonunit_offsets_do_not_publish() -> None:
    data = make_case(mode=PUBLIC_COMPAT)
    data["cu_seqlens"][:] = torch.tensor([0, 0, 2], dtype=torch.int32)
    ops, actual = npu_case(data)
    ops.mega_kda_decode(**actual)
    for name in OUTPUTS:
        assert bitwise_equal(actual[name].cpu(), data[name])


@NPU_TEST
def test_public_compat_native_inactive_and_discard() -> None:
    data = _public_discard_case()
    ops, actual = npu_case(data)
    ops.mega_kda_decode(**actual)
    assert_native(data, actual)


@NPU_TEST
def test_native_rejects_input_output_alias():
    data = make_case()
    ops, actual = npu_case(data)
    actual["conv_state_out"] = actual["conv_state_in"]
    with pytest.raises(RuntimeError, match="must not alias"):
        ops.mega_kda_decode(**actual)
