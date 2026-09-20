"""Independent model precision boundaries, not the legacy SDK oracle."""

import pytest
import test_mega_gdn_native as t
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("length", [1, 4])
def test_decay_gate_remains_fp32(length):
    x = t.make_case((length,), heads=(1, 1), width=4, strided=False)
    x.qkv.zero_()
    x.conv.zero_()
    x.ssm.fill_(0.125)
    x.a.fill_(0.3125)
    x.a_log.zero_()
    x.dt_bias.fill_(0.125)
    g = -torch.exp(x.a_log.cpu()) * F.softplus(x.a.cpu().float() + x.dt_bias.cpu())
    t.call(x)
    torch.npu.synchronize()
    state = torch.full((1, 128, 128), 0.125)
    ids = x.ids.cpu()
    for token in range(length):
        state = state * torch.exp(g[token])[:, None, None]
        torch.testing.assert_close(
            x.ssm[int(ids[0, token])].cpu(), state, rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("length", [1, 4])
def test_normalized_qk_materialized_bf16(length):
    x = t.make_case((length,), heads=(1, 1), width=4, strided=False)
    x.qkv.fill_(1)
    x.conv.zero_()
    x.ssm.zero_()
    x.weights.zero_()
    x.weights[3].fill_(1)
    x.a.fill_(-80)
    x.a_log.zero_()
    x.dt_bias.zero_()
    x.b.zero_()
    conv = F.silu(torch.ones(128)).bfloat16().float()
    key = (conv / torch.sqrt((conv * conv).sum() + 1e-6)).bfloat16().float()
    state = torch.zeros((128, 128))
    t.call(x)
    torch.npu.synchronize()
    ids = x.ids.cpu()
    for token in range(length):
        delta = (conv - key @ state) * 0.5
        state = state + key[:, None] * delta[None, :]
        torch.testing.assert_close(
            x.ssm[int(ids[0, token])].cpu()[0], state, rtol=2e-5, atol=2e-6
        )
