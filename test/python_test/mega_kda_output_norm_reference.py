"""Public BF16 KDA output followed by the model's FP32 gated RMSNorm."""

import torch


def gated_rms_norm(output, gate, weight, eps):
    x = output.float()
    variance = x.square().mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    return (normed * weight.float() * torch.sigmoid(gate.float())).to(output.dtype)
