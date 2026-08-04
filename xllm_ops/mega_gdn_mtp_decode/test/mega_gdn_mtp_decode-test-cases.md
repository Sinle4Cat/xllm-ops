<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnMtpDecode 用例设计文档

## 1. 算子标杆

PyTorch 参考实现：

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MegaGdnMtpResult:
    conv_out: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    out: torch.Tensor


def mega_gdn_mtp_reference(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
) -> MegaGdnMtpResult:
    batch_size, sequence_length, conv_dim = qkv.shape
    speculative_tokens = sequence_length - 1
    head_dim = 128
    num_v_heads = z.shape[2]
    num_k_heads = (conv_dim - num_v_heads * head_dim) // (2 * head_dim)
    state_stride = sequence_length

    conv_state_out = conv_state.clone()
    ssm_state_out = ssm_state.clone()
    conv_outputs = []
    outputs = []

    for batch_idx in range(batch_size):
        read_slot = int(read_state_indices[batch_idx])
        write_slot = int(write_state_indices[batch_idx])
        accepted = int(num_accepted_tokens[batch_idx])
        history = conv_state[read_slot, accepted - 1 : accepted + 2].float()
        batch_conv = []
        for token_idx in range(sequence_length):
            x_t = qkv[batch_idx, token_idx].float()
            conv_fp32 = F.silu(
                (history * conv_weight[:3].float()).sum(dim=0)
                + x_t * conv_weight[3].float()
            )
            conv_bf16 = conv_fp32.to(torch.bfloat16)
            batch_conv.append(conv_bf16)
            history = torch.cat((history[1:], x_t.unsqueeze(0)), dim=0)

        batch_conv_tensor = torch.stack(batch_conv)
        conv_outputs.append(batch_conv_tensor)
        conv_state_out[write_slot, :2] = conv_state[
            read_slot, accepted : accepted + 2
        ]
        conv_state_out[write_slot, 2 : speculative_tokens + 3] = qkv[batch_idx]

        q = batch_conv_tensor[:, : num_k_heads * head_dim].reshape(
            sequence_length, num_k_heads, head_dim
        )
        k = batch_conv_tensor[
            :, num_k_heads * head_dim : 2 * num_k_heads * head_dim
        ].reshape(sequence_length, num_k_heads, head_dim)
        v = batch_conv_tensor[:, 2 * num_k_heads * head_dim :].reshape(
            sequence_length, num_v_heads, head_dim
        )
        q = F.normalize(q.float(), dim=-1, eps=1e-6) / head_dim**0.5
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        repeat = num_v_heads // num_k_heads
        q = q.repeat_interleave(repeat, dim=1)
        k = k.repeat_interleave(repeat, dim=1)

        read_checkpoint = read_slot * state_stride + accepted - 1
        state = ssm_state[read_checkpoint].clone().float()
        batch_output = []
        for token_idx in range(sequence_length):
            decay = torch.exp(
                -torch.exp(a_log)
                * F.softplus(a[batch_idx, token_idx].float() + dt_bias)
            )
            beta = torch.sigmoid(b[batch_idx, token_idx].float())
            state = state * decay[:, None, None]
            prediction = torch.einsum("hkv,hk->hv", state, k[token_idx])
            delta = (v[token_idx].float() - prediction) * beta[:, None]
            state = state + torch.einsum("hk,hv->hkv", k[token_idx], delta)
            readout = torch.einsum("hkv,hk->hv", state, q[token_idx])
            write_checkpoint = write_slot * state_stride + token_idx
            ssm_state_out[write_checkpoint] = state

            norm_input = readout.to(torch.bfloat16).float()
            rms = torch.rsqrt(norm_input.square().mean(dim=-1, keepdim=True) + 1e-6)
            gated = norm_input * rms * norm_weight.float()
            gated = gated * F.silu(z[batch_idx, token_idx].float())
            batch_output.append(gated.to(torch.bfloat16))
        outputs.append(torch.stack(batch_output))

    return MegaGdnMtpResult(
        conv_out=torch.stack(conv_outputs),
        conv_state=conv_state_out,
        ssm_state=ssm_state_out,
        out=torch.stack(outputs),
    )
```

---

## 2. 用例说明

### 2.1 测试配置

配置 tuple 含义为：

```text
(batch_size, K, num_k_heads, num_v_heads, num_state_slots)
```

`head_dim` 固定为 128，`sequence_length=K+1`，
`conv_state_length=K+3`。

```python
SUPPORTED_DTYPES = [torch.bfloat16]

TEST_SHAPES = [
    ("K1", "BS1 minimal heads", (1, 1, 1, 1, 2)),
    ("K1", "BS2 grouped value heads", (2, 1, 1, 4, 4)),
    ("K1", "Qwen3.5 geometry BS1", (1, 1, 8, 24, 2)),
    ("K2", "BS1 minimal heads", (1, 2, 1, 1, 2)),
    ("K2", "BS3 two value heads", (3, 2, 1, 2, 5)),
    ("K2", "Qwen3.5 geometry BS2", (2, 2, 8, 24, 3)),
    ("K3", "BS1 minimal heads", (1, 3, 1, 1, 2)),
    ("K3", "BS2 grouped value heads", (2, 3, 2, 8, 4)),
    ("K3", "Qwen3.5 geometry BS1", (1, 3, 8, 24, 2)),
    ("K4", "BS1 minimal heads", (1, 4, 1, 1, 2)),
    ("K4", "BS4 two value heads", (4, 4, 1, 2, 6)),
    ("K4", "Qwen3.5 geometry BS1", (1, 4, 8, 24, 2)),
    ("K5", "BS1 minimal heads", (1, 5, 1, 1, 2)),
    ("K5", "BS2 grouped value heads", (2, 5, 2, 4, 4)),
    ("K5", "Qwen3.5 geometry BS1", (1, 5, 8, 24, 2)),
    ("K8", "BS1 minimal heads", (1, 8, 1, 1, 2)),
    ("K8", "BS2 grouped value heads", (2, 8, 2, 8, 4)),
    ("K8", "Qwen3.5 geometry BS1", (1, 8, 8, 24, 2)),
]

GENERAL_SHAPES = [
    ("SameSlot", "K1 accepted first", (1, 1, 1, 1, 1)),
    ("SameSlot", "K2 accepted all", (1, 2, 1, 2, 1)),
    ("SameSlot", "K3 middle accepted", (1, 3, 2, 4, 1)),
    ("SameSlot", "K4 middle accepted", (1, 4, 4, 8, 1)),
    ("SameSlot", "K5 accepted all", (1, 5, 1, 4, 1)),
    ("SameSlot", "K8 accepted first", (1, 8, 1, 1, 1)),
    ("CrossSlot", "K1 shared read", (2, 1, 1, 2, 3)),
    ("CrossSlot", "K2 shared read", (3, 2, 1, 4, 4)),
    ("CrossSlot", "K3 distinct nonzero slots", (2, 3, 2, 4, 5)),
    ("CrossSlot", "K4 distinct nonsequential slots", (3, 4, 2, 8, 6)),
    ("CrossSlot", "K5 shared read private writes", (4, 5, 1, 4, 5)),
    ("CrossSlot", "K8 shared read private writes", (2, 8, 4, 16, 3)),
    ("Batch", "BS3 K1", (3, 1, 1, 1, 4)),
    ("Batch", "BS4 K2", (4, 2, 1, 2, 5)),
    ("Batch", "BS8 K3", (8, 3, 1, 1, 9)),
    ("Batch", "BS4 K4", (4, 4, 2, 4, 5)),
    ("Batch", "BS3 K5", (3, 5, 4, 16, 4)),
    ("Batch", "BS4 K8", (4, 8, 2, 8, 5)),
]

BOUNDARY_VALUES = [
    ("accepted_first", "num_accepted_tokens = 1"),
    ("accepted_middle", "num_accepted_tokens = ceil((K+1)/2)"),
    ("accepted_all", "num_accepted_tokens = K+1"),
    ("same_slot", "read_state_indices == write_state_indices"),
    ("prefix_fork", "all batch rows share read slot; write slots are unique"),
    ("nonzero_slot", "read/write use nonzero and nonsequential slots"),
    ("zero_input", "qkv/z/a/b and states are all zero"),
    ("gate_extreme", "a/b contain -20, 0, 20"),
    ("nonzero_state", "Conv and SSM state use deterministic nonzero data"),
]
```

### 2.2 用例覆盖统计

下表是设计阶段的目标用例矩阵，不等同于当前 pytest 参数化后的 case 数。

| 类别 | Shape数量 | 边界值数量 | dtype数量 | 总用例数 |
|------|----------:|-----------:|----------:|-----------:|
| 常规形状 | 18 | 0 | 1 | 18 |
| 泛化形状 | 18 | 0 | 1 | 18 |
| 边界值 | 0 | 9 | 1 | 9 |
| **总计** | **36** | **9** | **1** | **45** |

---

## 3. 使用说明

### 生成测试数据示例

```python
def make_inputs(
    batch_size: int,
    speculative_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    num_state_slots: int,
    seed: int = 1234,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    sequence_length = speculative_tokens + 1
    head_dim = 128
    conv_dim = (2 * num_k_heads + num_v_heads) * head_dim
    conv_state_length = speculative_tokens + 3

    read_indices = torch.zeros(batch_size, dtype=torch.int32)
    write_indices = torch.arange(batch_size, dtype=torch.int32)
    write_indices.remainder_(num_state_slots)
    accepted = torch.arange(batch_size, dtype=torch.int32)
    accepted.remainder_(sequence_length).add_(1)

    return {
        "qkv": torch.randn(
            batch_size, sequence_length, conv_dim, dtype=torch.bfloat16
        ),
        "z": torch.randn(
            batch_size,
            sequence_length,
            num_v_heads,
            head_dim,
            dtype=torch.bfloat16,
        ),
        "a": torch.randn(
            batch_size, sequence_length, num_v_heads, dtype=torch.bfloat16
        ),
        "b": torch.randn(
            batch_size, sequence_length, num_v_heads, dtype=torch.bfloat16
        ),
        "conv_weight": torch.randn(4, conv_dim, dtype=torch.bfloat16),
        "conv_state": torch.randn(
            num_state_slots,
            conv_state_length,
            conv_dim,
            dtype=torch.bfloat16,
        ),
        "a_log": torch.randn(num_v_heads, dtype=torch.float32),
        "dt_bias": torch.randn(num_v_heads, dtype=torch.float32),
        "ssm_state": torch.randn(
            num_state_slots * sequence_length,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
        ),
        "read_state_indices": read_indices,
        "write_state_indices": write_indices,
        "num_accepted_tokens": accepted,
        "norm_weight": torch.randn(head_dim, dtype=torch.bfloat16),
    }
```

### 注意事项

1. BF16 输入必须先在 CPU 参考实现中升为 FP32，且在 Conv hand-off、
   recurrent readout 和最终输出处显式执行 BF16 RINT。
2. 测试 same-slot 时先保存输入 state 的 clone，避免原地写回污染 Golden。
3. Prefix fork 用例必须检查 shared read slot 的 Conv/SSM state 未变化。
4. 每个 batch 的 write slot 必须唯一；共享 read slot 是合法场景。
5. 大几何用例应串行执行，避免多个完整 SSM state 同时占用过多内存。

### 生产 ACLNN NPU 门禁

隔离 OPP 安装后，从 `xllm` Python 目录执行：

```bash
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
export MEGA_GDN_MTP_OPP_ROOT=/absolute/path/to/vendors/custom_xllm_math
export ASCEND_CUSTOM_OPP_PATH="${MEGA_GDN_MTP_OPP_ROOT}"
export LD_LIBRARY_PATH="${MEGA_GDN_MTP_OPP_ROOT}/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=.:..
export TMPDIR=/opt/wqy2/temp
export MEGA_GDN_MTP_ACLNN_LIB="${MEGA_GDN_MTP_OPP_ROOT}/op_api/lib/libcust_opapi.so"
/usr/local/python3.11.15/bin/python -m pytest -q \
  ../tests/core/kernels/npu/mega_gdn_mtp_decode_aclnn_test.py
```

当前 CPU Golden 为 27 例；TileLang/generated-source 为 17 例；PTO
地址/UB 合同为 10 例，三组合计 54 例。新增边界覆盖 read/write slot
负数与上界越界、跨 batch 行 read/write 冲突，以及 production PTO 的
57 个具名 UB 区域、174,112B 高水位和 175,360B Host reserve 联动。
生产 ACLNN NPU 门禁共 57 例：

| 类别 | 数量 |
| --- | ---: |
| identity/zero Conv | 1 |
| 六 K × in/out-place | 12 |
| 六 K × production heads | 6 |
| accepted/checkpoint × same/fork | 34 |
| K8 B4/B8 × same/fork | 4 |
| **合计** | **57** |

覆盖范围包括：

- K=1/2/3/4/5/8；
- accepted=1、中间、S；
- read/write same-slot 与 Prefix Cache fork-slot；
- Conv state、SSM state 的 in-place 写回；
- `conv_out`、private Conv state、全部 checkpoint 和最终 `out`。
