# `custom_ops_lib.mega_gdn_draft_decode`

```python
custom_ops_lib.mega_gdn_draft_decode(
    qkv,
    z,
    b,
    a,
    conv_weight,
    conv_state,
    a_log,
    dt_bias,
    ssm_state,
    read_state_indices,
    write_state_indices,
    q_cu_seq_lens,
    state_validity_mask,
    norm_weight,
    fla_ssm_state_layout=True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]
```

融合执行 MTP draft forward 中的 depthwise causal Conv、Q/K L2Norm、
GDN gate、recurrent state update 和 RMSNorm/Z gate。输入采用 packed-token
布局，同一次调用内每个逻辑请求可以包含 1 或 2 个 token。

算子支持 Conv/SSM State 异址读写；`read_state_indices` 指向共享前缀时，
`write_state_indices` 可以指向请求私有 slot，从而避免 forward 前的 State
H2H copy。测试 wrapper 返回全部中间结果；xLLM C++ wrapper 只返回最终
`out`，并原地更新 `conv_state` 和 `ssm_state`。

## 参数说明

- **qkv** (*Tensor*) – Conv 输入及 Q/K/V 拼接张量，BF16，形状为
  `(T, C)`。
- **z** (*Tensor*) – RMSNorm 后的 SiLU gate 输入，BF16，形状为
  `(T, Hv, 128)`。
- **b** (*Tensor*) – `beta = sigmoid(b)` 的输入，BF16，形状为
  `(T, Hv)`。
- **a** (*Tensor*) – decay gate 的动态输入，BF16，形状为 `(T, Hv)`。
- **conv_weight** (*Tensor*) – depthwise causal Conv 权重，BF16，形状为
  `(4, C)`。
- **conv_state** (*Tensor*) – Conv State cache，BF16，形状为
  `(S, 3, C)`；write slot 会被原地更新。
- **a_log** (*Tensor*) – decay 静态参数，FP32，形状为 `(Hv,)`。
- **dt_bias** (*Tensor*) – softplus bias，FP32，形状为 `(Hv,)`。
- **ssm_state** (*Tensor*) – recurrent State cache，FP32，形状为
  `(S, Hv, 128, 128)`；write slot 会被原地更新。
- **read_state_indices** (*Tensor*) – 每个逻辑请求的读 slot，INT32，形状为
  `(B,)`。
- **write_state_indices** (*Tensor*) – 每个逻辑请求的写 slot，INT32，形状为
  `(B,)`。
- **q_cu_seq_lens** (*Tensor*) – packed token 累计边界，INT32，形状为
  `(B + 1,)`。
- **state_validity_mask** (*Tensor*) – 初始 State 是否有效，BOOL，形状为
  `(B,)`；元素为 `False` 时忽略 read slot 并从零 State 开始。
- **norm_weight** (*Tensor*) – RMSNorm 权重，BF16，形状为 `(128,)`。
- **fla_ssm_state_layout** (*bool, optional*) – `True` 表示 State 使用 FLA
  `[K, V]` 布局，`False` 表示转置布局。默认值为 `True`。

## 支持的数据类型

- 激活、Conv 权重、Conv State 和 Norm 权重：`torch.bfloat16`
- SSM State、`a_log` 和 `dt_bias`：`torch.float32`
- State 索引和 packed 边界：`torch.int32`
- State 有效掩码：`torch.bool`

## Shape

记 `T` 为 packed token 总数，`B` 为逻辑请求数，`Hk` 为本 rank 的 key
head 数，`Hv` 为本 rank 的 value head 数，`D = 128`，
`C = (2 * Hk + Hv) * D`，`S` 为 State slot 数。

- **输入**：`qkv (T, C)`、`z (T, Hv, D)`、`a/b (T, Hv)`、
  `conv_weight (4, C)`、`conv_state (S, 3, C)`、
  `ssm_state (S, Hv, D, D)`。
- **元数据**：`read_state_indices/write_state_indices (B,)`、
  `q_cu_seq_lens (B + 1,)`、`state_validity_mask (B,)`。
- **输出**：`conv_out (T, C)`、原地更新后的 `conv_state (S, 3, C)`、
  原地更新后的 `ssm_state (S, Hv, D, D)`、`out (T, Hv, D)`。

## 约束条件

- `1 <= B <= 32`，每个逻辑请求的 sequence length 只能为 1 或 2，
  因而 `B <= T <= 2 * B`。
- `q_cu_seq_lens[0] == 0`，元素严格递增，最后一个元素等于 `T`。
- `1 <= Hk <= 16`，`Hk` 必须为 2 的幂；`Hv % Hk == 0`，且
  `1 <= Hv / Hk <= 4`。
- `0 <= read_state_indices[i], write_state_indices[i] < S`，`1 <= S <= 1024`。
- 同一请求可以原地读写同一 slot；不同并发请求不能对同一 write slot
  或仍被其他请求读取的 slot 产生冲突。
- 所有 Tensor 必须连续、位于同一 NPU device，且 dtype/shape 必须严格匹配。
- Conv、decay `g`、`beta` 和 recurrent readout 的 BF16 舍入点是数值合同，
  与小算子路径保持一致。

## 使用示例

```python
>>> import torch
>>> import torch_npu
>>> import custom_ops_lib
>>> device = "npu:0"
>>> B, T, Hk, Hv, D, S = 2, 3, 1, 2, 128, 3
>>> C = (2 * Hk + Hv) * D
>>> bf16 = lambda *shape: torch.randn(*shape, device=device, dtype=torch.bfloat16)
>>> qkv = bf16(T, C)
>>> z = bf16(T, Hv, D)
>>> b = bf16(T, Hv)
>>> a = bf16(T, Hv)
>>> conv_weight = bf16(4, C)
>>> conv_state = bf16(S, 3, C)
>>> a_log = torch.full((Hv,), -1.0, device=device, dtype=torch.float32)
>>> dt_bias = torch.zeros(Hv, device=device, dtype=torch.float32)
>>> ssm_state = torch.zeros(S, Hv, D, D, device=device, dtype=torch.float32)
>>> read_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
>>> write_indices = torch.tensor([0, 2], device=device, dtype=torch.int32)
>>> q_cu_seq_lens = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
>>> valid = torch.tensor([True, False], device=device, dtype=torch.bool)
>>> norm_weight = torch.ones(D, device=device, dtype=torch.bfloat16)
>>> conv_out, conv_state, ssm_state, out = custom_ops_lib.mega_gdn_draft_decode(
...     qkv=qkv,
...     z=z,
...     b=b,
...     a=a,
...     conv_weight=conv_weight,
...     conv_state=conv_state,
...     a_log=a_log,
...     dt_bias=dt_bias,
...     ssm_state=ssm_state,
...     read_state_indices=read_indices,
...     write_state_indices=write_indices,
...     q_cu_seq_lens=q_cu_seq_lens,
...     state_validity_mask=valid,
...     norm_weight=norm_weight,
...     fla_ssm_state_layout=True,
... )
>>> tuple(out.shape)
(3, 2, 128)
```

## 返回值

*tuple[Tensor, Tensor, Tensor, Tensor]* – 测试 wrapper 依次返回：

1. `conv_out`：BF16，形状为 `(T, C)`。
2. `conv_state`：原地更新后的 BF16 State cache，形状为 `(S, 3, C)`。
3. `ssm_state`：原地更新后的 FP32 State cache，形状为
   `(S, Hv, 128, 128)`。
4. `out`：最终 BF16 输出，形状为 `(T, Hv, 128)`。
