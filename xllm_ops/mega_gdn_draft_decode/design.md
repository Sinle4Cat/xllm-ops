# MegaGdnDraftDecode 设计文档

## 1. 算子目标与边界

`MegaGdnDraftDecode` 是 MTP draft forward 的 PTO 融合算子，用一次调用替代：

```text
CausalConv1dUpdate
→ Q/K/V split + L2Norm
→ Sigmoid / Softplus / Decay
→ Recurrent GDN
→ RMSNorm + Z gating
```

算子接收 packed token，每个逻辑请求在一次 draft forward 中处理 1 或 2 个
token。MTP 的 `K` 表示 draft forward 的迭代次数，不是本算子的单次 sequence
length，因此不按 `K=1..16` 生成多个 kernel。

与 `MegaGdnMtpDecode` 的区别：

- Draft 只写最终 Conv/SSM State，不写每 token checkpoint。
- Draft Conv State 始终是 3 行历史，不使用 `3+K` 扩展窗口。
- Draft 使用 `q_cu_seq_lens` 表示 packed 请求边界，支持同一批次中 1/2
  token 混合。
- Draft 使用 `state_validity_mask` 区分有效 cache 和 fresh state。

## 2. 算子接口

### 2.1 ACLNN ABI

```cpp
aclnnMegaGdnDraftDecode(
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
    fla_ssm_state_layout,
    conv_out,
    conv_state_out,
    ssm_state_out,
    out);
```

### 2.2 参数与 shape

记：

```text
T  = packed token 总数
B  = 逻辑请求数
Hk = local key head 数
Hv = local value head 数
D  = 128
C  = (2 * Hk + Hv) * D
S  = State slot 数
```

| 参数 | I/O | dtype | shape | 语义 |
|---|---|---|---|---|
| `qkv` | 输入 | BF16 | `[T, C]` | Conv 输入，Q/K/V 按 channel 拼接 |
| `z` | 输入 | BF16 | `[T, Hv, D]` | Norm 后 SiLU gate |
| `b` | 输入 | BF16 | `[T, Hv]` | `beta = sigmoid(b)` 的输入 |
| `a` | 输入 | BF16 | `[T, Hv]` | decay gate 的动态输入 |
| `conv_weight` | 输入 | BF16 | `[4, C]` | depthwise causal Conv 权重 |
| `conv_state` | 输入 | BF16 | `[S, 3, C]` | 按 slot 存储的 Conv 历史 |
| `a_log` | 输入 | FP32 | `[Hv]` | decay 静态参数 |
| `dt_bias` | 输入 | FP32 | `[Hv]` | softplus bias |
| `ssm_state` | 输入 | FP32 | `[S, Hv, D, D]` | 按 slot/head 存储的 recurrent State |
| `read_state_indices` | 输入 | INT32 | `[B]` | 每个请求的读 slot |
| `write_state_indices` | 输入 | INT32 | `[B]` | 每个请求的写 slot |
| `q_cu_seq_lens` | 输入 | INT32 | `[B+1]` | packed token 的累计边界 |
| `state_validity_mask` | 输入 | BOOL | `[B]` | false 表示忽略 read slot 并从零 State 开始 |
| `norm_weight` | 输入 | BF16 | `[D]` | RMSNorm 权重 |
| `fla_ssm_state_layout` | 属性 | BOOL | scalar | true 为 FLA `[K,V]`，false 为转置布局 |
| `conv_out` | 输出 | BF16 | `[T, C]` | Conv+SiLU 的 BF16 hand-off |
| `conv_state_out` | 输出 | BF16 | `[S, 3, C]` | 写 slot 的最终 Conv State |
| `ssm_state_out` | 输出 | FP32 | `[S, Hv, D, D]` | 写 slot 的最终 SSM State |
| `out` | 输出 | BF16 | `[T, Hv, D]` | 融合 Norm/Z gate 输出 |

xLLM wrapper 将 `conv_state_out` 与 `conv_state` 原地 alias，将 `ssm_state_out` 与
`ssm_state` 原地 alias。Kernel 先将当前请求的初始 State 搬入 UB，再写目标
slot，因此 `read_state_indices[i] == write_state_indices[i]` 安全。若使用非 alias
输出，调用方必须先保证未写 slot 的输入内容已保留到输出。

### 2.3 约束

- `1 <= B <= 32`，`T = sum(q_cu_seq_lens[i+1]-q_cu_seq_lens[i])`。
- 每个 sequence length 只能是 1 或 2，因此 `B <= T <= 2*B`。
- `q_cu_seq_lens[0] == 0`，严格单调，`q_cu_seq_lens[B] == T`。
- `1 <= Hk <= 16`，`Hk` 是 2 的幂，`Hv % Hk == 0`，`1 <= Hv/Hk <= 4`。
- `0 <= read_state_indices[i], write_state_indices[i] < S`，`1 <= S <= 1024`。
- 不同逻辑请求并发更新；一个请求的 write slot 不能与另一请求的
  read/write slot alias。同一请求内的 read/write 可相同。
- 所有 tensor 必须连续、位于同一 NPU device。
- Kernel 不从 GM 中主动校验 index 值和 `q_cu_seq_lens` 内容；这些是调用合同。

## 3. 计算逻辑

### 3.1 Conv 和状态窗口

对请求 `i` 和 channel `c`：

```text
h0, h1, h2 = conv_state[read_slot, :, c]  or  0 when state is invalid
for t in q_cu_seq_lens[i] .. q_cu_seq_lens[i+1]-1:
    acc = w0[c]*h0 + w1[c]*h1 + w2[c]*h2 + w3[c]*qkv[t,c]
    conv_out[t,c] = BF16(acc * sigmoid(acc))
    h0, h1, h2 = h1, h2, FP32(qkv[t,c])
conv_state_out[write_slot,:,c] = BF16(h0,h1,h2)
```

`conv_out` 必须在 Q/K/V split 之前落为 BF16，与小算子路径的 tensor
hand-off 保持一致。

### 3.2 Q/K/V、gate 和 recurrent update

```text
q_raw, k_raw, v = split(conv_out[t])
q = q_raw / sqrt(sum(q_raw^2) + 1e-6) / sqrt(128)
k = k_raw / sqrt(sum(k_raw^2) + 1e-6)

g = FP32(BF16(-exp(a_log[h]) * softplus(FP32(a[t,h]) + dt_bias[h])))
beta = FP32(BF16(sigmoid(FP32(b[t,h]))))

state = state * exp(g)
prediction = state^T * k
delta = (v - prediction) * beta
state = state + outer(k, delta)
readout = state^T * q
```

`Hv/Hk` 个 value head 共享同一个 Q/K head。非 FLA 布局在计算时按转置语义
解释 State，写回仍保持输入布局。

### 3.3 RMSNorm 和 Z gate

```text
norm_input = FP32(BF16(readout))
rms_inv = rsqrt(mean(norm_input^2) + 1e-6)
out[t,h] = BF16(norm_input * rms_inv * norm_weight * silu(FP32(z[t,h])))
```

四个数值合同点不得被后续调优删除：

1. Conv 输出到 Q/K/V 之间的 BF16 舍入。
2. decay `g` 在 `exp(g)` 前的 BF16 舍入。
3. `beta` 在 recurrent update 前的 BF16 舍入。
4. recurrent `readout` 到 RMSNorm 之间的 BF16 舍入。

## 4. PTO Kernel 设计

### 4.1 执行阶段

```text
Phase 1: Conv channel tiles
  task = channel_tile, tile width = 128
  每个 task 串行走过 B 个请求及各自 1/2 token

Global AIV barrier

Phase 2: Recurrent heads
  task = batch_idx * Hv + head_idx
  初始 State 只从 GM 加载一次
  在 UB 中连续更新 1/2 token
  每 token 输出 Norm 结果，最后只写一份 State
```

A2/A3 使用 mixed block 中的两个 AIV；A5 使用 AIV-only block。Host 端从
platform API 读取 AIC/AIV 数量，实际使用核数为：

```text
task_count = max(C / 128, B * Hv)
used_aiv_cores = min(task_count, physical_aiv_cores)
```

核内使用 `task += vector_core_count` 的 grid-stride loop，不依赖写死核数。

### 4.2 Tiling key

| key | SSM 布局 | 平台 |
|---:|---|---|
| `1` | FLA | A2/A3/A5 |
| `1001` | non-FLA | A2/A3/A5 |

Draft sequence length 是动态 1/2，由 `q_cu_seq_lens` 决定，不增加按 MTP K
特化的 tiling key。

### 4.3 Tiling data

```cpp
struct MegaGdnDraftDecodeTilingData {
  int64_t total_tokens;
  int64_t batch_size;
  int64_t num_k_heads;
  int64_t num_v_heads;
};
```

shape 约束由 Host tiling 校验，UB 容量和核数通过 `PlatformAscendC` 读取。

### 4.4 UB 分配和生存期

本算子是固定 `D=128` 的融合 recurrent kernel，不是按可变 `tileLength`
线性伸缩的 elementwise kernel。因此线性 `bufferCoefficient` 不适用；等价的
UB 合同是每 AIV 固定上限 `175360 B`，Host 端对实际 UB 大小做门禁。

| 区域 | dtype/shape | byte range | 生存阶段 |
|---|---|---:|---|
| Conv weights/history/input/output | BF16/FP32, `[* ,128]` | `[0, 8448)` | Conv；同一 channel tile 内复用 |
| Q/K/scalars + L2 scratch | BF16/FP32 | `[8704, 19200)` | 单 recurrent token |
| V BF16 | `BF16[128]` | `[19200, 19456)` | 单 recurrent token |
| State | `FP32[128,128]` | `[19456, 84992)` | 一个 head 的全部 1/2 token |
| V FP32 | `FP32[128]` | `[84992, 85504)` | 单 recurrent token |
| State compute | `FP32[128,128]` | `[85504, 151040)` | projection/update 临时区 |
| prediction/delta | `FP32[128]` 各一份 | `[151040, 152064)` | 单 recurrent token |
| Norm BF16/FP32 buffers | `[* ,128]` | `[152064, 156448)` | 单 Norm token |
| column reduction scratch | `FP32[32,128]` | `[156672, 173056)` | recurrent readout |
| `a_log` / `dt_bias` static | `FP32[8]` 各一份 | `[173824, 174112)` | 一个 head |
| 安全对齐余量 | - | `[174112, 175360)` | Host 端容量门禁 |

所有 offset 均 32 B 对齐。State 和 State compute 各为 64 KiB，是 UB 占用的
主要部分。

## 5. 状态语义与 Prefix Cache

### 5.1 普通 draft

```text
read_state_indices[i] == write_state_indices[i]
```

初始 Conv/SSM State 先进 UB，之后才原地写回，不需要额外 H2H copy。

### 5.2 Prefix fork

```text
read_state_indices[i]  = shared_prefix_slot
write_state_indices[i] = private_request_slot
```

算子只写 private slot，shared prefix slot 保持不变。Conv 和 SSM 使用同一组
read/write metadata，避免 forward 之前的两次 H2H copy 及相关 stream/event 同步。

### 5.3 Fresh state

`state_validity_mask[i] == false` 时，read slot 仅用作安全 GM 地址，Conv/SSM 初值
在 UB 中清零，最终 State 写入 write slot。

## 6. Workspace 和同步

- ACLNN workspace 大小使用 `platform.GetLibApiWorkSpaceSize()`。
- Kernel 本身不使用自定义 GM workspace。
- GM→UB、Vector、UB→GM 通过明确 event 保证数据依赖。
- Conv 与 recurrent 的 ownership 不同，两阶段之间使用全 AIV barrier，确保
  `conv_out` BF16 GM hand-off 可见。

## 7. 性能设计

- 融合 Conv/QKV/gating/recurrent/Norm，减少中间 tensor 和 kernel launch。
- 每 head 初始 State 只读一次，1/2 token 在 UB 内连续更新。
- 每 head 只写最终 State，不写 verify 路径所需的逐 token checkpoint。
- Conv 按 128 channel 分块，recurrent 按 `B*Hv` 并行，两阶段分别使用合适粒度。
- 异址读写在 Prefix Cache 首轮可消除 Conv/SSM H2H copy，但需模型端 A/B
  才能量化收益。

## 8. 实现和验收清单

- [x] op definition、shape inference、tiling data 和 PTO kernel。
- [x] FLA/non-FLA tiling key。
- [x] A2/A3/A5 构建配置，核数从 platform API 获取。
- [x] Conv/SSM 同 slot 原地安全、Prefix Cache 异 slot 写回、fresh state。
- [x] BF16 hand-off/g/beta/readout 四个数值合同点。
- [x] xLLM eager 调用和 draft packed metadata。
- [x] A2 设备精度用例全通过（40/40）。
- [ ] A5 编译和设备精度。
- [ ] xLLM eager/ACL Graph/Prefix Cache 模型冒烟。
- [ ] 小算子对比的 msprof 性能报告。

## 9. 参考实现

- `mega_gdn_decode/op_kernel/mega_gdn_decode_pto_kernel.h`：数值对齐的 Conv、GDN
  和 Norm PTO primitive。
- `mega_gdn_mtp_decode/op_kernel/mega_gdn_mtp_decode_pto_kernel.h`：状态布局、
  recurrent 步骤和 UB layout。
- `test/python_test/test_mega_gdn_draft_decode.py`：PyTorch 小算子标杆与可执行精度用例。
