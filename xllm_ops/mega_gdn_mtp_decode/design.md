<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0 -->

# MegaGdnMtpDecode 设计文档

## 1. 设计评审结论

### 1.1 结论

新增独立算子 `MegaGdnMtpDecode`，不向 `MegaGdnDecode` 注入 MTP
分支。两个算子的调度和状态合同不同：

- `MegaGdnDecode` 固定 `seq_len=1`、checkpoint stride 为 1，保留
  BS1 专用快路径。
- `MegaGdnMtpDecode` 固定 `seq_len=K+1`，从上一轮 accepted
  checkpoint 读取，并为本轮每个 token 写一份 SSM checkpoint。
- MTP Conv state 长度为 `K+3`，不是普通 decode 的 3。
- 首版 ABI 必须同时带 `read_state_indices` 和
  `write_state_indices`，使 Prefix Cache 首轮无需 H2H state copy。

TileLang 原型继续为 `K=1/2/3/4/5/8` 提供静态 specialization。生产 PTO
支持 `1 <= K <= 16`：六个常用 K 使用静态 key，`K=6/7/9..16` 保留动态
key 100 fallback；UB 不小于 182,816B 时，K=10～16 使用 deferred-Norm
key 210～216。`K=8/B=4/NK=8/NV=24` 进入独立 key 208 热点路径。

### 1.2 对原方案补充的强约束

1. `num_accepted_tokens` 表示上一轮接受的 target token 数，包含
   correction token，合法范围为 `[1, K+1]`。
2. SSM 读取 checkpoint：

   ```text
   read_ssm_idx =
       read_state_id * (K+1) + num_accepted_tokens - 1
   ```

3. SSM 写 checkpoint：

   ```text
   write_ssm_idx(token_idx) =
       write_state_id * (K+1) + token_idx
   token_idx ∈ [0, K]
   ```

4. Conv state 的槽内布局为 `[K+3, conv_dim]`。上一轮 state 中从
   `num_accepted_tokens-1` 开始的 3 行是本轮 token 0 的历史窗口。
5. 本轮写入的 Conv state 精确为：

   ```text
   write_conv_state[0:2] =
       read_conv_state[num_accepted_tokens:
                       num_accepted_tokens+2]
   write_conv_state[2:K+3] = qkv[0:K+1]
   ```

   因此下一轮再次从 `accepted-1` 开始读取时，窗口与拆分路径一致。
6. `read_state_id == write_state_id` 时，kernel 必须先完成当前 head
   初始 SSM state 的 GM→UB，再写 checkpoint 0；不能先覆盖后读取。
7. `read_state_id != write_state_id` 时，read slot 只读，所有 Conv/SSM
   写入只能落到 private write slot。

### 1.3 非目标

- 不融合 input/output projection MatMul。
- 不改变 BF16 舍入点、epsilon、softplus threshold 或 state 更新顺序。
- 不删除 `conv_out`、`conv_state_out`、`ssm_state_out` 观察点；这些输出
  用于 ATK 和逐段精度定位。
- 不把生产 PTO 的 K=1～16 支持反向扩展为 TileLang specialization；
  TileLang 仍只构建六种 K。

## 2. 算子接口

### 2.1 xLLM C++ 接口

```cpp
torch::Tensor mega_gdn_mtp_decode(
    const torch::Tensor& qkv,
    const torch::Tensor& z,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& conv_weight,
    torch::Tensor& conv_state,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    torch::Tensor& ssm_state,
    const torch::Tensor& read_state_indices,
    const torch::Tensor& write_state_indices,
    const torch::Tensor& num_accepted_tokens,
    const torch::Tensor& norm_weight);
```

wrapper 创建 `conv_out` 和 `out`，并将 `conv_state`、`ssm_state` 同时作为
ACLNN 输入和输出传入，以保持 Graph 捕获期间的 state Tensor 地址稳定。

### 2.2 ACLNN 逻辑接口

```cpp
MegaGdnMtpDecode(
    qkv,
    z,
    b,
    a,
    convWeight,
    convState,
    aLog,
    dtBias,
    ssmState,
    readStateIndices,
    writeStateIndices,
    numAcceptedTokens,
    normWeight,
    convOut,
    convStateOut,
    ssmStateOut,
    out);
```

### 2.3 参数与 Shape

记：

```text
S = K + 1
D = 128
C = (2 * NK + NV) * D
L = K + 3
N = num_state_slots
```

| 参数 | 方向 | dtype | Shape | 约束 |
| --- | --- | --- | --- | --- |
| `qkv` | 输入 | BF16 | `[B,S,C]` | contiguous |
| `z` | 输入 | BF16 | `[B,S,NV,D]` | contiguous |
| `b` | 输入 | BF16 | `[B,S,NV]` | contiguous |
| `a` | 输入 | BF16 | `[B,S,NV]` | contiguous |
| `convWeight` | 输入 | BF16 | `[4,C]` | width 固定 4 |
| `convState` | 输入/状态 | BF16 | `[N,L,C]` | `L=K+3` |
| `aLog` | 输入 | FP32 | `[NV]` | contiguous |
| `dtBias` | 输入 | FP32 | `[NV]` | contiguous |
| `ssmState` | 输入/状态 | FP32 | `[N*S,NV,D,D]` | checkpoint stride `S` |
| `readStateIndices` | 输入 | INT32 | `[B]` | 每项 `[0,N)` |
| `writeStateIndices` | 输入 | INT32 | `[B]` | 每项 `[0,N)` |
| `numAcceptedTokens` | 输入 | INT32 | `[B]` | 每项 `[1,S]` |
| `normWeight` | 输入 | BF16 | `[D]` | contiguous |
| `convOut` | 输出 | BF16 | `[B,S,C]` | 保留 Conv 舍入点 |
| `convStateOut` | 输出/状态 | BF16 | `[N,L,C]` | 与 `convState` alias |
| `ssmStateOut` | 输出/状态 | FP32 | `[N*S,NV,D,D]` | 与 `ssmState` alias |
| `out` | 输出 | BF16 | `[B,S,NV,D]` | 最终输出 |

Host 快路径支持：

- `1 <= B <= 32`；
- `D=128`；
- `1 <= NK <= 16` 且 NK 为 2 的幂；
- `NV % NK == 0` 且 `1 <= NV/NK <= 4`；
- `1 <= K <= 16`；
- 910B、910_93、950 平台。

TileLang PTO AOT family 仍只支持 `K=1/2/3/4/5/8`，当前只在 A3 默认构建中
启用；A2/A5 默认全量构建会过滤该 family，显式请求则 fail-closed。生产执行
路径不是这组六种 TileLang AOT object，而是 xLLM wrapper 调用 ACLNN 后进入
本目录的手写 production PTO OPP。TileLang AOT 目前承担 lowering、ABI、UB
与缓存门禁。

## 3. 数学与数值合同

### 3.1 每个 token 的计算

对 `t ∈ [0,K]`：

```text
conv_t = SiLU(
    W0 * history_t[0] +
    W1 * history_t[1] +
    W2 * history_t[2] +
    W3 * qkv_t)

q_hat_t = q_t / sqrt(sum(q_t*q_t) + 1e-6) / sqrt(128)
k_hat_t = k_t / sqrt(sum(k_t*k_t) + 1e-6)

g_fp32_t = -exp(a_log) * softplus(a_t + dt_bias, threshold=20)
g_t = FP32(BF16_RINT(g_fp32_t))
decay_t = exp(g_t)
beta_t = FP32(BF16_RINT(sigmoid(b_t)))

H_t = decay_t * H_(t-1)
pred_t = H_t^T * k_hat_t
delta_t = (v_t - pred_t) * beta_t
H_t = H_t + k_hat_t * delta_t^T
o_t = H_t^T * q_hat_t

out_t = RMSNorm(BF16_RINT(o_t), norm_weight, eps=1e-6)
        * SiLU(z_t)
```

其中 `H_-1` 从 read checkpoint 读取。每次得到 `H_t` 后立即写
`write_state_id*S+t`，但下一 token 继续消费 UB 中的 `H_t`，不从 GM
重读。

### 3.2 舍入合同

| 阶段 | 精度 |
| --- | --- |
| Conv 累加、SiLU | FP32 |
| Conv 到 Q/K/V hand-off | BF16 RINT |
| Q/K L2Norm | FP32 |
| gate `g` | FP32 计算，BF16 RINT 物化后回到 FP32 |
| `beta` | FP32 sigmoid，BF16 RINT 物化后回到 FP32 |
| decay、SSM state | FP32 |
| recurrent readout 到 Norm | BF16 RINT |
| RMSNorm 与 Z gate | FP32 |
| 最终输出 | BF16 RINT |

### 3.3 PTO/AscendC 调用序列

```cpp
// Conv channel owner，channel tile 固定 128。
TLOAD(history_bf16, read_conv_state + accepted_offset);
TLOAD(weight_bf16, conv_weight);
TCVT(history_fp32, history_bf16, CAST_NONE);
TCVT(weight_fp32, weight_bf16, CAST_NONE);
for (token_idx = 0; token_idx < S; ++token_idx) {
  TLOAD(x_bf16, qkv[token_idx]);
  TCVT(x_fp32, x_bf16, CAST_NONE);
  TMUL/TADD(conv_acc, history_fp32, weight_fp32);
  TSILU(conv_fp32, conv_acc);
  TCVT(conv_bf16, conv_fp32, CAST_RINT);
  TSTORE(conv_out[token_idx], conv_bf16);
  roll(history_fp32, x_fp32);
}
TSTORE(write_conv_state[0:2], selected_history_tail);
TSTORE(write_conv_state[2:2+S], qkv[0:S]);

SyncAllAiv();

// Recurrent head owner，一个 owner 完整处理一个 batch × value_head。
TLOAD(H, read_ssm_checkpoint);
for (token_idx = 0; token_idx < S; ++token_idx) {
  TLOAD(qkv/z/a/b for token_idx);
  TCVT(BF16 inputs, FP32, CAST_NONE);
  TMUL + TROWSUM + TRSQRT(q_hat, q);
  TMUL + TROWSUM + TRSQRT(k_hat, k);
  TEXP/TLOG(g_fp32, a, a_log, dt_bias);
  TCVT(g_bf16, g_fp32, CAST_RINT);
  TCVT(g_fp32, g_bf16, CAST_NONE);
  TEXP(decay, g_fp32);
  TEXP/TRECIP(beta_fp32, b);
  TCVT(beta_bf16, beta_fp32, CAST_RINT);
  TCVT(beta_fp32, beta_bf16, CAST_NONE);
  TMULS(H, H, decay);
  TROWEXPANDMUL(compute, H, k_hat);
  ColSum128(pred, compute);
  TSUB/TMULS(delta, v, pred, beta);
  OuterProductAdd128(H, k_hat, delta);
  TROWEXPANDMUL(compute, H, q_hat);
  ColSum128(readout, compute);
  TSTORE(write_ssm_checkpoint[token_idx], H);
  TCVT(readout_bf16, readout, CAST_RINT);
  RMSNorm + Silu(z) + Mul;
  TCVT(out_bf16, out_fp32, CAST_RINT);
  TSTORE(out[token_idx], out_bf16);
}
```

### 3.4 TileLang 到生产 PTO 的开发链

本算子遵循以下单向开发链，任何阶段都不能被后续阶段的“可编译”替代：

```text
Python Golden
  → TileLang PrimFunc 语义原型
  → TileLang PTO lowering
  → generated PTO source 审计
  → 六种 K 的 Bisheng AOT object
  → 生产手写 PTO 特化
  → OPP/ACLNN/xLLM
```

各阶段的职责和门禁如下：

| 阶段 | 主要产物 | 必须通过的门禁 |
| --- | --- | --- |
| Python Golden | 四个输出和 read/write state 合同 | 六种 K、accepted 边界、同槽/跨槽 |
| TileLang PrimFunc | 可读的公式、舍入点、状态副作用 | PrimFunc 构造、非法 K fail-closed |
| TileLang PTO lowering/JIT adapter | 经源码变换后可直接编译的 kernel | PrimFunc、lowering、source、ABI、UB 审计 |
| TileLang PTO AOT | 每个 specialization 的 `.cpp/.o`、manifest | ABI 一致、object 非空、缓存指纹覆盖 flags/include/header |
| generated source | 最终 PTO intrinsic、UB offset、同步和地址公式 | marker、指令计数、UB 高水位和 checkpoint 公式 |
| 生产 PTO/OPP | owner remap、专用 MatVec/OuterProduct、event | ACLNN NPU 四输出 Golden、Real ATK、连续状态、msprof |

单体 family 的 JIT 与 AOT 必须共用
`build_mega_gdn_mtp_decode_kernel()`；分段候选的三个 AOT family 必须分别
共用 `build_mega_gdn_mtp_{conv,recurrent,norm}_kernel()`。四条路径都复用
相同的数学常量、shape 校验和 source audit。PTO lowering 生成的 vector body
只有在 `__DAV_VEC__/__DAV_C220_VEC__` 下参与编译，并在第一条 Vector 指令
前执行 `set_mask_norm()` 和 `set_vector_mask(-1, -1)`。单体 AOT 在 launch
入口设置 `KERNEL_TYPE_MIX_AIC_1_2`；三个分段 AOT 不自行覆盖 task type，
由正式 runtime metadata 决定。pattern 不唯一或关键 marker 缺失时直接停止
构建，禁止带着部分替换继续编译。

当前 TileLang 版本 `0.1.4` 不能把跨 AIV 的 `T.sync_all` 降到 PTO，因此
单体 correctness-first 原型使用 `batch × key-head` owner：同一个 AIV
完成该 key-head 对应的 Conv、recurrent 和 Norm，避免跨 owner 的 GM
hand-off。分段 TileLang 候选则用 kernel launch boundary 代替跨 AIV
barrier，采用第 5.5 节的三种 owner。生产 PTO 使用第 5.3 节的
channel-owner → head-owner remap，并在 Conv 写回后执行显式 all-AIV
同步。三种调度必须在源码和性能报告中明确区分。

当前环境为 TileLang `0.1.4` + CANN `9.0.0`。分段 K=1 Conv 曾在同一进程
重复 launch 后非确定性报 `507015`（Vector UB address out of bounds）。
静态地址审计未发现越界，故障点位于首组 BF16→FP32 `TCVT`；对照
`MegaGdnDecode` 后发现 generated Vector body 没有初始化 Vector mask。
前序 kernel 遗留的 partial/count mask 可能让首组 TCVT 使用错误访问范围。
当前 source transform 已把 full-mask 初始化提升为三个分段 kernel 的强制
合同，并由 source test fail-closed。正式 AOT wrapper 已完成 K1 same-slot、
六 K 四输出和同进程 256 次重复 launch 验证；重复过程中穿插会修改 Vector
mask 的 Sigmoid，未再出现 `507015`。full-mask 根因和设备稳定性已闭环。

`-xasc` 手工 shared object 不能代替正式 runtime 注册。因此 JIT 只承担
lowering/source/ABI/UB 和单次隔离 smoke，连续精度与性能必须通过正式 AOT
family/registry/wrapper 路径验证，不能引用未注册 shared object 的输出。

## 4. 状态语义

### 4.1 SSM checkpoint

`ssmState` 的物理首维按 state slot 分组：

```text
slot 0: checkpoint 0 ... checkpoint K
slot 1: checkpoint 0 ... checkpoint K
...
```

普通 MTP：

```text
read_state_id == write_state_id
```

Prefix Cache 首轮：

```text
read_state_id  = shared_prefix_slot
write_state_id = private_request_slot
```

算子永远不写 `read_state_id`，除非它与 `write_state_id` 相同。Prefix
Cache 管理器必须在调用前给 private slot 建立所有权，在请求结束或驱逐时
释放，不能让两个活跃请求共享 write slot。

### 4.2 Conv 扩展窗口

Conv state 的状态机与现有 `CausalConv1d` spec update 路径一致。
例如 `K=4, S=5, L=7`：

```text
上一轮 state:
  [old0, old1, cand0, cand1, cand2, cand3, cand4]

上一轮 accepted=3:
  本轮读取 [cand1, cand2, cand3]

本轮写 state:
  [cand2, cand3, new0, new1, new2, new3, new4]
```

Conv 写入保存的是原始 `qkv` token，而不是 `convOut`。

### 4.3 alias 与重叠约束

- `convStateOut` 必须与 `convState` 使用同一 storage。
- `ssmStateOut` 必须与 `ssmState` 使用同一 storage。
- 同一 batch 内 `write_state_indices` 必须唯一，避免两个 owner 竞争写同
  slot。
- `read_state_indices` 可重复，允许多个请求读取同一个 shared prefix。
- 当某个 write slot 同时被其他 batch 项作为 read slot 时，Host 必须证明
  无跨请求覆盖；首版 wrapper 直接拒绝该重叠。

这里的 Host 指 xLLM 调用侧持有的 sequence-scoped host 元数据，不是
OPP tiling 回调。tiling 回调只校验 tensor 的 dtype、rank 和 shape，不能为
了读取 `readStateIndices`、`writeStateIndices` 或
`numAcceptedTokens` 的值引入 device-to-host 同步。值域、write 唯一性和
跨 batch 重叠必须在 batch builder/路由阶段用 host vector fail-closed
校验；kernel 只消费已校验的 contiguous INT32 device tensor。

## 5. Tiling 与调度

### 5.1 TilingData

```cpp
struct MegaGdnMtpDecodeTilingData {
  int64_t batch_size;
  int64_t sequence_length;  // K + 1，动态 key 100 使用
  int64_t num_k_heads;
  int64_t num_v_heads;
};
```

`K`、Conv state 长度、Conv tile 数和 checkpoint stride 均可由 tiling key
及上述 shape 字段推导，不重复放入 Host/device ABI。

### 5.2 Tiling key

| Key | K | 路径 |
| ---: | ---: | --- |
| 101 | 1 | `RunMtp<1>` |
| 102 | 2 | `RunMtp<2>` |
| 103 | 3 | `RunMtp<3>` |
| 104 | 4 | `RunMtp<4>` |
| 105 | 5 | `RunMtp<5>` |
| 108 | 8 | `RunMtp<8>` |
| 208 | 8 | `RunMtpQkGroupCache<8>`，仅 `B=4/NK=8/NV=24` |
| 210～216 | 10～16 | `RunMtp<K, deferred_norm=true>`，要求 UB≥182,816B |
| 100 | 6/7/9..16 | `RunMtpDynamic`；K10～16 在 UB 不足时回退该路径 |

key 208 按 `batch × key-head` 分配 32 个 AIV owner。每个 owner 只对 Q/K
执行一次 load、BF16→FP32 和 L2Norm，再将 token 1～8 的归一化结果缓存在
UB 中，供同组 3 个 value head 复用。该路径的 Q/K、readout 和 Z cache
extent 固定为 9 行，由 `kQkGroupCacheSequenceLength=9` 显式约束。其他 shape
继续使用原 key，不能把该特化泛化为动态 K 或动态 head 配置。

### 5.3 Block 级 Tiling

Conv owner 按 128 channel 切分：

```text
conv_tasks = C / 128
```

每个 Conv owner 在核内循环 `B*S`，避免为每个 token 重复加载四组 weight。

Recurrent owner 按 `batch × value_heads` 切分：

```text
recurrent_tasks = B * NV
task_count = max(conv_tasks, recurrent_tasks)
used_aiv = min(task_count, platform_aiv_count)
```

910B/910_93 使用 `CalcTschBlockDim`，950 使用 AIV-only block dim。
Host 设置 `schedule_mode=1`。Conv 写回后执行 MTE3 completion 和
`SyncAllAiv`，再切换到 recurrent owner。

### 5.4 UB 级 Tiling

固定 `head_dim=channel_tile=128`。K 通过时间循环执行，不按 K 复制
state/compute buffer，因此峰值 UB 与 K 基本无关。

| UB 区域 | 字节 | 生命周期 |
| --- | ---: | --- |
| Conv weight/history/token BF16+FP32 | 8,704 | Conv 阶段 |
| Q/K、gate scalar、归约 scratch | 10,752 | recurrent token 内 |
| 完整 FP32 state `128×128` | 65,536 | 整个 head 的 S 次 update |
| FP32 MatVec/OuterProduct compute | 65,536 | 每个 token 复用 |
| pred/delta/norm/z/gate/output | 7,680 | 每个 token 复用 |
| `ColSum128` scratch | 16,384 | 每次 MatVec 复用 |
| head scalar cache 与对齐余量 | 768 | head 间复用 |
| **Host reserve / 设计上限** | **175,360** | 小于 192 KiB |

对齐规则：

- BF16 vector：16 元素/32B；
- FP32 vector：8 元素/32B；
- state 和 compute 起始地址至少 512B 对齐。

本算子使用固定业务 tile，不用普通 elementwise 的线性
`tileLength = UB/bufferCoefficient`。为 codegen 明确记录：

```text
fixed_ub_bytes = 175360
bufferCoefficient(BF16 channel tile) = 0 bytes / extra K
```

即增加 K 只增加循环次数和 GM checkpoint 写回，不增加 UB tile 数。
Host 必须通过平台 API 获取 UB 大小；小于 Host reserve 的平台拒绝快路径。

TileLang correctness-first 原型在每个 AIV 上只保留 `128×64` 的 state
half；generated PTO source 的最大起始 offset 为 118,240B，计入最大的
32 KiB 临时 tile 后，当前 UB 高水位为 151,008B（约 147.5 KiB）。生产 PTO
按完整 `128×128` state 规划。原 key 的 57 个具名 UB 区域显式高水位为
174,112B，Host reserve 为 175,360B，余量 1,248B。key 208 使用
`[174112,183328)` 作为九行 Q/K group cache，`[183328,187936)` 作为九行
readout/Z cache，Host reserve 为 187,936B；A2/A3 PTO scratch 从 188,416B
开始，两者不重叠且仍低于 192 KiB。回归测试从 production header 解析每个
`TASSIGN` 的实际 tile footprint，校验区域不重叠、Q/K 连续布局、
`a_log/dt_bias` 各占 32B，并联动 Host reserve 与 A3 192 KiB 上限。源码审计
必须分别验证两条路径的最大 `TASSIGN` offset 加 tile 大小，不能把原型的
较低占用作为生产 PTO 的容量证明。

key 210～216 从 `kUbQkCacheTail=174112` 开始缓存 `K+1` 行 BF16 readout
和 Z。最重 K16 的末地址为 182,816B，Host 仅在平台 UB 不小于该值时下发；
否则继续使用 key 100。deferred-Norm 不缓存动态 K 的行数，模板通过
`static_assert` 限制为静态 K，并校验末地址不超过 192 KiB。

生产 PTO correctness baseline 使用单 Buffer 和显式 ready/free 事件：

```text
MTE2 load → set/wait MTE2→V
Vector compute → set/wait V→MTE3
MTE3 store → set/wait MTE3→MTE2/V
```

稳态循环不能用 `pipe_barrier(PIPE_ALL)` 代替跨流水 Buffer 所有权。只有
单 Buffer 的 ACLNN Golden 全部通过后，才允许逐机制引入 ping-pong 和
流水重叠。

实测仅 `K=8` 保留 recurrent 输入预取：当前 token 的 Q/K/V/a/b/z 全部转成
FP32 后，Vector 通过 `V→MTE2` free event 归还输入 BF16 Buffer，MTE2 再搬入
下一 token；下一轮仍以 `MTE2→V` ready event 为准。K=1/2/3/4/5 和动态 K
继续使用逐 token load/compute/store 的 serial 路径。该特化不改变 UB
分配，也不展开 token 循环。

第二轮仅对 `K=8` 的 Conv BF16 channel row 增加 input/output ping-pong。
input ping/pong 起始地址为 `3840/7936`，output ping/pong 为 `7680/8192`，
每个槽位 256B 且 32B 对齐。event 2/3 分别管理两个槽位的四条依赖：

```text
MTE3 → MTE2：input free
MTE2 → V：input ready
MTE3 → V：output free
V → MTE3：output ready
```

每个 Conv tile 结束时 drain 两个槽位的最终 free token，再进入复用 event 2/3
的后续阶段。该改动不改变 GM 地址、`TLOAD/TSTORE` 数量、FP32 计算顺序或
BF16 RINT 舍入点；新增 Buffer 位于原 Conv 预留区间内，生产 PTO 的 UB
显式高水位仍为 174,112B，Host reserve 仍为 175,360B。K=1/2/3/4/5 和
动态 K 继续走单 Buffer serial Conv。

### 5.5 分段 TileLang 调度候选

参考 `MegaGdnDecode` 的 owner remap，纯 TileLang 候选拆成三个 kernel：

| Stage | owner | B=1 逻辑任务 | 主要理由 |
| --- | --- | ---: | --- |
| Conv | `channel_tile` | 40 | 每个 owner 固定 128 channel，在 owner 内循环 B，四组 weight 只搬一次 |
| Recurrent/checkpoint | `batch × value_head × state_half` | 48 | `128×64` FP32 state half 连续驻留 UB，避免完整 state 加临时 Tile 超过 192 KiB |
| RMSNorm/Z | `batch × value_head` | 24 | 汇合左右 readout half 后按完整 128 维归约 |

三个 stage 之间通过 `conv_out` 和 BF16 `readout` 在 GM 交接。launch boundary
同时提供前一 stage 的完成与可见性，不依赖当前 TileLang PTO 后端不能表达的
`T.sync_all`。

当前 TileLang 表达能力下，三段是最小可执行拆分，而不是任意增加 kernel：

- 两段方案要求 Recurrent owner 持有完整 `128×128` FP32 state，并在同一
  owner 完成 Norm。完整 state 本身为 64 KiB；再加两个完整广播/计算 Tile
  会超过 192 KiB UB。
- 尝试只保留完整 state、对动态 half slice 执行 Tile op 时，TileLang 0.1.4
  lowering 报 `BufferLoad has no attribute access_ptr`。
- 因此 Recurrent 暂按左右 state half 拆 owner；Norm 必须单独汇合两个 half。

分段 family 对 K=1/2/3/4/5/8 各生成一个静态变体：共 18 个 AOT object，但
runtime 每次只按当前 K 各选一个 entry，固定下发 3 个 kernel，不会下发
18 个。保留单体 family 作为严格 A/B 与回退，不把两种 schedule 混在同一
device entry。

三个分段的 GM 交接只保留算法或 owner remap 必需的数据：

| 交接 | dtype | Tensor | 原因 |
| --- | --- | --- | --- |
| Conv → Recurrent | BF16 | `conv_out` | 保留原 Conv BF16 RINT 舍入点，并提供 V |
| Conv → Recurrent | FP32 | `qk_prepared` | Q/K 每个 key head 只归一化一次，避免每个 value head 重算 |
| Conv → Recurrent | FP32 | `gate_prepared` | decay/beta 每个 value head、token 只计算一次 |
| Recurrent → Norm | BF16 | `readout` | 保留 recurrent 到 Norm 的 BF16 RINT 舍入点，并汇合两个 state half |

上表描述 TileLang 分段原型，保持现状。生产 PTO 不通过 GM `gate_prepared`
交接 gate；它在 recurrent owner 的 UB 内分别对 `g` 和 `sigmoid(b)` 执行
BF16 RINT 再转回 FP32，以匹配 small-chain 舍入合同。两条路径不混合调优。

继续拆分 Q/K Normalize、gate 或 checkpoint store 会新增 launch 和 GM
读写，而不会获得新的 owner 并行维度；因此不作为当前 TileLang 方案。

K8 lowering 的 UB 高水位为：

| Stage | UB 高水位 |
| --- | ---: |
| Conv | 8,960 B |
| Recurrent | 141,376 B |
| Norm | 4,384 B |
| 单体 TileLang | 168,288 B |

Conv owner 已按参考实现改为只映射 40 个 channel tile，再在 owner 内循环
batch。Recurrent 与 Norm 暂保持逻辑 task 一 owner，由设备调度多波次；是否
引入 `T.Persistent` 必须在正式 AOT 精度通过后做独立 fresh A/B，不能仅凭
减少 launch block 数接受。

## 6. Workspace

- 使用 `platform.GetLibApiWorkSpaceSize()` 作为 system workspace。
- 算子不使用 workspace 保存中间 q/k/v、gate 或 state。
- checkpoint 必须直接写目标 GM state，因为 accepted count 在整次 target
  forward 结束后才确定。

## 7. Graph、元数据与 Prefix Cache

### 7.1 ModelInputParams

新增 sequence-scoped 元数据：

```cpp
std::vector<int32_t> linear_state_read_ids;
std::vector<int32_t> linear_state_write_ids;
torch::Tensor linear_state_read_indices;
torch::Tensor linear_state_write_indices;
```

兼容规则：

- 非 Prefix Cache 路径若只提供旧 `linear_state_ids`，read/write 都由其派生。
- `ModelInputParams::to()` 以 sequence-scoped Host IDs 和
  `num_accepted_tokens_host` 为权威来源；显式 fork 时重新派生 device
  tensor，不能沿用值不一致的旧 tensor。
- 同槽路径的 read/write tensor 直接 alias `linear_state_indices`，避免三次
  相同 H2D；`enable_graph` 必须在 `.to()` 后保持不变。
- MTP fused wrapper 不接受隐式 CPU state id，Graph/eager 调用前必须形成
  contiguous INT32 device tensor。

### 7.2 ACL Graph persistent 参数

Graph capture 为以下三个动态输入维护独立 persistent Tensor：

```text
persistent_linear_state_read_indices
persistent_linear_state_write_indices
persistent_num_accepted_tokens
```

replay 前更新实际 batch 范围。由于 kernel 会处理 capture shape 中的每一
行，padding 项不能使用负数或越界的“无效 slot”。每个 padding 行必须绑定
graph executor 预留的、互不相同的 sink write slot，并使用合法 read slot
和 `accepted=1`；sink slots 在 graph 生命周期内不得分配给真实请求。后续
如果 ABI 增加 `active_rows` mask，才可取消这些 sink slots。Graph 测试必须
执行：

```text
capture: read=A, write=B
replay:  read=C, write=D
```

并验证只有 D 更新，A/B/C 的非目标 slot 保持不变。

### 7.3 Prefix Cache 所有权

Prefix Cache 命中时：

1. shared prefix slot 只获得 read lease；
2. request private slot 获得 exclusive write lease；
3. 首轮 MTP 把 read/write id 同时传入算子；
4. 完成后请求后续轮次 read/write 都指向 private slot；
5. shared slot 无 H2H copy、无写入。

## 8. xLLM 路由

`Qwen3GatedDeltaNetBaseImpl::forward` 仅在以下条件使用新算子：

- `input_params.is_spec_verify`；
- dense same-length verify，`seq_len=K+1`；
- `K ∈ {1,2,3,4,5,8}`；
- checkpoint stride 等于 `seq_len`；
- FLA state layout；
- dtype、shape、contiguous 和 state id 约束全部通过；
- host accepted/slot/write ownership 校验通过；
- `input_params.enable_graph == false`；
- `XLLM_DISABLE_MEGA_GDN_MTP_DECODE` 未设置。

失败时完整回退到：

```text
CausalConv1d
+ QKV split/L2Norm
+ GDN gating
+ RecurrentGatedDeltaRule
+ Output Norm/Z gating
```

不允许部分执行新算子后再回退。

当前版本对 `K=1..16` 放行 eager dense verify。Graph 在独立 sink-slot owner、
capture/replay state 隔离和失败原子性完成前明确走旧路径，不能把已接入的
persistent tensor 字段表述为 fused Graph 已验收。Prefix Cache 的 shared
read/private write ABI 已具备，但 lease/回收 producer 尚未闭环。

对生产 PTO 支持的 K=1..16，eager 在任何 Conv/SSM 写入前校验 accepted、query length、
slot 范围、write 唯一性和跨行 read/write 冲突。只要
`read_state_id != write_state_id`，融合未命中就直接失败，不能静默进入只支持
same-slot 的旧 Conv/SSM fallback。

## 9. 精度与性能门禁

### 9.1 Golden 观察点

| 观察点 | 准入 |
| --- | --- |
| `conv_out` | BF16 bitwise |
| write slot 的 `conv_state` | BF16 bitwise |
| 每个 token 的 SSM checkpoint | `rtol=5e-3, atol=2e-6` |
| final `out` | `rtol=5e-3, atol=2e-2` |
| read/shared slot | bitwise 不变 |

### 9.2 用例维度

- 生产 PTO K：1～16；
- TileLang K：1、2、3、4、5、8；
- B：1、2、3、4、8；
- accepted：1、中间值、K+1；
- read/write：相同、不同、多个 batch 共享 read；
- state slot：非零、非顺序；
- eager 与 Graph capture/replay；
- 连续至少 256 轮 state 回灌；
- Prefix Cache 首轮和后续轮。

### 9.3 性能

纯 kernel 使用两级对比：

- 融合收益以拆分小算子链为 baseline，使用 msprof Task Duration；
- kernel 内调度候选以 serial fused kernel 为 baseline；
- 短 kernel event A/B 使用每个 K 独立 fresh 进程、5000 次 warmup、
  50 个 sample、每 sample 聚合 20 次 launch，并执行 fresh ABBA；
- 单次 launch event 只用于发现候选，不能作为保留优化的证据；
- 分别报告 K、B、checkpoint 写回带宽和最差 fresh pair；
- 不在设备数据缺失时声明 speedup。

模型 A/B 固定模型、TP、输入输出长度、并发、Graph 和 overlap。单体
production PTO 路径核对每个 GDN 层每次 target verify 只出现 1 个新
kernel call；分段 TileLang 路径核对固定出现 Conv/Recurrent/Norm 3 个
kernel call。两种路径都要求旧小算子链和 fallback 为零。

### 9.4 TileLang/generated PTO 门禁

- 六种 K 都能构造单体和三段 PrimFunc。单体 PrimFunc 为 18 个业务参数
  （13 输入、4 输出、`batch_size`），generated exported `call()` 再增加
  `stream`，共 19 个；分段 ABI 分别按 Conv、Recurrent、Norm 的最小输入输出
  集合独立审计。
- 六种 K 都生成 PTO `.cpp/.o`，manifest 的 target 为 `pto`，toolchain
  fingerprint 必须包含 `dav-c220`、编译 flags、`MEMORY_BASE`、include
  路径和 PTO header 内容哈希。
- generated source 必须包含 vector body guard、full-mask 初始化、
  `TROWEXPAND/TCOLSUM/TCOLEXPAND` 及对应 state/output handle。单体
  source 还必须包含 mixed task type；三个分段 source 不得自行覆盖 task
  type。
- 审计最大 UB offset、归约 scratch、Conv state 和 SSM checkpoint 地址；
  地址公式分别与第 4.1、4.2 节一致。
- JIT adapter 必须编译源码变换后的 source；未经相同 transform 的
  `@tilelang.jit(target="pto")` 结果不计入 lowering/source 门禁。JIT
  direct-library 重复执行存在第 3.4 节的 task-type 限制，不计入稳定性或
  性能门禁。
- AOT 编译通过只证明 lowering、ABI 与编译链成立，不等价于 NPU 数值
  通过；单体生产路径由 OPP/ACLNN 验证，分段路径由正式
  family/registry/runtime wrapper 验证。只有四个输出逐项对齐 Golden 后
  才能进入性能比较。
- 增量 OPP 构建的 `src_copy.done` 不跟踪 PTO header 变化；精度验收必须
  使用干净 build 目录，或明确失效 source-copy/kernel done stamp，并核对
  JSON 中的 object SHA256 已变化。

当前 AOT fingerprint 已覆盖真实 compile flags/include、Bisheng resolved
path/version/SHA256 和 TileLang/PTO header tree SHA256，并在 family 级复用
header fingerprint。JIT/AOT 使用同一经过校验的 bundled PTO include。

### 9.5 PTO/TileLang 优化闭环

性能优化遵循《PTO-ISA 优化手段指导书》和《TileLang 算子性能优化指导书》
的共同约束：先建立可重复的正确性/性能基线，再通过时间线判断 bound，每轮
只改变一个主要变量。

本算子的优化顺序固定为：

1. 保留当前单 Buffer + 完整 ready/free 事件链作为 correctness baseline。
2. 分别采集 Conv、recurrent、checkpoint store、Norm 的 Vector/MTE2/MTE3/
   Scalar 时间，不用总 kernel 时间猜测瓶颈。
3. 先尝试 Conv token load/store 的 ping-pong；生产者 ready 和消费者 free
   必须成对，不能用稳态 `PIPE_ALL` 屏障代替。
4. recurrent state 在 S 步内继续常驻 UB；不得为了 double buffer 重新从 GM
   读取 state。checkpoint 写回仅在证明 MTE3 气泡可被后续 token Vector
   计算覆盖时引入异步流水。
5. 生产 PTO 分别对 K=1～16、B=1/4/8 测量；TileLang 仍只测
   K=1/2/3/4/5/8。模板展开、流水深度和 core 数只对实测 shape bucket 生效，
   保留当前单 Buffer fallback。
6. TileLang `T.Pipelined` 只在安装版本能正确 lower 且资源审计通过时使用；
   当前跨 AIV hand-off 仍由生产 PTO 的显式 `SyncAllAiv` 表达。

每轮实验记录 source/object hash、SoC/CANN、shape/dtype、warmup/measured、
UB 高水位、流水利用率、中位/最小/最大延迟和四输出误差。CPU Simulator
只证明语义，CostModel 只用于候选排序，最终性能结论只来自目标 SoC。

### 9.6 当前调优结论

目标环境为 Ascend 910B3、CANN 9.0.0、BF16 输入/FP32 state、
`NK=8/NV=24`。第一轮保留 K8 recurrent 输入 MTE2 预取：

| 路径 | serial p50 | candidate p50 | 结论 |
| --- | ---: | ---: | --- |
| K8 A2/B1 | 51.306 us | 50.065 us | 快 2.42% |
| K8 A3/B2 | 51.185 us | 49.930 us | 快 2.45% |
| fresh final B3 vs A2/A3 | 51.185～51.306 us | 50.158 us | 快 2.01%～2.24% |

第一轮 fresh object 审计显示 K1/2/3/4/5 和动态 K 的 AIC/AIV 函数大小与
SHA256 均不变，只有 K8 AIV 从 5,548B 增至 5,852B。

第二轮在该 prefetch 版本上增加 K8 Conv BF16 input/output ping-pong。
正式短 kernel A/B 使用每个点独立进程、5000 warmup、50 samples、
20 launches/sample，并在同一物理卡执行 A-B-B-A：

| B | prefetch p50 A1/A2 | ping-pong p50 B1/B2 | 两组配对收益 |
| ---: | ---: | ---: | ---: |
| 1 | 51.273/50.014 us | 48.558/49.139 us | 5.30% / 1.75% |
| 4 | 126.759/126.737 us | 119.540/119.799 us | 5.70% / 5.47% |
| 8 | 208.260/207.422 us | 190.954/190.367 us | 8.31% / 8.22% |

同卡 `PipeUtilization` 的中位数进一步解释了收益：

| B | prefetch Task/AIV | ping-pong Task/AIV | Task/AIV 降幅 |
| ---: | ---: | ---: | ---: |
| 1 | 47.241/32.536 us | 46.561/31.193 us | 1.44% / 4.13% |
| 4 | 124.843/103.439 us | 117.883/96.473 us | 5.58% / 6.73% |
| 8 | 205.555/196.614 us | 188.094/178.925 us | 8.49% / 9.00% |

Vector 时间在三个 B 上基本不变；B8 MTE2 从 20.321us 降至 18.775us，
MTE3 从 35.353us 降至 34.765us，说明主要收益来自搬运/写回与 Vector 的
重叠，而不是减少数学计算。逐函数机器码审计显示 K1/2/3/4/5/动态路径及
K8 AIC 均不变，只有 K8 AIV 从 5,852B 增至 5,936B。最终 object SHA256 为
`8b65f1cbcf77eec63ec2d95175e0db92d4d43d48c2b2d1b2155c9c5e6683d927`，
并通过 39 项 Golden/generated-source 门禁和完整 57 项生产 ACLNN NPU
Golden。

下列候选不进入生产代码：

| 候选 | 结果 | 拒绝原因 |
| --- | --- | --- |
| 全 K token unroll | 单次 event 曾出现假阳性 | K1/K4 回退，聚合 ABBA 不稳定 |
| K8 recurrent-only unroll | 约回退 9.9% | p50 从约 51.3 us 增至约 56.4 us |
| K8 Conv+recurrent unroll | 约回退 9.9% | K8 AIV 膨胀到 31,928B |
| checkpoint store 与 Norm overlap | screen 正向 | 53 项 ACLNN 中 52 项 final out 失败 |
| all-K recurrent MTE2 prefetch | K1/K4 未稳定获益 | 只保留 K8 模板特化 |
| 全变量 int64 SSM 地址 | B1 回退 3.44%/3.55% | 只在 checkpoint×stride 前提升 |
| Q/K L2Norm Vector→Scalar 合并 | B1 最差回退 30.86% | 交接开销和调度退化 |
| K8 Conv FP32 history ring | B1 回退 0.73%/2.30% | 动态地址开销抵消 TMOV 删除 |
| key 208 checkpoint MTE3 free-wait 延后 | ACLNN 57/57 | exact matrix 的 state/out 非确定，存在真实 UB 生命周期竞态 |

这组结果说明短 kernel 必须聚合多次 launch；同时，移动 event wait 即使不改
数学公式也必须重新跑四输出 NPU Golden，不能只凭 UB 地址不重叠判断安全。

SSM offset 最终采用最小 64 位提升。Host 上界为 `N<=1024`、`S<=17`、
`NV<=64`：checkpoint、stride 和 head offset 各自安全地保留 int32，只有
`checkpoint * stride` 在乘法前显式 cast 到 int64。该版本通过 48 项
Golden/source/address 和 57 项 ACLNN，并消除了全变量 int64 的 B1 回退。

第四轮仅 K8 使用 `[2,128]` BF16 dynamic-stride TLOAD 合并 Q/K row load。
Q/K UB 连续且 32B 对齐，GM row stride 为 `num_k_heads*128` 个元素；TCVT、
Normalize 和 event 顺序不变。相对最小 int64 基线，B4 两组提升
1.06%/1.15%，B1 为 -0.10%/1.45%，B8 为 0.21%/0.09%。仅 K8 AIV 从
5,936B 降至 5,880B，object SHA256 为
`a5ed03e56582e00f68d9748c628c658b34c81fc8201f6b81a47cc5547b765457`，
因此该候选保留。

第五轮在第四轮基础上只为 `K8/B4/NK8/NV24` 增加 key 208，不改变旧 key
机器码。fresh object 审计确认 key 100/101/102/103/104/105/108 的 14 个
AIC/AIV 函数均与 production byte-identical；candidate object SHA256 为
`aa0cee40fb222a8f087898d6cfd92df1f00cf3c208e071c973cb717570e84cac`。

- 57 项 ACLNN 全部通过，覆盖 K=1/2/3/4/5/8；
- exact matrix 的 14 records × 4 outputs 与 production raw SHA 全部相同，
  same-slot/prefix-fork 重复运行均确定；
- K8/B4 fresh A-B-B-A 两组 p50 分别为
  `119.608→115.271 us` 和 `117.980→113.570 us`，下降
  3.63%/3.74%；中心估计 `118.794→114.421 us`，下降 3.68%；
- fallback K8/B1 p50 未回退，K8/B8 为 `190.017→190.266 us`
  （+0.13%，在噪声内）；
- 最后 120 条 profile 中 Task `116.872→111.382 us`（-4.70%）、
  AIV Vector `64.511→61.691 us`（-4.37%）、MTE2
  `12.296→10.245 us`（-16.68%），与减少重复 Q/K normalize/load 的
  因果假设一致。

该结果只证明纯算子 K8/B4 bucket，不外推为模型 MTP TPOT。

第六轮以已接受的 key 208 为基线，只移除 checkpoint 写回后的立即
MTE3→V free wait，尝试依赖 token-end EVENT_ID3 drain。机器码隔离确认仅
key 208 AIV 从 6,552B 变为 6,540B，其余 15 个函数逐字一致；57 项 ACLNN
也全部通过。但更严格的 14-record exact matrix 检出 21/56 个输出 fingerprint
变化和 36 个 CPU tolerance gate 失败。同一 seed 的 same-slot `ssm_state`
三次运行产生三个不同 SHA256，same-slot/prefix-fork 的 final out 也非确定。
这证明 ordered MTE3 提交不能代替 state source UB 的 free 依赖。

该候选在性能测试前拒绝，立即 EVENT_ID2 wait 已恢复，PTO header SHA256
回到 key208-r1 的
`0d37c1085c778148fc626875002b79448e9568552cdd53a55132f03bc714e642`。
当时 K8/B4 中心值为 114.421 us，距离 114 us 目标仅 0.421 us，低于
0.500 us 噪声线，因此在 key208-r1 checkpoint 暂停。后续第七轮 720ah
由新的 UB 生命周期假设重新开启。

第七轮 `candidate_720ah` 在安全 key 208 上保留每个 head 的九行 BF16
readout/Z，最后一个 checkpoint 完成后复用已结束生命周期的 state 区域，
一次完成九行精确 Norm；同时批处理九行 Q/K L2Norm。没有增加 GM hand-off
或跨核 barrier，checkpoint 的立即 free wait 保持不变。

- 正式无 profiling A-B-B-A 两组分别为
  `114.167→101.978 us` 和 `113.935→102.486 us`；
- 平均 `114.051→102.232 us`，延迟下降 **10.363%**；
- 14 个 same-slot/prefix-fork record 的四输出逐 bit 相同，重复运行确定；
- 最终 ACLNN suite 301/301 通过：181 项覆盖 K=1～16、accepted、
  same-slot/prefix-fork 和 32 项四输出逐 bit 确定性；另有 120 项覆盖
  Qwen3.5 全模型族 rank-local head 几何；
- 最后 120 条 profile 的 Task `111.382→100.892 us`（-9.418%）、
  AIV `90.8715→82.679 us`（-9.015%）、Vector
  `61.691→56.395 us`（-8.585%）、MTE3
  `17.3635→15.1475 us`（-12.762%）；
- 最终 object SHA256 为
  `72670fd6ac5e1ae331875d18ce07001330d509a2c66342431471fd059ead2b5b`。

K=1～16 泛化不增加静态 kernel：最终 key 仍为
`100/101/102/103/104/105/108/208`。逐函数机器码审计确认 key 100、101、
102、103、104、105、108 以及 key 208 AIC 与优化前逐字节一致，只有
key 208 AIV 从 6,552B 变为 8,308B。K=2 单次跨进程 -2.98% 因而定级为
DVFS/进程噪声，不是代码回退。正式 10.363% 声明只适用于
`K8/B4/NK8/NV24` 纯算子，不外推为 K>8 或模型 MTP TPOT。

第八轮为 K10～16 增加 deferred-Norm 静态模板。recurrent 循环只将每步
BF16 readout/Z 保存到 UB，最后统一执行 K+1 行 RMSNorm 和 Z gating，避免
每个 token 立即进入 Norm 的重复调度；K16 沿用已有 key 216，仅新增
key 210～215。

- Host 在 `K>=10 && UB>=182816B` 时选择 `200+K`，低 UB 平台回退动态
  key 100；最终 object SHA256 为
  `f56a00f1381cdf6bc15025e422004c3adaecb2900c832cd7116ef6e5d25fb197`；
- PTO 地址/tiling 静态门禁 22/22、定向 K10～16 ACLNN 74/74、原完整
  ACLNN 312/312 均通过；扩展 suite 832/832 通过，其中 Qwen3.5 四个
  head 几何组 × TP1/2/4/8/16 × K1～16 × same-slot/prefix-fork 为
  640/640；
- 逐函数机器码确认 key 100/101/102/103/104/105/108/208/216 的 18 个
  AIC/AIV 函数逐字不变，只新增 key 210～215；
- 最重 `NV64/B8/TP1` 工程 A-B-B-A 的所有 K 均超过 10%；K16 最弱为
  12.80%，K10～15 分别为 24.03%、19.21%、16.07%、15.13%、14.55%、
  13.19%；
- Qwen3.5 全几何矩阵覆盖四个模型组、TP1/2/4/8/16、B1/4/8 和 K1～16，
  共 960/960 点相对正确 seq17 小算子链达到至少 10%。最弱点为
  `122B/397B、TP1、B8、K16`：`1098.127→966.959 us`，下降 11.945%；
- 全矩阵口径为 500 warmup、30 samples、每 sample 聚合 10 次 launch 的
  NPU Event stream interval。公平性采集器仍不能解析本机 NPU 4 mapping，
  因此该结果定级为工程 `device_microbenchmark`，不能作为正式发布
  benchmark，也不外推为模型端 MTP TPOT。

全矩阵 JSON/TSV 和最重 geometry A-B-B-A 分别位于
`/opt/wqy2/temp/mega_gdn_mtp_pto_vs_small_chain_20260802/reports/` 下的
`candidate-k210-216-vs-small-seq17-qwen35-k1-16-full-matrix.{json,tsv}`
和 `formal-k210-216-abba-nv64-b8-k1-16.tsv`。

### 9.7 分段 TileLang 设备验收

2026-07-31 在 Ascend 910B3、CANN 9.0.0、BF16 输入/FP32 state、
`B=1/NK=8/NV=24` 上通过正式 family/registry/runtime wrapper 验收。

精度和稳定性结果：

- K=1/2/3/4/5/8 × accepted(first/mid/last) ×
  same-slot/prefix-fork，共 36/36 组通过；
- `conv_out` 和 write slot `conv_state` 为 BF16 bitwise；
- SSM checkpoint 最大绝对误差 `1.12e-8`，final out 最大绝对误差
  `2.44e-4`，均低于第 9.1 节门限；
- K1 同进程 256 次重复 launch、每次前插入 Sigmoid 污染 Vector mask，
  随后的 fresh 六组 K1 精度仍全部通过。

短算子 A/B 使用每个 K 独立 fresh 进程、5000 warmup、50 samples、
20 launches/sample，执行 `small → segmented → segmented → small`。
下表为两次 fresh run 的 mean 平均值；计时域是 NPU Event stream interval，
包含设备可见的 inter-op 下发间隙：

| K | 三段 TileLang | 小算子链 | 延迟降低 | ACLNN |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 165.79 us | 1236.30 us | 86.59% | 41.50 us |
| 2 | 221.02 us | 1251.77 us | 82.34% | 41.89 us |
| 3 | 271.82 us | 1149.42 us | 76.35% | 41.45 us |
| 4 | 324.79 us | 1040.04 us | 68.77% | 41.94 us |
| 5 | 379.59 us | 1085.91 us | 65.04% | 44.01 us |
| 8 | 541.64 us | 1035.44 us | 47.69% | 48.53 us |

Profiler 进一步分离纯 kernel duration，排除 host 和 device idle gap：

| K | 三段 TileLang | 小算子链 | ACLNN |
| ---: | ---: | ---: | ---: |
| 1 | 3 kernels / 157.50 us | 30 kernels / 95.15 us | 1 kernel / 18.99 us |
| 8 | 3 kernels / 534.39 us | 30 kernels / 159.62 us | 1 kernel / 46.70 us |

K1/K8 的 Recurrent 分别为 136.78/503.96 us，占三段纯 kernel 总时
86.84%/94.30%，剩余差距主要在 recurrent/checkpoint 路径，不在 Conv
或 Norm。结论必须按计时域分层：

- 精度和连续启动稳定性：通过；
- eager 相对当前小算子链的整链下发收益：六 K 均通过；
- 纯 kernel 相对小算子链、以及相对 ACLNN 的生产性能门禁：不通过；
- 分段 TileLang 保持 opt-in/fail-closed，不作为默认生产路径，也不以减少
  kernel 数量替代 Task Duration 结论。

本次公平性采集器不能解析本机 `npu-smi info -m` 表格，且宿主机
`load1=34.10`、swap 使用约 3.11 GiB，因此性能数据定级为
`device_microbenchmark`，不能作为正式 PR 百分比。原始 `npu-smi` 显示目标
卡 Health OK、AICore/AIVector 0%、HBM 使用率 5%；完整 artifact 位于
`/opt/wqy2/temp/mega_gdn_mtp_tilelang_segmented_acceptance_20260731`。

### 9.8 父仓 Release 构建与 C++ UT

2026-07-31 使用父仓现有
`build/cmake.linux-aarch64-cpython-311` Release tree 完成增量构建。
`npu_mega_gdn_mtp_decode.cpp`、Qwen3.5 GDN/decoder、MTP worker、ACL
Graph executor/persistent 参数均编译通过，以下目标成功链接：

- `mega_gdn_mtp_decode_wrapper_test`；
- `batch_test`；
- `acl_graph_executor_test`。

设备/Host C++ UT 共 55/55 通过：

- wrapper 3/3，覆盖 CPU 输入拒绝、NPU dispatch/state storage 和分段
  TileLang 对 ACLNN 的六种 K/state mode 一致性；
- batch 41/41，覆盖 K=1～16 route、forked state ownership、linear
  state ID 传输/canonicalization 和 Graph verify fail-closed；
- ACL Graph 11/11，覆盖 persistent 参数、capture/replay、不同 batch
  和 linear-only hybrid cache。

该结果证明父仓编译和现有 eager/fail-closed Graph 合同无回退，不表示
融合 PTO 已完成 Graph sink-slot owner 或 Prefix Cache lease 的生产验收。

## 10. 实现检查清单

- [x] 新增独立 `mega_gdn_mtp_decode` op definition、proto、tiling、kernel。
- [x] Python Golden → TileLang lowering → generated PTO → AOT object 功能链路可运行。
- [x] 六种 K 的 generated source ABI、UB、指令和地址公式审计通过。
- [x] 三个分段 family、18 个 AOT object 和正式 runtime wrapper/CMake 接入。
- [x] 三段 source 的 Vector full-mask 初始化静态门禁通过。
- [x] 六种 K 模板 key 和动态 K kernel 编译。
- [x] 生产 PTO、Host wrapper 和 Qwen eager route 支持 K=1～16；动态 K
      保留 key 100，K10～16 在 UB 充足时使用 key 210～216。
- [x] Host wrapper 校验静态 shape、dtype、contiguous；动态 accepted/slot
      值域由 engine 元数据所有者保证，不触发 D2H。
- [x] PTO kernel 按 head 将初始 state 只加载一次。
- [x] 每个 token 都写 SSM checkpoint，下一 token 不从 GM 重读。
- [x] Conv 从 read slot 读取，扩展窗口写 write slot。
- [x] ACLNN wrapper 与 xLLM API 接入代码。
- [x] Qwen3.5 `is_spec_verify` 路由与环境变量回退代码。
- [x] read/write state metadata、eager host gate 和 Graph persistent 字段接入。
- [x] Graph fused route 在 sink owner 完成前 fail-closed。
- [ ] Graph padding sink slot 所有权和 capture/replay 状态隔离验收。
- [ ] Prefix Cache read/write lease 接入。
- [x] CPU Golden、六 K source/AOT、生产 OPP/ACLNN 设备 Golden 通过。
- [x] K8 kernel 内部 prefetch/ping-pong fresh ABBA 与同卡 msprof 门禁通过。
- [x] K8/B4 key 208 Q/K group cache 通过机器码隔离、57 项 ACLNN、
      四输出 bitwise、fallback 与 fresh A-B-B-A/profile 门禁。
- [x] candidate 720ah 达成 K8/B4 正式 A-B-B-A 10.363%，K=1～16 与
      Qwen3.5 head 几何 ACLNN 301/301、逐函数机器码隔离通过。
- [x] deferred-Norm key 210～216 通过 ACLNN 832/832、旧 key 机器码隔离；
      Qwen3.5 K1～16 全几何工程矩阵 960/960 点相对小算子链超过 10%。
- [x] 父仓 Release 三个相关目标构建通过，wrapper/batch/ACL Graph C++
      UT 共 55/55 通过。
- [x] key 208 checkpoint wait-overlap 完成设备精度验证并因非确定 state/out
      拒绝；安全 EVENT_ID2 wait 已恢复。
- [x] AOT cache fingerprint 覆盖编译 flags、include 路径和 PTO header 内容。
- [ ] Real ATK、Graph、Prefix Cache、拆分链/模型精度和模型性能门禁通过。
- [x] 分段 K1 same-slot、六 K 四输出、重复 launch 和 fresh A/B 已执行。
- [ ] 分段 TileLang 纯 kernel/ACLNN 性能门禁通过并允许默认启用。

## 11. 参考实现

- `xllm_ops/mega_gdn_decode`：单 token PTO kernel、Host tiling 和 ABI。
- `xllm_ops/causal_conv1d`：MTP Conv 扩展窗口与 accepted offset。
- `xllm_ops/recurrent_gated_delta_rule`：MTP checkpoint 选择和写回。
- `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp`：
  `is_spec_verify` 的 K gate、fused route 和完整 split fallback。
- `/opt/wqy2/temp/llm-generated-kernel-blog/llm-generated-kernel-inference-framework.md`
  （SHA256
  `bb5d4234584ca1896be92ecffe2d2c5b09f6d02fec2535de69d15b4c305e263e`）：
  Golden、owner remap、PTO source、ACLNN/Graph 和分层验证方法。
- 《PTO-ISA 优化手段指导书》：Tile/Buffer、ready/free 事件、流水与硬件
  profiling 门禁。
- 《TileLang 算子性能优化指导书》：正确性/性能基线、bound 分类、单变量
  实验和 shape 分桶方法。
