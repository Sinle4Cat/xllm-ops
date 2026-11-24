# MegaKdaPrefill

[English](../en/mega_kda_prefill.md) | 简体中文

Ascend910B 和 Ascend950（A5）分别选择对应内核。910B 要求有效的
`ffts_addr`，A5 接受 0。算子融合四 tap Conv/SiLU、Q/K L2 归一化、
向量 gate，以及 KKT、inverse、WY 和 chunk H/O。注册、构建及混合核
启动沿用 `mega_gdn_prefill_op` 的结构。Python 测试绑定在 910B 查询
FFTS 地址，在 A5 传入 0。

GLM模型已通过环境开关接入，默认仍为`fla_npu`组合路径。
`XLLM_KDA_KERNEL=mega_kda`为支持的输入启用原生prefill/普通decode，
默认`XLLM_KDA_PREFILL_BACKEND=native`。非MTP引擎可以显式选择
`XLLM_KDA_PREFILL_BACKEND=fla`，保留FLA prefill并使用原生decode。
在 Ascend910B 上，旧环境变量 `XLLM_KDA_DECODE_NORM=public` 选择 decode 的 mode 3 完整
公开兼容语义，包括 Conv 输入转换和 gate 的运算顺序。
默认值`native`选择mode0。public要求显式FLA-prefill配置，兼容配置要求
verify width=1。模型 MTP verify 在支持的布局上使用 decode mode 1，其余布局走 FLA V3。
完整差异见[decode合同](mega_kda_decode.md)。单算子测试不代表模型精度或性能验收。

## 输入输出

- BF16原始QKV `[T,3*H*128]`和原始gate `[T,H,128]`。
- beta是FP32 **sigmoid后的结果** `[T,H]`，不是原始投影。
- `a_log [H]`、`gate_bias [H,128]`为FP32，gate下界固定为 -5。
- Conv权重`[4,3*H*128]`，可选bias `[3*H*128]`，均为BF16。
- Conv输入/输出池分别为`[Ci,3,3*H*128]`、`[Co,3,3*H*128]` BF16。
- SSM输入/输出池分别为`[Si,H,V=128,K=128]`、`[So,H,128,128]` FP32。
- 输出`[T,H,128]`为BF16，三个输出均由调用者提前分配。
- `cu_seqlens [B+1]`和四组独立读写索引`[B]`都是Device int32。

读索引`-1`表示零初态，写索引`-1`表示不发布该类状态；空序列不写状态，
未写槽位保持原值。输出与任意输入及彼此必须没有存储别名。不保留原地更新路径。
同类状态池的有效写槽位必须唯一。索引和offsets每次执行从Device读取，
支持Graph replay改变其内容，但一次调用期间不可修改。

## 验证边界

两算子共用私有 OPP 打包及安装步骤。单独打包 Prefill 时给 `xllm_ops/build.sh`
传入 `--ops=mega_kda_prefill`，并选择对应 SoC。私有安装命令见
[Decode 验证边界](mega_kda_decode.md#验证边界)。

K/V head dimension 为 128，Conv width 为 4。Ascend910B 的所有支持
head 数都使用 64-token 分块。Ascend950 的通用路径使用 128-token 分块，
B=1、H=8、T=1–4 时走逐 token 递推。调用前将输入所在 NPU 设为当前设备。

910B 真机测试使用独立 CPU FP32 C64 标杆 `reference_910b(data)`，
保留 BF16 阶段边界。A5 使用 r003 标杆 `reference(data, a5=True)`，
按 A5 的 Conv 与归一化边界做 FP64 递推。两平台共用 Prefill 容差。
以 `abs(golden)=1e-3` 分组，正常值相对误差阈值为 1%，近零值绝对误差
阈值为 `1e-5`，每组超限比例均须不超过 0.1%。还检查 shape、dtype、
有限值、Conv state 和未写槽位的字节一致性。CPU 指标检查见
`test_mega_kda_accuracy_checks.py`。

当前共用 KDA pytest 清单含 805 个节点，覆盖 Prefill、Prefill carry500、
Decode、误差指标及 BF16 舍入。Prefill 保留以下范围。

- TP1/2/4/8/16/32/64 对应每卡 H=64/32/16/8/4/2/1，覆盖带或不带 Conv
  bias 的尾块、空序列及状态池边界。
- 常用长度为 1024、2048、4096、8192、16384、32768。保留组合剔除了
  TP1 的 2048 以及 TP8、TP16 的 8192。每个保留组合先与 CPU 标杆比较，
  再对固定输入执行 20 轮图回放，逐字节核对三个输出。
- TP1/2/4/16/32/64 执行固定输入及独立推进状态的 100 轮图回放。
  固定输入检查按原始字节比较，区分正零和负零。
- carry500 保留 gate bias 为 `-4.0` 的正常衰减场景，覆盖 H=1/8/64，
  CPU 与 NPU 独立推进状态，默认使用图模式。

这些用例在单卡验证本地 head 分片。多卡 TP、整网精度和性能另有验收口径。

torch_npu 2.9的NPUGraph每次回放前使用
两个Fill核更新RNG seed和offset；它们属于框架回放开销，不是本算子的
workspace或状态初始化，算子中不包含这类Fill实现。纯kernel耗时应单列，
完整图调用性能对比应计入框架开销。

只有CPU测试或编译通过不能宣称NPU精度通过。历史模型结果不能替代
本次接入的精度、性能或模型验收；正式性能收益另测。


## Conv 前端分块

两平台都把 BF16 Conv 输入与历史直接转换为 FP32，以四 tap 累加和
FP32 SiLU 计算后发布 BF16 激活。Q/K 归一化的 epsilon 为 `1e-6`，
结果以 BF16 发布。910B 对平方和加 epsilon 后乘 FP32 倒数平方根，
A5 保留 r003 的归一化实现，包括大值行的缩放保护。

910B Conv 前端使用 16、32 或 64 行的分块。序列前三个 token 沿用读取历史
状态的逐行路径，后续 token 使用分块路径，尾块只写有效行。gate 前缀
采用顺序 FP32 扫描，复用前端缓冲区前完成 MTE3 存储。MTP 的 Prefill
调用方传入最后三行 Conv 历史，并保留投机阶段所需的额外历史。
