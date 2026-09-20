# 原生 KV 状态池的 GDN prefill / decode / MTP

本分支增加 `MegaGdnNativeDecode` 和 `MegaGdnNativePrefill`，用于 Qwen3.6-27B
W8A8、TP4、MTP=3 的原生 KV SSM 接入。既有 compact 算子的公开接口不变。
验证硬件为 Ascend 910B3 / CANN 9.1.0；其他硬件没有本分支的验证结论。

最新模型数值修复与复测见 [modelmath_validation_20260919.md](modelmath_validation_20260919.md)。

## 接口合同

Python 绑定位于 `test/python_test/NativeOps.cpp`，模块名 `native_ops_lib`。

| 输入 | 合同 |
| --- | --- |
| QKV | BF16 `[T, (2*HK+HV)*128]`，通道连续；decode 直接支持非连续 token stride |
| Z | BF16 `[T, HV, 128]`，head 内连续；decode 直接支持投影切片 stride |
| A / B | BF16 `[T, HV]` |
| 卷积权重 | BF16 `[4, C]`，调用前准备好；不发布未执行的 capture-local 转置缓存 |
| conv bank | BF16 `[N, R, C]`，允许 slot 间 padding；MTP3 的 R=6 |
| SSM bank | FP32 `[N, HV, 128, 128]`，KV 轴顺序，允许 slot 间 padding |
| norm weight | BF16 `[128]` |
| state indices | decode 为 INT32 `[capacity, S]`，当前 MTP3 的 S=4；各 token ID 可以不连续 |
| query start loc | decode 为 INT32 `[capacity+1]`，每次 replay 更新，空请求用重复边界 |
| accepted | INT32 `[capacity]`，一基编号；读取 `indices[row, accepted[row]-1]` |

SSM bank 的 FP32 是原生 decode / prefill 的公开 ABI，不是调试产物。chunk GDN
内部仍按验证过的 ComputeT（当前为 FP16）计算，并在最终状态写回时提升到 FP32；
因此测试同时检查公共 dtype 和最终状态的 FP16 投影边界。

`capacity` 为 1–32，S 为 1–17，HK 为 1–16 的二次幂，HV/HK 为 1–4。
每个请求的本轮长度可以在 0–S 之间独立变化。卷积 bank 必须容纳 S+2 行，
即使当前行只有一个 token，也可能从上轮较后的 accepted 历史读取。

每个有效 token 写自己的真实 checkpoint ID；卷积历史写在该请求第一个 ID
对应的 conv slot。slot 0 保留，不写回。不同有效请求的写集合不能相交，也
不能覆盖另一请求仍需读取的初始状态。非法 decode 行会跳过并保持零输出；
调用者仍负责提交合法且无别名冲突的调度 metadata。

prefill 使用原有独立 conv/SSM read/write ID 向量，`-1` 表示无初始状态，
有效写 ID 必须大于 0。它只读取、更新最前 3 行卷积历史，保留额外 MTP 行，
并按原生 SSM stride 写最终状态。prefill 的 read ID 应由框架选择到正确的
初始 checkpoint；它不生成验证阶段的中间 checkpoint。

prefill warmup 可以有非空序列但 cache ID 为 0：此时仍须计算无初始状态的
卷积及输出，只禁止持久缓存写回。不能跳过卷积后让后续 GDN 读取未初始化
的 packed Q/K/V 工作区。测试必须在固定工作区内改变输入并重放，避免参考
算子偶然预填相同工作区而掩盖错误。

单序列 `ProcessSingle` 也必须在调用会写回的 `ProcessFnChunk` 前检查写槽位。
`0` 转换为 `-1` 后直接传给后者，会让 `WriteBackState` 使用负的物理 slot
偏移，改写 bank 前方内存。测试覆盖单序列 4-token 预热，分别提交 `0`、
`-1` 和上界外写索引，检查 bank 前后完整 slot 保护区以及所有持久状态。
多序列分支的保护不能替代这个快速分支的保护。

## 如何消除接入开销

decode 统一执行普通 S1、完整 MTP 和 ragged MTP。长度和 accepted 在设备端
读取，同一个 token bucket 可以从 B1×S4 重放为 B4×S1；不根据 capture 时的
Python request 数选择执行路径。不再同时运行 MegaMTP 和独立 fallback。

SSM / conv 使用一维 storage alias 加显式 slot stride。该 alias 不复制数据，
避免 ACLNN 自动连续化整个状态池；不分配 B×S 的 compact SSM，也不需要
框架 pack/scatter 状态。decode 的 QKV/Z 同样使用 storage alias 和真实行
stride，避免投影切片的连续化复制。少量输出初始化、卷积中间结果和 SDK
内部 workspace 仍存在；这不是零 workspace 实现。

SDK 不按 capacity 对 QKV、SSM 做填零。空行只做 metadata 检查后跳过。
框架仍可按 token 图 bucket 的最大可重放 request 数设置 capacity，例如
4-token 图使用 capacity 4；不能取 capture 时的 B，也不能简单使用 tokens/S。
SDK 支持 capacity32 的空行跳过，因此框架缩小 capacity 不是正确性前提。

prefill 继续使用 chunk/Cube 计算，仅改原生状态访问；没有把长 prefill 改成
逐 token decode。旧 compact prefill 的 checkpoint_stride 与 MTP 预留卷积行
关联，新接口固定逻辑 checkpoint_stride=1，保留已有最小历史精度修复。

新 decode 按模型框架的数值合同执行：衰减 gate `g` 保持 FP32，
`beta` 为 BF16，Q/K 在 L2Norm 后先舍入到 BF16，再转 FP32 进行 recurrent
计算；query scale 在该舍入后应用。readout 为 BF16，再进行 FP32 RMSNorm
和 SiLU gate，最终输出 BF16。S1 和 MTP 使用相同规则。

旧 compact MegaMTP 使用 BF16 `g` 和 FP32 normalized Q/K，不能作为新原生
decode 的模型精度 oracle。旧公开算子行为保持不变；新测试使用独立 CPU
模型数学参考，并在框架侧对照真实卷积、L2Norm、gating 和 recurrent 算子。

## 构建与加载

先加载 CANN、目标 Torch NPU / Python 环境并准备仓库依赖，然后运行：

```bash
bash scripts/build_native_gdn.sh /absolute/path/to/fresh-native-build
```

脚本只安装到指定私有目录，生成新算子以及供回归比较使用的旧 GDN 算子。
必须选择新的输出目录。上游增量构建的 kernel `.done` 标记、历史二进制和
算子索引可能保留旧版本，混合新 host / 旧 kernel 会产生无效结果。

启动新 Python 进程前，将生成的 `binding` 放入 PYTHONPATH，将
`install/vendors/custom_xllm_math` 放在 ASCEND_CUSTOM_OPP_PATH 最前，并把其
`op_api/lib` 放在 LD_LIBRARY_PATH 最前。保留框架自有算子 vendor 路径。
无需替换系统安装、驱动、模型或正在使用的 vendor。

```python
import native_ops_lib as native

output = native.mega_gdn_native_decode(
    qkv, z, b, a, conv_weights, conv_bank, a_log, dt_bias, ssm_bank,
    state_indices, query_start_loc, num_accepted_tokens, norm_weight,
)

output = native.mega_gdn_native_prefill(
    qkv, b, a, z, conv_weights, conv_bank, a_log, dt_bias,
    conv_read_ids, conv_write_ids, ssm_read_ids, ssm_write_ids, ssm_bank,
    mask_lower, mask_full, minus_identity, cu_seqlens, norm_weight, num_matrices,
)
```

## 验证与性能口径

```bash
python -m pytest -q test/python_test/test_mega_gdn_native.py test/python_test/test_native_model_contract.py
```

测试覆盖普通 decode、MTP、混合长度、随机 checkpoint、accepted 边界、
stride / padding、slot0、B32/S17、图重放改变请求数、400 次 checkpoint
重复检查、独立 CPU golden、prefill 最小历史（含 801 / 8192 token），以及 prefill→MTP 状态衔接。
checkpoint 与独立模型数学参考的检查维持 rtol=2e-5 / atol=2e-6。
另有隔离测试分别锁定 FP32 decay gate 和 BF16 normalized Q/K 的舍入边界；
修改前四项隔离测试全部失败，修改后通过。

`benchmark_native_gdn.py` 比较冻结的旧框架适配器与新原生接口。必须显式
提供带固定源码 SHA 的旧适配器归档：

```bash
python test/python_test/benchmark_native_gdn.py \
  --legacy-archive /path/to/pre-native-20260919.tar.gz \
  --output /path/to/fresh-perf.json --iterations 100 --profile
```

这是单卡、每层、每 rank 的图微基准；模型加载、编译和 capture 不计时。
prefill 微基准也使用图来测设备路径，不能替代当前服务的 eager prefill
端到端计时。profile 与性能计时分开，不用 profile 延迟计算收益。

完整模型 TP4/MTP3 的精度和吞吐必须在框架接入后另行验收。历史单次
CEval50 的预测翻转记录保留；baseline 自身存在重复运行变化，需结合固定
调度的重复对照判断。算子测试通过不能替代模型精度与性能验收。


### 旧 SDK oracle 的独立运行环境

旧测试套件的 standalone `CausalConv1d` 使用数组 metadata，框架的同名
算子使用张量 metadata，两者 ABI 不兼容。运行旧 SDK 的 unfused oracle
时，用 `build_native_gdn.sh /fresh/oracle-build --legacy-oracles` 额外构建
前者，且该测试进程的 vendor 路径只指向此私有包。服务接入和框架微基准
使用默认构建，保留框架自己的卷积 vendor。不要混用这两个 oracle 环境。
