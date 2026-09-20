# xllm-ops 原生 MTP 算子验证记录

本记录对应 `adn_gdn` 发布候选，验证硬件为 Ascend 910B3 / CANN 9.1。
源码发布只包含源文件、测试和文档；私有构建目录、安装包、二进制和日志不属于
发布内容。SSM cache 的公开输入/输出 dtype 固定为 FP32，内部 ComputeT 的舍入
边界见 [mega_gdn_native.md](mega_gdn_native.md)。

## 已实现

- `MegaGdnNativeDecode`：普通 decode、完整 MTP 和 ragged MTP 统一设备端调度；每轮可更新请求数、长度、accepted 和 checkpoint ID。
- 直接读取和回写原生 conv / FP32 KV SSM 状态池，支持 slot stride 与非连续 checkpoint；保护 slot 0 和 padding。
- decode 直接读取带 stride 的投影 QKV / Z，不再连续化复制；空行跳过，不执行 capacity32 的虚假状态计算。
- `MegaGdnNativePrefill`：沿用 chunk/Cube 核心，支持原生状态池及独立读写 slot；逻辑卷积历史固定 3 行，保留 MTP 额外行。
- 保留旧 API，提供独立绑定、构建脚本、接口文档、正确性测试与冻结旧适配器的性能对比脚本。

原来的三项接入开销已在 SDK 路径消除：capacity padding 的无效计算、compact SSM 的 pack/scatter、同一图中 SDK 与 fallback 重复执行。无需 compact SSM，原 B32/S4/Hv12/K=V128 的 96 MiB 临时状态变为 0；SDK 仍有其他 workspace。

## 验证

- 原生接口 33 项通过：包括 400 次图重放逐 checkpoint 检查、独立 CPU golden、随机槽位、非连续存储、B32/S17、true T1、图逆序重放、非法 decode metadata、slot0，以及 prefill→MTP。
- prefill 覆盖 3、128、512、801、8192 token 与无初始 / 原地 / 异地状态模式；与最小历史 compact oracle 的输出及状态逐位一致。
- SDK checkpoint 检查保持 rtol=2e-5、atol=2e-6。独立 CPU golden 使用既有 SDK 测试容差。
- 旧 SDK 回归：54 项通过，覆盖 prefill 状态模式、1024→801 状态衔接、MTP 长度 1–16 和 accepted 首/中/末边界。
- Python Ruff、`git diff --check`、构建脚本 `bash -n` 通过。所有构建安装到私有目录。

## 性能

Ascend 910B3 物理卡 6，HK4/HV12/K=V128；每层、每 rank 的图重放微基准，单位 µs。每块 100 次，decode 四块交叉顺序，prefill 两块；表中为块中位数，排除加载、capture、编译及 profiler。
decode 旧、新均使用 capacity32；MTP 长度 `[4,4,4,4]`，ragged `[4,2,1,0]`，普通 `[1,1,1,1]`。prefill 两请求长度分别为 `[128,131]`、`[512,515]`。

| 场景 | 旧适配器 µs | 原生接口 µs | 加速比 | 延迟下降 |
| --- | ---: | ---: | ---: | ---: |
| mtp4 | 483.75 | 68.44 | 7.07× | 85.9% |
| ragged | 456.45 | 43.57 | 10.48× | 90.5% |
| ordinary | 448.61 | 36.43 | 12.31× | 91.9% |
| prefill_128 | 361.93 | 202.76 | 1.79× | 44.0% |
| prefill_512 | 512.76 | 350.27 | 1.46× | 31.7% |

另测 token bucket capacity（16/7/4），延迟分别 71.74/44.13/36.15 µs；缩小 capacity 并非所有场景更快，主要收益来自原生状态访问和单一路径。
分离的 profiler 中，新 decode 只有输出置零和 `MegaGdnNativeDecode`；没有 pack/scatter、投影连续化或 fallback。旧图仍含 pack、MegaMTP、scatter 和 recurrent fallback。

## 加载与复现

完整接口、构建和测试命令见 `docs/zh/mega_gdn_native.md`。验证时使用的安装包、
二进制哈希、日志和临时环境均保留在验证环境，不随源码发布。

旧测试最初因同名 CausalConv1d 的数组 / 张量 metadata ABI 混用而崩溃；改用包含 standalone 卷积的专用 oracle 包后运行回归。早期增量构建也出现旧 kernel / 新 host 混用；本报告使用全新构建目录的结果，失败日志保留但不计入通过项。

## 2026-09-19 历史验收边界

这是当前 TP4/MTP3 所需 GDN 算子的 SDK 分支。共享实验框架正在被其他任务修改，本次没有覆盖其文件。完整模型接入后的 TP4/MTP3 CEval 和吞吐仍需重新验收；原实验的两道新增精度错误尚不能判定已修复。上述加速比不能换算为整模型吞吐；prefill 微基准使用图，也不能替代服务 eager TTFT。
验证限于 910B3；其他硬件没有本分支的结论。

## 2026-09-20 最终发布候选

上述历史问题不代表后续候选的验收状态。发布分支为 `Sinle4Cat/xllm-ops:adn_gdn`，配套框架为 `Sinle4Cat/vllm-ascend:adn_gdn_new`。

最终 A2/A3 H/O producer、QS 和 O consumer 的启用上限一致：每条 2048 chunks、整批 8192 chunks（chunk=128，即单条 256K、对齐整批 1M tokens）。保留 CSR 范围、chunk 数一致性及 ready mailbox 容量检查；超限回退，A5 条件不变。发布源码不包含被拒绝的 FP32-final-cache/KKT 实验。

在 910B3、HK4/HV12/D128、空卡上的旧→新→旧微基准：64K×4、128K×4、256K×4 算子耗时分别降低 6.70%、7.19%、5.18%。前两档使用逐级扩容候选，256K×4 使用最终候选，不能视为同一二进制的三档测量。256K×4 最终耗时 202.58 ms，user workspace 23.24 GiB，实测峰值 allocated 34.38 GiB；模型权重、KV 与额外图缓存需另外预算。未尝试预计超过可用显存的 512K×4。

最终候选的 256K×4 固定及交替输入结果与旧版逐位一致；16K、混合长度 `[262144,131073,65537,32769]`、单条 262145 和整批 8193 chunks 的回归也全部逐位一致。非连续状态 stride、padding、finite 和重复稳定性检查通过。只覆盖合法连续 CSR 与所列形状，不能推断任意 head 数或硬件均已验证。

前一档 32K×4 候选另做了 TP4/MTP3 模型验收：CEval 两次均为 41/50，预测与基线一致；单次配对测得 TTFT 下降约 0.52%，不能认为小幅收益稳定成立。最终 8192-chunk 二进制未重跑模型端到端验收；本次发布未改变框架调度预算 16384 或上下文上限 49152。
