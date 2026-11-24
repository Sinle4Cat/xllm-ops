# MegaKdaDecode

[English](../en/mega_kda_decode.md) | 简体中文

算子融合 GLM-5.3-Flash vector KDA 的 Conv 和 recurrent decode。Ascend910B
支持普通 decode（mode 0）、MTP verify（mode 1）和公版兼容（mode 3）。
Ascend950（A5）支持 mode 0 和 mode 1。两平台分别选择对应内核。OpDef、
Shape 推导、Host tiling、PTO Device 入口、CMake 构建和 ACLNN 绑定沿用
`mega_gdn_prefill_op` 的结构。

GLM模型已通过`KdaLinearAttentionMixin`接入环境开关，默认仍为`fla_npu`
组合路径。`XLLM_KDA_KERNEL=mega_kda`为支持的输入启用原生prefill/普通decode及MTP verify；
不支持的路径在计算前选择，执行异常直接传播。单算子验证不代表整网精度或性能验收。

非MTP引擎可显式设置`XLLM_KDA_PREFILL_BACKEND=fla`，保留FLA prefill并使用
原生普通 decode。在 Ascend910B 上，此时 `XLLM_KDA_DECODE_NORM=public` 选择完整公开兼容模式
（mode3）；保留该旧环境变量名，但其语义不只包含归一化。默认值`native`选择mode0。
FLA-prefill兼容配置要求同时启用`XLLM_KDA_KERNEL=mega_kda`且verify width=1；
public模式要求该显式兼容配置。MTP mode=1接入会把V3逐token快照映射为算子检查点槽表；
仅支持的状态布局走原生算子，其余形状保留FLA V3回退。单算子验证不代表整网精度或性能验收。

## 入口与状态

- 仅一个`MegaKdaDecode`算子，Host根据显式mode设置tilingKey。
- `mode=0/key=1`为普通单token decode；`mode=1/key=2`为MTP verify。
  单token的MTP仍走key 2，不根据token数量猜测语义。
- `mode=3/key=3` 为 Ascend910B 的公开兼容普通 decode。与mode0一样，要求capacity=1、
  accepted=None、Conv history=3、offsets=`[0,1,...,B]`，不能按非零mode判断MTP。
- Conv与SSM各自有独立输入池、输出池和读写映射，输入输出容量允许不同。
  Python绑定拒绝输出与输入共享storage，Device不修改输入状态或QKV。
- accepted count、offsets和slot映射均为Device tensor，不固化为Host tiling属性。
  MTP使用`accepted-1`恢复SSM checkpoint与Conv history；采样接受判定不在算子内。
- slot 0有效，读slot -1表示inactive，写slot -1表示丢弃该项状态写回。
  调度器必须保证有效写目标唯一；Host不通过D2H校验动态映射。

## 支持范围

Ascend910B 和 Ascend950 均支持 BF16 输入、FP32 `[slots,H,V,K]` SSM，D=128，
H=1..128，MTP query capacity=1..17。模型MTP=K表示K个草稿token，
验证时还包含当前token，因此capacity=K+1；模型MTP1..16对应capacity2..17。
capacity=1仍支持单token MTP verify。普通decode的Conv history长度为3，
MTP为capacity+2。Conv权重按MegaGDN的`[4,C]`布局预打包，不能逐次转置。

与`MegaKdaPrefill`不同，decode的beta是sigmoid前的BF16投影，sigmoid
在算子内部以FP32计算；prefill接收sigmoid后的FP32 beta。decode的读slot -1
表示inactive，不是prefill的零初态。阶段切换时必须分别遵守各自接口契约。

两平台均以 FP32 做四 tap Conv 累加及 SiLU，SiLU 结果舍入为 BF16 后
进入递推。beta 从 BF16 转为 FP32 后计算 sigmoid，并在递推时保留 FP32。
Q 的缩放系数为 `128**-0.5`，SSM 使用 FP32 `[V,K]`。

| 平台及模式 | Conv 输入与历史 | SiLU 前的累加结果 |
| --- | --- | --- |
| Ascend910B mode 0/1 | BF16 → FP16 → FP32 | 舍入 BF16 后转回 FP32 |
| Ascend910B mode 3 | BF16 → FP32 | 舍入 BF16 后转回 FP32 |
| Ascend950 mode 0 | BF16 → FP32 | 保持 FP32 |
| Ascend950 mode 1 | BF16 → FP16 → FP32 | 舍入 BF16 后转回 FP32 |

910B 的 mode 0、1、3 都按 `sqrt(sum(x*x))` 计算 FP32 行范数。范数
为正时乘以其倒数，范数为零时保留原行，随后对 Q 做缩放。A5 的 mode 0
和 mode 1 保留 r003 语义，以 FP32 除以 `sqrt(sum(x*x) + 1e-6)` 后
再缩放 Q。

mode 0/1 以 `-5` 除以 sigmoid 分母得到 gate。910B mode 3 先求分母
倒数，再乘 `-5`。在 910B 的 64 行 V 任务中，普通 mode 0/3 采用两组
32 行归约，MTP mode 1 使用一组 64 行。mode 3 的 BF16 直转 FP32
支持超出 FP16 范围的有限 Conv 输入。

910B 每个 AIV 任务负责 32 或 64 行 V，显式 UB 分别为 62,368 或 120,032 字节
，无用户workspace。当64行版本的总分片数`2 * B * H`足以覆盖已启动
AIV核时采用该版本，否则保留32行版本。更大分片使每个请求/head的重复
Conv、Q/K和gate计算减半，只有首个V分片发布Conv state，状态读写所有权
保持不变。两版本的FP32归约分组可能不同，按同一Golden阈值验证数值；
每个版本内的固定输入Graph回放仍要求逐字节一致。收益边界包含batch、head数、
实际活跃行分布和MTP capacity。

## 验证边界

加载目标设备的 CANN 环境后，在仓库根目录执行打包并安装到私有 OPP。
A5 使用 `SOC=ascend950`。环境变量在当前 shell 下选择这套算子包。
Python 测试扩展使用同一套 Torch 和 CANN 编译。

```bash
BUILD_ROOT=$(mktemp -d)
SOC=ascend910b
bash xllm_ops/build.sh --pkg --ops=mega_kda_prefill,mega_kda_decode --soc="$SOC" -j4 \
  --build-dir="$BUILD_ROOT/build" --output-dir="$BUILD_ROOT/output" \
  --build-out-dir="$BUILD_ROOT/build_out"

OPP_ROOT="$BUILD_ROOT/opp"
mkdir -p "$OPP_ROOT"
bash "$BUILD_ROOT"/build/cann-ops-xllm-custom_linux-*.run --install-path="$OPP_ROOT"
KDA_VENDOR="$OPP_ROOT/vendors/custom_xllm_math"
export ASCEND_CUSTOM_OPP_PATH="$KDA_VENDOR"
export LD_LIBRARY_PATH="$KDA_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`test/python_test/test_mega_kda_decode.py` 默认执行 CPU 合同测试。
真机测试通过 `MEGA_KDA_TEST_NPU` 选择独占设备，覆盖固定输入 fixed20、
独立 CPU/NPU carry20、动态 accepted/mapping 图回放、padding、独立池容量
及别名拒绝。固定输入回放按原始字节比较，区分正零与负零。

910B 真机使用独立 CPU FP32 递推标杆 `reference_910b(data)`，保留各模式的
转换边界。A5 真机使用 r003 标杆 `reference(data, a5=True)`，按 A5 的
Conv 和归一化语义执行 FP64 递推。报告以 `abs(golden)=1e-3` 分组记录
正常值相对误差、近零值绝对误差和超限比例。Attention 的逐元素容差为
`rtol=1e-2, atol=1e-3`，FP32 SSM 为 `rtol=1e-3, atol=1e-4`。
同时检查 shape、dtype、有限值及 Conv 和未写区的字节一致性。

共用 KDA pytest 清单含 805 个节点，Decode 模块收集 523 个。A5 跳过
mode 3 的正向真机用例，仍执行非法 accepted counts 和 capacity 的绑定
拒绝用例。保留的 A5 SiLU 中点用例覆盖 mode 1。上述容差用于算子验证，
整网精度和性能另行测量。

TP矩阵按64个KDA heads切分，覆盖TP1/2/4/8/16/32/64，对应每卡
H=64/32/16/8/4/2/1。每档覆盖普通 mode 0，以及 query capacity 为 1–17
的 MTP mode 1。910B 另覆盖 mode 3。包含 CPU 标杆、dense fixed20 图回放、
动态元数据 replay20 和独立状态 carry20。
原batch/ragged/padding用例保留，Host policy也覆盖全部head数。
可用`pytest -k tp`选择矩阵，`-k tp8`选择单档；这只是单卡分片shape
测试，不代表多卡TP通信、模型显存容量或整网验证通过。

mode3专项覆盖H=8、B=1/4/5，以及H=3、B=2的fixed20、changed-metadata20、
carry20、零/近零向量、Conv大值和gate饱和。前两个H=8用例走合批Conv，
B=5和H=3走通用路径。验证32/64行时，应以选中设备的`GetCoreNumAiv`
设置`MEGA_KDA_TEST_AIV_CORES`，运行`-k public_compat_row_dispatch`；测试记录
实际核数对应的launch block数及行数选择，不能只由shape名称推断分支覆盖。

910B 的本地 head 数 H=8 且 batch B<=4 时，kernel 合并 Q/K/V Conv 预处理。
普通单token decode还批量加载四tap输入/权重，并合并原始Conv状态发布。
32/64行状态任务划分及浮点转换边界保持不变。普通decode的显式UB为
83,872/141,536字节，MTP为70,048/127,712字节；其他shape保留原路径。
这些原生算子特化不代表整模型TPOT已获得同等收益。
