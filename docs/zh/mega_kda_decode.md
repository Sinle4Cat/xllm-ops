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
