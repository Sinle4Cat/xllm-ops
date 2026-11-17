# MegaKdaDecode

支持 Ascend910B 和 Ascend950（A5）。两种架构分别选择对应内核；910B 保留当前 FP32 归约和状态提交实现。A5 支持 native（mode=0）与 MTP（mode=1）；mode=3 公版兼容模式仅支持 910B，A5 在 host tiling 阶段拒绝。


GLM-5.3-Flash vector KDA的实验性Conv + recurrent decode融合算子。
接入结构参考`mega_gdn_prefill_op`，包含独立的OpDef、Shape推导、Host tiling、
PTO Device入口、CMake构建和ACLNN/Python测试绑定。
GLM模型已通过`KdaLinearAttentionMixin`接入环境开关，默认仍为`fla_npu`
组合路径。`XLLM_KDA_KERNEL=mega_kda`为支持的输入启用原生prefill/普通decode及MTP verify；
不支持的路径在计算前选择，执行异常直接传播。单算子验证不代表整网精度或性能验收。

非MTP引擎可显式设置`XLLM_KDA_PREFILL_BACKEND=fla`，保留FLA prefill并使用
原生普通decode。此时`XLLM_KDA_DECODE_NORM=public`选择完整公开兼容模式
（mode3）；保留该旧环境变量名，但其语义不只包含归一化。默认值`native`选择mode0。
FLA-prefill兼容配置要求同时启用`XLLM_KDA_KERNEL=mega_kda`且verify width=1；
public模式要求该显式兼容配置。MTP mode=1接入会把V3逐token快照映射为算子检查点槽表；
仅支持的状态布局走原生算子，其余形状保留FLA V3回退。单算子验证不代表整网精度或性能验收。

## 入口与状态

- 仅一个`MegaKdaDecode`算子，Host根据显式mode设置tilingKey。
- `mode=0/key=1`为普通单token decode；`mode=1/key=2`为MTP verify。
  单token的MTP仍走key 2，不根据token数量猜测语义。
- `mode=3/key=3`为公开兼容普通decode。与mode0一样，要求capacity=1、
  accepted=None、Conv history=3、offsets=`[0,1,...,B]`，不能按非零mode判断MTP。
- Conv与SSM各自有独立输入池、输出池和读写映射，输入输出容量允许不同。
  Python绑定拒绝输出与输入共享storage，Device不修改输入状态或QKV。
- accepted count、offsets和slot映射均为Device tensor，不固化为Host tiling属性。
  MTP使用`accepted-1`恢复SSM checkpoint与Conv history；采样接受判定不在算子内。
- slot 0有效，读slot -1表示inactive，写slot -1表示丢弃该项状态写回。
  调度器必须保证有效写目标唯一；Host不通过D2H校验动态映射。

## 支持范围

目前仅注册Ascend 910B。BF16输入、FP32 `[slots,H,V,K]` SSM，D=128，
H=1..128，MTP query capacity=1..17。模型MTP=K表示K个草稿token，
验证时还包含当前token，因此capacity=K+1；模型MTP1..16对应capacity2..17。
capacity=1仍支持单token MTP verify。普通decode的Conv history长度为3，
MTP为capacity+2。Conv权重按MegaGDN的`[4,C]`布局预打包，不能逐次转置。

与`MegaKdaPrefill`不同，decode的beta是sigmoid前的BF16投影，sigmoid
在算子内部计算；prefill接收sigmoid后的FP32 beta。decode的读slot -1
表示inactive，不是prefill的零初态。阶段切换时必须分别遵守各自接口契约。

共同数学边界为Conv累加后先舍入BF16，再计算SiLU并再次舍入BF16；
beta sigmoid也先舍入BF16，再转回FP32递推。Q缩放为`128**-0.5`，
SSM使用FP32 `[V,K]`，不复用GDN的标量decay公式。

| 数值阶段 | native/MTP（mode0/1） | 公开兼容（mode3） |
| --- | --- | --- |
| Conv输入与历史 | BF16→FP16→FP32 | BF16→FP32，省略FP16窄化 |
| Q/K归一化 | 分母为`sqrt(sum(x*x)+1e-6)` | 无epsilon；零向量保持零，避免`0/0` |
| gate | 直接以`-5`除以sigmoid分母 | 先求分母倒数得到sigmoid，再乘`-5` |
| 64行递推归约 | 一组64行 | 两组各32行 |

四项差异由mode3/key3一起选择。省略FP16允许超出FP16范围的有限BF16输入；
省略epsilon也会改变很小的非零向量行为。实数公式相同不代表浮点计算顺序相同。
每个task负责32或64行V，使用AIV，显式UB分别为62,368或120,032字节
，无用户workspace。当64行版本的总分片数`2 * B * H`足以覆盖已启动
AIV核时采用该版本，否则保留32行版本。更大分片使每个请求/head的重复
Conv、Q/K和gate计算减半，只有首个V分片发布Conv state，状态读写所有权
保持不变。两版本的FP32归约分组可能不同，按同一Golden阈值验证数值；
每个版本内的固定输入Graph回放仍要求逐字节一致。收益边界包含batch、head数、
实际活跃行分布和MTP capacity。

## 验证边界

构建入口为`bash build.sh --op-name mega_kda_decode`。
测试为`test/python_test/test_mega_kda_decode.py`，未设置`MEGA_KDA_TEST_NPU`
时只执行CPU合同测试；真机测试需显式选择独占NPU。
测试包含fixed20、独立CPU/NPU carry20、动态accepted/mapping Graph replay、
padding、不同状态池容量和alias拒绝。
fixed20使用图模式并按原始字节比较，区分正零与负零。
独立CPU golden的Conv、归一化、gate和递推使用FP64，保留模式对应的FP16
转换与共同的Conv累加、SiLU输出、beta sigmoid的BF16舍入边界。
它是数学参考，不逐比特模拟Device FP32累加或归约树。额外FP32阶段检查覆盖
gate先倒数后乘法、beta舍入、零/近零norm和超出FP16范围的有限输入。
shape/dtype不匹配或输出、golden、派生误差出现
NaN/Inf一律失败。按`abs(golden)=1e-3`分组报告正常值相对误差、近零值
绝对误差及超限比例，作为额外诊断；原逐元素验收阈值不变：输出
`rtol=1e-2, atol=1e-3`，状态`rtol=1e-3, atol=1e-4`。

编译成功、CPU golden自检成功都不等于原生算子精度通过。仍需完成真实ACLNN/Graph
状态保持验证、冻结Triton基线对齐、预热收敛和正式性能验收。
目前不声明超过Triton 20%，也不声明整网精度通过。

TP矩阵按64个KDA heads切分，覆盖TP1/2/4/8/16/32/64，对应每卡
H=64/32/16/8/4/2/1。每档交叉mode0/3普通decode与全部整数query capacity1..17，
覆盖模型MTP1..16，包含CPU
golden、dense fixed20图回放、动态元数据replay20和独立状态carry20。
原batch/ragged/padding用例保留，Host policy也覆盖全部head数。
可用`pytest -k tp`选择矩阵，`-k tp8`选择单档；这只是单卡分片shape
测试，不代表多卡TP通信、模型显存容量或整网验证通过。

mode3专项覆盖H=8、B=1/4/5，以及H=3、B=2的fixed20、changed-metadata20、
carry20、零/近零向量、Conv大值和gate饱和。前两个H=8用例走合批Conv，
B=5和H=3走通用路径。验证32/64行时，应以选中设备的`GetCoreNumAiv`
设置`MEGA_KDA_TEST_AIV_CORES`，运行`-k public_compat_row_dispatch`；测试记录
实际核数对应的launch block数及行数选择，不能只由shape名称推断分支覆盖。

完整参数、状态前置条件和复现命令见[英文接口合同](../en/mega_kda_decode.md)。

当本地head数H=8且batch B<=4时，kernel合并Q/K/V Conv预处理。
普通单token decode还批量加载四tap输入/权重，并合并原始Conv状态发布。
32/64行状态任务划分及浮点转换边界保持不变。普通decode的显式UB为
83,872/141,536字节，MTP为70,048/127,712字节；其他shape保留原路径。
这些原生算子特化不代表整模型TPOT已获得同等收益。
