# MegaKdaPrefill

支持 Ascend910B 和 Ascend950（A5）。两种架构分别选择对应内核；910B 保留当前 FP32 归约和状态提交实现。A5 的 `ffts_addr` 可为 0，910B 仍要求有效地址。


参考`mega_gdn_prefill_op`的注册、构建和单次混合核启动结构，新增GLM
vector KDA prefill。完整参数和边界见[英文接口文档](../en/mega_kda_prefill.md)。

融合范围为Conv/SiLU、Q/K L2归一化、vector gate、chunk cumsum，以及
KKT、FP32 inverse、WY、H、O。保留FP32 A2、K_eff/W和状态计算；不包含输出
norm/projection，也不提供decode/MTP算法分支。

GLM模型已通过环境开关接入，默认仍为`fla_npu`组合路径。
`XLLM_KDA_KERNEL=mega_kda`为支持的输入启用原生prefill/普通decode，
默认`XLLM_KDA_PREFILL_BACKEND=native`。非MTP引擎可以显式选择
`XLLM_KDA_PREFILL_BACKEND=fla`，保留FLA prefill并使用原生decode。
此时旧环境变量`XLLM_KDA_DECODE_NORM=public`选择decode的mode3完整
公开兼容语义：Conv窄化、epsilon、gate顺序和归约分组四项一起改变；
默认值`native`选择mode0。public要求显式FLA-prefill配置，兼容配置要求
verify width=1。模型MTP verify继续使用FLA V3，decode mode1仅表示算子接口能力。
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

支持 Ascend910B 和 Ascend950，K/V head dimension 为 128，Conv width 为 4。
Ascend910B 的内部 chunk size 为 64，Ascend950 为 128。
与既有MegaGDN绑定一致，调用前须将输入所在NPU设为当前设备。
CPU golden使用FP64递推和明确的BF16舍入边界，正常值相对误差、近零值
绝对误差及超限比例分别报告。测试包含空序列、尾块、独立池容量、输入不变、
未写区保持，以及100轮固定输入/独立推进状态的图回放。

当前单算子smoke门槛：以`abs(golden)=1e-3`分组，正常值相对误差阈值
为1%，近零值绝对误差阈值为`1e-5`，两类超限比例各不超过0.1%。这不等于
所有元素误差均小于1%，也不代表模型精度验收。1k至32k各档另检验20轮
固定输入图回放的逐比特一致性；TP1、2k加强到1000轮，以覆盖chunk状态
缓冲区复用引起的延迟错误。每轮分别检查输出、Conv state和SSM state，
按原始字节比较，区分正零与负零。
输出和golden必须同shape、同dtype；输出、golden或派生误差出现NaN/Inf
一律失败。CPU漏洞回归见`test_mega_kda_accuracy_checks.py`。

TP shape测试按模型的64个KDA heads设置：TP1/2/4/8/16/32/64对应
每卡H=64/32/16/8/4/2/1。每档交叉覆盖1k至32k、带/不带Conv bias的
尾块与空序列，以及fixed100/carry100图回放；CPU golden和Host policy
也覆盖这7档head数。可用`pytest -k tp`选择矩阵，`-k tp8`选择单档。
这是单卡执行的分片shape验证，不是多卡TP/HCCL或整网验收。

torch_npu 2.9的NPUGraph每次回放前使用
两个Fill核更新RNG seed和offset；它们属于框架回放开销，不是本算子的
workspace或状态初始化，算子中不包含这类Fill实现。纯kernel耗时应单列，
完整图调用性能对比应计入框架开销。

只有CPU测试或编译通过不能宣称NPU精度通过。历史模型结果不能替代
本次接入的精度、性能或模型验收；正式性能收益另测。


## Conv前端分块

Conv和Q/K归一化一次处理16个独立token。序列开始的前三个token
沿用读取历史状态的逐行路径；后续chunk直接使用分块路径，尾块只写有效行。
四个tap的加法顺序、Conv输入的FP16转换、Conv与归一化Q/K的BF16
边界保持一致。Conv累加先舍入BF16，再计算SiLU并再次舍入BF16；
gate累加继续采用顺序FP32扫描。前端UB占用46,336字节，
没有新增GM workspace；进入gate扫描前完成MTE3存储。
MTP模型的prefill调用方传入最后三行Conv历史，并单独保留额外的投机历史。
