# GDN 模型数值修复及性能复测（2026-09-19）

原生 decode / MTP 的两处数值边界已修复，并完成 TP4/MTP3 模型 A1/B1/A2 测量。
这轮结果仍属于调试性能结论：历史 801-token 偶发差异尚未定位，模型精度的严格逐题一致性尚未关闭。

## 修改与回归

- `g` 保留 FP32，移除旧 SDK MegaMTP 的 BF16 舍入；`beta` 保持 BF16。
- L2Norm 后的 Q/K 先舍入 BF16，再进行 FP32 recurrence，query scale 在舍入后应用。
- 新增四个独立数值边界测试；旧包 4/4 失败，新包通过。SDK 总计 44/44 通过。
- 框架测试改用真实 convolution、L2Norm、gating 和 recurrent 算子作为对照。旧包 4/4 失败，新包完整 21/21 通过，误差阈值未放宽。
- 测试接入阶段的累计长度格式和 Q/K 连续化错误已经修正；对应失败日志保留，不能把这些测试接法错误算作产品修复。
- 保留此前 prefill 空槽位无状态计算、单序列非法索引保护及 workspace 生命周期修复。slot0 预热负偏移写回是此前 MTP 接受率约 4.55% 的主因。

## 模型性能

Qwen3.6-27B W8A8，910B3，CANN 9.1，TP4、MTP3、并发4，FP32 SSM，prefix cache 关闭。
物理卡4–7、端口8826，每轮独立服务/图缓存。每场景预热5批后测5批，共20请求；启动、编译、capture不计时。
吞吐按测量批次 wall time 之和计算，排除批次间指标抓取与日志时间。三轮 payload SHA 完全一致，prefix cache 命中均为0。

| 场景 | 指标 | baseline A1 | native B1 | baseline A2 |
| --- | --- | ---: | ---: | ---: |
| decode128 | TTFT ms | 522.978 | 401.386 | 433.936 |
| decode128 | TPOT ms | 10.805 | 9.689 | 10.543 |
| decode128 | 输出 tokens/s | 244.934 | 282.323 | 261.836 |
| decode128 | MTP 接受率 % | 80.212 | 78.111 | 81.278 |
| full-original | TTFT ms | 11563.233 | 11454.175 | 11564.713 |
| full-original | TPOT ms | 556.812 | 567.210 | 555.704 |
| full-original | 输出 tokens/s | 1.408 | 1.409 | 1.410 |
| full-original | MTP 接受率 % | 100.000 | 100.000 | 100.000 |

相对前后两轮baseline，decode128 TPOT降低 8.09%–10.32%，输出吞吐提高 7.82%–15.27%。此范围为两个对照点的结果，不是统计置信区间。

decode128 为四条固定短输入，每条生成128 tokens。full-original 为35269-token固定输入，自然EOS输出6 tokens；后者的 TPOT 不能代表稳态 decode。
长请求全部60份输出token IDs一致：True，序列为 [19186, 31515, 1522, 283, 2912, 248046]。
共享服务器、单个候选重启；A1/B1/A2控制了前后baseline漂移，但不等同于多轮ABBA、置信区间或SLA最大QPS验收。

## 精度与未关闭项

固定50题，temperature=0、seed=42、max_tokens=32，同一模板和请求。

| 服务 | C4 三次正确数/50 | C1 两次正确数/50 | C1 两次预测翻转数 |
| --- | --- | --- | ---: |
| baseline A1 | [40, 45, 41] | 未执行相同串行合同 | — |
| native B1 | [42, 40, 41] | [42, 39] | 5 |
| baseline A2 | [42, 42, 37] | [39, 39] | 2 |

C1保持max_tokens=32，增加top5 logprobs并逐条提交，答案token日志和逐题翻转另存。
baseline A1早期max_tokens=1探针只能观察首token，不用于答案精度比较。
baseline自身的重复变化说明不能把历史42→40/50全部归因于原生算子；这些小样本也不能证明模型精度等价，严格精度验收仍未关闭。

历史空槽位801-token输出有一次968/2465280元素差异，最大绝对误差0.00112915，原始失败保留。
本轮旧nullguard包五个新进程完整复测200项均通过，新modelmath包44项通过；这不构成该偶发问题已修复的证据。

## 分支、包与复现

- 复现分支：`adn_gdn`；SDK 与框架接入均在独立临时构建目录中完成。
- 测量控制器及每轮命令保留在验证环境，未作为源码发布文件。
- SDK测试：`test_native_model_contract.py`、`test_mega_gdn_native.py`；框架NPU测试：`test_gdn_kv.py`。
- 指标、接受计数、输出 IDs 和逐题比较结果均保存在验证环境；发布树不包含二进制、日志或凭据。

三轮服务都通过进程、端口和 NPU 清理检查；生产安装、共享旧实验目录与卡0任务未改动。
本报告采用 FP32 SSM cache 公共 ABI；失败的 FP32 final-cache / KKT 实验产物不构成发布来源。
候选运行的环境路径、服务路由日志均已归档；此调试控制器未采集候选worker的`/proc/maps`，不宣称完成正式加载身份验收。
