# MegaGdnDraftDecode 用例设计文档

## 1. 算子标杆

PyTorch 无同名接口，使用 FP32 小算子拼接作为标杆，并保留生产路径的
BF16 舍入点：

```python
def reference_step(qkv, z, b, a, conv_weight, history, state,
                   a_log, dt_bias, norm_weight):
    conv_acc = (history.float() * conv_weight[:3].float()).sum(0)
    conv_acc += qkv.float() * conv_weight[3].float()
    conv = (conv_acc * torch.sigmoid(conv_acc)).to(torch.bfloat16)

    q, k, v = split_qkv(conv)
    q = F.normalize(q.float(), dim=-1, eps=1e-6) / (128**0.5)
    k = F.normalize(k.float(), dim=-1, eps=1e-6)
    g = (-torch.exp(a_log) * F.softplus(a.float() + dt_bias)) \
        .to(torch.bfloat16).float()
    beta = torch.sigmoid(b.float()).to(torch.bfloat16).float()
    state = state * torch.exp(g)[:, None, None]
    prediction = torch.einsum("hkv,hk->hv", state, k)
    delta = (v.float() - prediction) * beta[:, None]
    state = state + torch.einsum("hk,hv->hkv", k, delta)
    readout = torch.einsum("hkv,hk->hv", state, q)

    norm_input = readout.to(torch.bfloat16).float()
    rms_inv = torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + 1e-6)
    out = norm_input * rms_inv * norm_weight.float() * F.silu(z.float())
    return conv, state, out.to(torch.bfloat16)
```

完整参考实现位于 `test/python_test/test_mega_gdn_draft_decode.py`。

---

## 2. 用例说明

### 2.1 测试配置

算子的 dtype 不是可自由选择的单一 dtype，而是固定混合 profile：

```python
SUPPORTED_DTYPES = [
    "BF16 activations/Conv State + FP32 SSM State/gate parameters",
]
```

`TEST_SHAPES` 覆盖常用 batch、同 slot/prefix fork、FLA/non-FLA 和 1/2-token packed
sequence：

```python
# (name, q_lens, Hk, Hv, prefix_fork, fla_layout, validity)
TEST_SHAPES = [
    ("b1_s1_same_fla",       (1,),                 1, 2, False, True,  None),
    ("b2_s1_same_fla",       (1, 1),               1, 2, False, True,  None),
    ("b4_s1_same_fla",       (1, 1, 1, 1),         1, 2, False, True,  None),
    ("b8_s1_same_fla",       (1,) * 8,             1, 2, False, True,  None),
    ("b1_s2_same_fla",       (2,),                 1, 2, False, True,  None),
    ("b2_s2_same_fla",       (2, 2),               1, 2, False, True,  None),
    ("b4_mixed_same_fla",    (1, 2, 1, 2),         1, 2, False, True,  None),
    ("b8_mixed_same_fla",    (2, 1, 2, 1, 1, 2, 1, 2), 1, 2, False, True, None),
    ("b1_s1_prefix_fla",     (1,),                 1, 2, True,  True,  None),
    ("b2_s2_prefix_fla",     (2, 2),               1, 2, True,  True,  None),
    ("b4_mixed_prefix_fla",  (1, 2, 1, 2),         1, 2, True,  True,  None),
    ("b8_mixed_prefix_fla",  (2, 1, 2, 1, 1, 2, 1, 2), 1, 2, True, True, None),
    ("b1_s1_same_nonfla",    (1,),                 1, 2, False, False, None),
    ("b2_s2_same_nonfla",    (2, 2),               1, 2, False, False, None),
    ("b4_mixed_prefix_nonfla", (1, 2, 1, 2),       1, 2, True,  False, None),
    ("b8_mixed_prefix_nonfla", (2, 1, 2, 1, 1, 2, 1, 2), 1, 2, True, False, None),
]
```

`GENERAL_SHAPES` 覆盖支持范围内的 head geometry、大 batch 和 fresh/valid State
混合：

```python
GENERAL_SHAPES = [
    ("heads_1x1",       (1,),       1, 1,  False, True,  None),
    ("heads_1x4",       (2,),       1, 4,  False, True,  None),
    ("heads_2x2",       (1, 2),     2, 2,  False, True,  None),
    ("heads_2x8",       (2, 1),     2, 8,  False, True,  None),
    ("heads_4x8",       (1,),       4, 8,  False, True,  None),
    ("heads_4x16",      (2,),       4, 16, False, True,  None),
    ("heads_8x8",       (1,),       8, 8,  False, True,  None),
    ("heads_8x32",      (2,),       8, 32, False, True,  None),
    ("heads_16x16",     (1,),      16, 16, False, True,  None),
    ("heads_16x64",     (2,),      16, 64, False, True,  None),
    ("b16_alternating", (1, 2) * 8, 1, 2,  False, True,  None),
    ("b32_s1",          (1,) * 32,  1, 2,  False, True,  None),
    ("b32_alternating", (1, 2) * 16, 1, 2, True, True,  None),
    ("fresh_all",       (2, 1),     1, 2,  False, True,  (False, False)),
    ("fresh_mixed",     (1, 2, 2, 1), 1, 2, False, True,
                         (True, False, True, False)),
    ("fresh_prefix_nonfla", (2, 1), 1, 2, True, False, (False, True)),
]
```

边界值测试聚焦于数值而非增加不支持的 shape：

```python
BOUNDARY_VALUES = [
    "all-zero qkv/z/a/b with zero initial State",
    "near-zero q/k norm to exercise epsilon=1e-6",
    "large positive b to approach beta=1",
    "large negative b to approach beta=0",
    "positive a+dt_bias softplus region",
    "negative a+dt_bias softplus region",
    "read_state_indices == write_state_indices",
    "multiple requests read one shared prefix and write distinct private slots",
]
```

### 2.2 用例覆盖统计

| 类别 | Shape 数量 | 边界值数量 | dtype profile 数量 | 总用例数 |
|---|---:|---:|---:|---:|
| 典型 shape | 16 | - | 1 | 16 |
| 泛化 shape | 16 | - | 1 | 16 |
| 边界值 | - | 8 | 1 | 8 |
| **总计** | **32** | **8** | **1** | **40** |

---

## 3. 使用说明

### 生成测试数据示例

```python
q_lens = (1, 2, 1, 2)
q_cu_seq_lens = torch.tensor(
    (0, *torch.tensor(q_lens).cumsum(0).tolist()),
    dtype=torch.int32,
)
read_state_indices = torch.zeros(len(q_lens), dtype=torch.int32)
write_state_indices = torch.arange(1, len(q_lens) + 1, dtype=torch.int32)
state_validity_mask = torch.tensor(
    (True, False, True, True), dtype=torch.bool
)
```

### 注意事项

1. CPU 标杆先 snapshot 所有 read slot，然后才写 write slot，以正确模拟原地
   State tensor。
2. Prefix fork 用例必须额外检查 shared slot 在算子前后逐元素不变。
3. Conv State 要求 bitwise 一致；SSM State 和 `out` 按四个 BF16 舍入合同点
   设置阈值。
4. FLA/non-FLA 必须使用同一数学 State 进行转置对比，不得只对比
   各自随机输入。
