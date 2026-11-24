# MegaKdaPrefill

English | [简体中文](../zh/mega_kda_prefill.md)

Ascend910B and Ascend950 (A5) select separate kernels. Ascend910B requires a
valid `ffts_addr`, and Ascend950 accepts zero. The native operator fuses
width-four Conv/SiLU, Q/K L2 normalization, vector gating and KDA prefill.
Its packaging and mixed AIC/AIV launch follow the `mega_gdn_prefill_op` conventions.

The GLM model integration is environment-gated: the default uses the `fla_npu`
composition. `XLLM_KDA_KERNEL=mega_kda` enables supported native prefill and
ordinary decode, with `XLLM_KDA_PREFILL_BACKEND=native` as the default.
Non-MTP engines may explicitly set `XLLM_KDA_PREFILL_BACKEND=fla` to retain FLA
prefill and native decode. On Ascend910B, the legacy option
`XLLM_KDA_DECODE_NORM=public` selects decode's complete mode 3 public compatibility
contract for Conv input conversion and gate evaluation.
The default `native` value selects mode 0. Public compatibility requires FLA
prefill, and this compatibility route requires verify width 1. Model MTP verify uses decode mode 1 for supported layouts and falls back to FLA V3
for other layouts. See [the decode contract](mega_kda_decode.md).
Isolated operator checks do not establish full-model acceptance or performance.

## Interface

All tensors are contiguous ND on the same NPU. Select that NPU as the current
device before calling the binding, as for the existing MegaGDN bindings.
`T` is the total packed token
count, `B` the number of sequences, `H` the head count, and `C = 384 * H`.
K/V dimensions are 128 and Conv width is four. Ascend910B uses 64-token
chunks for every supported head count. Ascend950 uses 128-token chunks in
its general path and a direct recurrence for B=1, H=8 and T=1–4.

| Argument | Shape | Dtype | Meaning |
|---|---|---|---|
| qkv | `[T,C]` | BF16 | Raw projection, channel order Q then K then V |
| gate | `[T,H,128]` | BF16 | Raw gate projection |
| beta | `[T,H]` | FP32 | Post-sigmoid beta, not raw logits |
| a_log | `[H]` | FP32 | Gate scale parameter |
| gate_bias | `[H,128]` | FP32 | Gate bias |
| conv_weight | `[4,C]` | BF16 | Oldest to newest tap |
| conv_bias | `[C]`, optional | BF16 | Conv bias, `None` means zero |
| conv_state_in | `[Ci,3,C]` | BF16 | Read-only raw projection history |
| ssm_state_in | `[Si,H,128,128]` | FP32 | Read-only state in **VK** layout |
| cu_seqlens | `[B+1]` | int32 | Packed offsets, first 0 and last T |
| conv_read_indices | `[B]` | int32 | Conv source slots |
| conv_write_indices | `[B]` | int32 | Conv destination slots |
| ssm_read_indices | `[B]` | int32 | SSM source slots |
| ssm_write_indices | `[B]` | int32 | SSM destination slots |
| output | `[T,H,128]` | BF16 | Caller-allocated attention output |
| conv_state_out | `[Co,3,C]` | BF16 | Caller-allocated independent Conv pool |
| ssm_state_out | `[So,H,128,128]` | FP32 | Caller-allocated independent **VK** pool |

ACLNN attributes are `conv_output_slots`, `ssm_output_slots` and `ffts_addr`
in that order. The Python test binding derives capacities from the output
pools. It queries the FFTS address on Ascend910B and passes zero on Ascend950.
Offsets and slot mappings stay on the device.

```python
output, conv_out, ssm_out = custom_ops_lib.mega_kda_prefill(
    qkv, gate, beta, a_log, gate_bias, conv_weight, conv_bias,
    conv_state_in, ssm_state_in, cu_seqlens,
    conv_read_indices, conv_write_indices, ssm_read_indices, ssm_write_indices,
    output, conv_state_out, ssm_state_out,
)
```

## State Contract

- Read `-1` means a zero initial history/state, **not** an inactive sequence.
- Write `-1` discards that family's final state, without suppressing output.
- Empty sequences publish nothing. Unwritten output slots retain their contents.
- Slot zero is valid. Conv/SSM pool capacities and mappings are independent.
- All three outputs must have disjoint storage from every input and each other.
  The binding rejects storage aliases; direct ACLNN callers have the same obligation.
- Nonempty sequences must have unique nonnegative destinations within each
  output family. Device metadata must remain immutable during one invocation.
- Host rejects unsupported shapes/dtypes. Invalid Device offsets/slots or duplicate
  destinations fail closed with no Device publication. There is no error-status
  output; the scheduler must validate metadata when constructing it.

## Numerical Contract

Both platforms convert the BF16 Conv input and history directly to FP32,
accumulate four taps and apply SiLU before rounding the activation to BF16.
Ascend910B uses FP32 fused multiply-adds for tap accumulation where applicable.
Both use a Q/K normalization epsilon of `1e-6` and publish normalized Q/K
in BF16. Ascend910B multiplies by the FP32 reciprocal square root of the
sum of squares plus epsilon. Ascend950 uses the r003 normalization path,
with range scaling for large rows.

The GLM safe gate is `g = -5 * sigmoid(exp(a_log) * (gate + gate_bias))`.
Gate prefix sums and beta use FP32. The mathematical recurrence in VK layout
is:

```text
S = S * exp(g)[None, :]
delta = beta * (v - S @ k)
S = S + delta[:, None] * k[None, :]
o = (S @ q) / sqrt(128)
```

The general paths evaluate the recurrence through KKT, inverse, WY and chunk
H/O stages. Ascend910B's C64 implementation uses FP32 state and explicit BF16
stage boundaries. Ascend950 retains the r003 C128 stages. Both transpose the
internal KV representation at the public VK state pools.

Supported static bounds: `1 <= T <= 131072`, `1 <= B <= 1024`,
`1 <= H <= 128`, positive pools up to 1048576 slots, and workspace <= 32GiB.
Actual allocation is limited by available HBM. No all-empty/T=0 invocation,
GQA, alternate head dimension, alternate gate lower bound, FP16 public input,
norm/projection fusion or speculative verification is provided. A caller using
MTP can use this prefill operator with the last three rolling Conv history rows;
the caller retains any additional speculative history/checkpoint rows.

## Build And Validate

Use the private OPP build and installation commands for both KDA operators.
For a Prefill-only package pass `--ops=mega_kda_prefill` to `xllm_ops/build.sh`
and choose the target SoC. [Private OPP commands](mega_kda_decode.md#build-and-test)

The exported entry points are `aclnnMegaKdaPrefillGetWorkspaceSize` and
`aclnnMegaKdaPrefill`. Compile `test/python_test/setup.py` against the same
Torch and CANN environment.

```bash
MEGA_KDA_PREFILL_TEST_NPU=0 python -m pytest -x -s \
  test/python_test/test_mega_kda_prefill.py test/python_test/test_mega_kda_prefill_carry500.py
```

The command requires an exclusively available device and the custom
OPP on the loader path. The native test harness selects `reference_910b(data)`
on Ascend910B, an independent CPU FP32 C64 reference with BF16 stage boundaries.
On Ascend950 it selects `reference(data, a5=True)`, the r003 reference with FP64
recurrence and the A5 Conv and normalization boundaries. Both references use
the same Prefill checks. Those checks require matching shapes and dtypes, finite
values, byte-identical Conv state and untouched slots, and the following error
limits for attention and written SSM state.

| Reference magnitude | Error threshold | Maximum fraction above threshold |
|---|---|---|
| `abs(golden) >= 1e-3` | 1% relative | 0.1% |
| `abs(golden) < 1e-3` | `1e-5` absolute | 0.1% |

The Prefill and carry500 modules cover the following NPU cases.

- TP1/2/4/8/16/32/64, representing local H=64/32/16/8/4/2/1 of the model's
  64 heads, with tail and empty sequences, bias variants and state-pool checks.
- Common lengths 1024, 2048, 4096, 8192, 16384 and 32768. The retained pairs
  exclude 2048 at TP1 and 8192 at TP8/TP16. Each retained pair checks the
  CPU reference once and compares all three output buffers byte-for-byte
  across 20 fixed-input graph replays.
- Fixed-input and independent-state 100-replay tests at TP1/2/4/16/32/64.
  Fixed-input replay compares raw bytes, including signed zero.
- The carry500 suite advances independent CPU and NPU states at H=1/8/64
  with gate bias `-4.0`. Graph replay is the default.

These cases exercise each local head shape on one device. Distributed TP,
model accuracy and performance have separate evaluation requirements.

One mixed Device entry does not imply one total runtime task. In torch_npu 2.9,
NPUGraph replay updates the RNG seed and offset with two Fill kernels before
executing the graph. These are framework replay overhead, not MegaKdaPrefill
workspace/state initialization, and no such Fill implementation is included in
this operator. Report kernel-only time separately; include framework overhead
when comparing full graph calls. Performance and model accuracy require their own
fresh measurements.


## Conv Frontend Tiling

The Ascend910B frontend has 16-, 32- and 64-row Conv tiles. The first three
sequence tokens use the history-aware row path. Later
rows use the tiled path, with valid-row stores for ragged tails. Gate prefix
sums use a sequential FP32 scan. MTE3 stores drain before the frontend arena
is reused by the gate scan.
