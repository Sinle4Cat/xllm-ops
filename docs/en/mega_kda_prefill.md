# MegaKdaPrefill

Ascend910B and Ascend950 (A5) select separate kernels. The 910B backend retains the current FP32 reduction and state publication implementation. On A5, `ffts_addr` may be zero; 910B still requires a valid address.


Ascend910B mixed AIC/AIV single-launch vector-gated KDA prefill. The native
operator follows `mega_gdn_prefill_op` packaging and launch conventions, with
FP32 KKT/inverse/WY/H/O numerical stages. It does not use the GDN
scalar-gate recurrence and is not a decode/MTP implementation.

The GLM model integration is environment-gated: the default uses the `fla_npu`
composition. `XLLM_KDA_KERNEL=mega_kda` enables supported native prefill and
ordinary decode, with `XLLM_KDA_PREFILL_BACKEND=native` as the default.
Non-MTP engines may explicitly set `XLLM_KDA_PREFILL_BACKEND=fla` to retain FLA
prefill and native decode. In that configuration the legacy option
`XLLM_KDA_DECODE_NORM=public` selects decode's complete mode 3 public compatibility
contract; it changes Conv narrowing, epsilon, gate order and reduction grouping.
The default `native` value selects mode 0. Public compatibility requires FLA
prefill, and this compatibility route requires verify width 1. Model MTP verify
keeps the FLA V3 path. Decode mode 1 is an operator-level interface, not evidence
of model MTP integration. See [the decode contract](mega_kda_decode.md).
Isolated operator checks do not establish full-model acceptance or performance.

## Interface

All tensors are contiguous ND on the same NPU. Select that NPU as the current
device before calling the binding, as for the existing MegaGDN bindings.
`T` is the total packed token
count, `B` the number of sequences, `H` the head count, and `C = 384 * H`.
K/V dimensions are 128 and Conv width is four. The internal chunk size is
64 on Ascend910B and 128 on Ascend950.

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

ACLNN attributes are `conv_output_slots`, `ssm_output_slots`, `ffts_addr`, in
that order. The Python C++ binding derives capacities and obtains the FFTS
address from the active runtime. It never copies Device metadata to CPU.

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

Conv evaluates SiLU after four taps. It preserves the existing Ascend input/history
FP16 cast, rounds accumulation to BF16 before SiLU, then rounds SiLU output to
BF16. Q/K are L2-normalized with epsilon `1e-6` and
rounded to BF16 before KDA. The fixed GLM safe gate is
`g = -5 * sigmoid(exp(a_log) * (gate + gate_bias))` in FP32. Chunk-local gate
cumsum is FP32; beta stays FP32. The recurrence oracle in VK layout is:

```text
S = S * exp(g)[None, :]
delta = beta * (v - S @ k)
S = S + delta[:, None] * k[None, :]
o = (S @ q) / sqrt(128)
```

The Device implements this with a chunk algorithm, including FP32
A2, K_eff/W, inverse and state. It is not a tokenwise Device fallback. Internal
states use KV; the frontend/epilogue transpose to/from the public VK pools.
Reassociating Conv, normalization and cumsum can change rounding relative to
the separate kernels, so historical results do not certify this integration.

Supported static bounds: `1 <= T <= 131072`, `1 <= B <= 1024`,
`1 <= H <= 128`, positive pools up to 1048576 slots, and workspace <= 32GiB.
Actual allocation is limited by available HBM. No all-empty/T=0 invocation,
GQA, alternate head dimension, alternate gate lower bound, FP16 public input,
norm/projection fusion or speculative verification is provided. A caller using
MTP can use this prefill operator with the last three rolling Conv history rows;
the caller retains any additional speculative history/checkpoint rows.

## Build And Validate

Use the repository's normal isolated build/OPP flow with
`--ops=mega_kda_prefill --soc=ascend910b`. The exported entry points are
`aclnnMegaKdaPrefillGetWorkspaceSize` and `aclnnMegaKdaPrefill`.
Compile `test/python_test/setup.py` against the same Torch/CANN environment.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q test/python_test/test_mega_kda_prefill.py
MEGA_KDA_PREFILL_TEST_NPU=0 python -m pytest -x -s test/python_test/test_mega_kda_prefill.py
```

The second command requires an exclusively available device and the new custom
OPP on the loader path. Tests compare against an independent FP64 CPU oracle,
separate normal relative errors from near-zero absolute errors, exclude
untouched pool padding from error fractions, and check all 100 fixed/carry
Graph replays. The provisional operator smoke gate splits at `abs(golden)=1e-3`:
normal relative tolerance is 1%, near-zero absolute tolerance is `1e-5`, and
each category's exceeding fraction must be <= 0.1%. This is not an all-elements
1% guarantee or a production model accuracy criterion. Common 1k through 32k
tests also check 20 fixed-input bitwise-identical Graph replays per shape;
TP1 at 2k uses 1000 replays to cover delayed chunk-state buffer reuse failures.
Every replay compares output, Conv state and SSM state separately.
Bitwise checks compare raw bytes, including signed zero. Output and golden must
have identical shapes/dtypes; nonfinite outputs, golden values or derived errors
always fail. CPU regression coverage is in `test_mega_kda_accuracy_checks.py`.

TP shape coverage uses the model's 64 KDA heads: TP1/2/4/8/16/32/64 map to
local H=64/32/16/8/4/2/1. Each TP is crossed with all six common lengths,
tail/empty-sequence cases with and without Conv bias, and fixed100/carry100
Graph replay. CPU oracle and Host policy checks cover the same head counts.
These are single-device per-rank shape tests, not distributed TP or HCCL tests.
Select the TP matrix with `pytest -k tp` (or one size with `-k tp8`).

One mixed Device entry does not imply one total runtime task. In torch_npu 2.9,
NPUGraph replay updates the RNG seed and offset with two Fill kernels before
executing the graph. These are framework replay overhead, not MegaKdaPrefill
workspace/state initialization, and no such Fill implementation is included in
this operator. Report kernel-only time separately; include framework overhead
when comparing full graph calls. CPU tests/skips are not native acceptance. Performance and
model accuracy require their own fresh measurements; no speedup is implied.


## Conv Frontend Tiling

Conv and Q/K normalization process 16 independent token rows per tile. The
first three sequence tokens retain the history-aware row path; subsequent
chunks use the tiled path directly. Valid-row stores handle ragged tails.
The tap addition order, FP16 Conv input conversion, BF16 Conv and normalized
Q/K boundaries remain unchanged. Gate cumulative sums retain their sequential
FP32 scan. The frontend uses 46,336bytes of UB and adds no GM workspace.
MTE3 stores drain before the shared arena is reused by the gate scan.
