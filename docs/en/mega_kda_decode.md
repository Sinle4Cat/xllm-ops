# MegaKdaDecode

Experimental GLM-5.3-Flash vector KDA Conv + recurrent decode operator for
Ascend 910B. It follows `mega_gdn_prefill_op`'s OpDef, separate shape inference,
Host tiling, PTO Device entry, CMake discovery and generated ACLNN integration.
The GLM model has an environment-gated integration through
`KdaLinearAttentionMixin`; the default remains the `fla_npu` composition.
`XLLM_KDA_KERNEL=mega_kda` selects supported native prefill/ordinary decode and MTP verify.
Unsupported routes are selected before computation; execution errors propagate.
Operator validation does not establish model accuracy or performance acceptance.

For non-MTP engines, `XLLM_KDA_PREFILL_BACKEND=fla` retains FLA prefill while
using native ordinary decode. With this configuration,
`XLLM_KDA_DECODE_NORM=public` selects full public compatibility (mode 3); the
legacy environment name is retained, although it controls more than normalization.
`native` remains the default value and selects mode 0. The FLA-prefill compatibility
configuration requires `XLLM_KDA_KERNEL=mega_kda` and verify width 1. Public
compatibility requires that explicit FLA-prefill configuration. The MTP mode=1
integration maps V3 per-token snapshots to the operator checkpoint tables.
Supported state layouts use the native operator; other shapes retain the FLA V3
fallback. Operator validation does not establish model accuracy or performance.

## Single Operator

`OP_ADD(MegaKdaDecode)` generates one ACLNN API family:
`aclnnMegaKdaDecodeGetWorkspaceSize` and `aclnnMegaKdaDecode`.
There is one global Device entry, `mega_kda_decode`:

| mode | tilingKey | Semantics |
| --- | --- | --- |
| 0 | 1 | Dense ordinary decode, one token per request |
| 1 | 2 | MTP verify, restore accepted checkpoint, publish every query checkpoint |
| 3 | 3 | Public compatibility ordinary decode, one token per request |

A one-token MTP call still uses key 2. Host selects the key from the explicit
mode and static capacity, not from accepted-count values. Token offsets,
accepted counts and all slot mappings remain Device inputs. Sampling and the
accept/reject decision are outside this operator. Mixed ordinary/MTP semantics
within one launch are not supported. Modes 0 and 3 both require capacity 1, absent
accepted counts, history 3 and unit offsets; mode 3 is never interpreted as MTP.

## Tensor Contract

All tensors must be contiguous ND, on the same NPU. `D=128`, `1 <= H <= 128`,
`C=3*H*128`, `1 <= M <= 17`. B and N must be positive. M=1 for ordinary decode.
Model MTP K denotes K speculative tokens plus the current token: M=K+1.
Thus model MTP1 through MTP16 uses capacities 2 through 17. Capacity 1 remains
supported for single-query MTP verification.
Packed QKV order is all Q heads, all K heads, all V heads.

| Parameter | Shape | Dtype |
| --- | --- | --- |
| qkv | [N,C] | BF16 |
| gate | [N,H,128] | BF16 |
| beta | [N,H] | BF16 |
| a_log | [H] | FP32 |
| gate_bias | [H,128] | FP32 |
| conv_weight | [4,C] | BF16 |
| conv_bias (optional) | [C] | BF16 |
| conv_state_in | [conv_input_slots,L,C] | BF16 |
| ssm_state_in | [ssm_input_slots,H,V=128,K=128] | FP32 |
| cu_seqlens | [B+1] | int32 |
| conv_read_indices, conv_write_indices | [B] | int32 |
| ssm_read_indices, ssm_write_indices | [B,M] | int32 |
| num_accepted_tokens (MTP only) | [B] | int32 |
| output | [N,H,128] | BF16 |
| conv_state_out | [conv_output_slots,L,C] | BF16 |
| ssm_state_out | [ssm_output_slots,H,128,128] | FP32 |

`L=3` for ordinary decode; `L=M+2` for MTP. Input and output pool capacities
are independent. Required attributes `conv_output_slots` and `ssm_output_slots`
are inferred from caller-provided outputs by the test binding.
Conv weights use the same prepacked `[4,C]` layout as MegaGDN; transpose an
upstream `[C,1,4]` weight once after loading/TP sharding, not per invocation.
The model keeps this BF16 pack in a nonpersistent buffer and reuses its storage
on a compatible reload.

The PyBind entry `custom_ops_lib.mega_kda_decode` takes inputs in table order,
then `mode`, `max_query_tokens`, `output`, `conv_state_out`, `ssm_state_out`.
Pass `None` for absent Conv bias and, in ordinary decode, accepted counts.
It returns the same three output tensors without cloning state or projected QKV.

Unlike `MegaKdaPrefill`, beta here is the raw BF16 projection and sigmoid is
computed inside the operator. Prefill consumes FP32 post-sigmoid beta. A decode
read slot of -1 means inactive, not prefill's zero initial state. Callers must
honor each operator's contract when switching between phases.

## State And Numerical Semantics

- Slot zero is valid. A read slot of -1 marks an inactive request. Cold state
  uses a real zero-initialized slot. A write slot of -1 discards that state write
  without suppressing computation or the other state output.
- Offsets start at zero, end at N and are nondecreasing. Ordinary offsets are
  exactly `[0,1,...,B]`. MTP allows empty and ragged requests of length <= M.
- MTP accepted counts are in `[1,M]`, using the upstream convention: initial
  SSM comes from `ssm_read_indices[b, accepted-1]`; Conv starts at history offset
  `accepted-1`. The scheduler must provide valid previous checkpoints/history.
- Width-four Conv accumulates in FP32, rounds the accumulation to BF16 before
  SiLU, then rounds SiLU to BF16 before recurrence. Beta sigmoid also rounds to
  BF16 before returning to FP32. Q scale is `128**-0.5`; state is FP32 `[V,K]`.
  Decay is mathematically `exp(-5 * sigmoid(exp(a_log) * (gate + gate_bias)))`.
  This is KDA vector decay, not GDN scalar decay. The mode differences below
  are floating-point contracts, including operation order.
- For a query of length Q, Conv writes `[old[accepted:accepted+2], raw_qkv]`
  into the first `Q+2` positions. The remaining capacity tail is untouched.
  SSM publishes the state after each query token into its own write slot.
- All three output allocations must be disjoint from every input and from
  each other. The binding rejects storage aliasing and noncontiguous buffers.
  Input states and projected QKV are never written by the Device body.
- Nonnegative active destinations must be unique across requests/checkpoints.
  Read sharing is allowed. The scheduler owns this invariant; Host tiling
  cannot validate device slot contents without a D2H. The CPU oracle rejects
  duplicate destinations. Native bounds guards do not constitute a complete
  metadata-validation API and invalid metadata must not be submitted.
- Callers initialize output buffers as needed. Inactive output rows, unwritten
  state slots and Conv tails are not initialized by the kernel. Preservation
  through the actual ACLNN/Graph call remains a required native test.

| Numerical stage | Native/MTP (modes 0/1) | Public compatibility (mode 3) |
| --- | --- | --- |
| Conv input/history | BF16→FP16→FP32 narrowing | BF16→FP32 directly |
| Q/K normalization | `sqrt(sum(x*x)+1e-6)` | No epsilon; zero vectors remain zero, without `0/0` |
| Gate | Divide `-5` by the sigmoid denominator | Compute sigmoid as a reciprocal, then multiply by `-5` |
| 64-row recurrence reduction | One 64-row group | Two independent 32-row groups |

The FP16 omission permits finite BF16 inputs outside the FP16 range. No epsilon
changes the treatment of very small nonzero vectors as well as exact zeros.
These four differences are selected together by mode 3/key 3; they are not four
independent switches. Shared BF16 Conv/SiLU/beta boundaries still apply.

The Device uses only AIV and zero user workspace. Each task owns 32 or 64 V
rows and processes a request serially, with 62,368 or 120,032bytes of explicitly
assigned UB respectively. The 64-row variant is selected when its total shard
count, `2 * B * H`, covers every launched AIV block; smaller workloads keep the
32-row variant. Wider tiles halve duplicated Conv/Q/K/gate work per request/head.
Only the first V shard writes Conv state, and state input/output ownership remains
unchanged. FP32 reduction grouping can differ between variants, so numerical
equivalence uses the same Golden tolerances; fixed-input Graph replay remains
byte-identical within each variant. This does not imply a speedup for every batch,
head count, active-row distribution or MTP capacity.

## Build And Test

```bash
# From the xllm-ops checkout, after sourcing CANN:
bash build.sh --op-name mega_kda_decode

# Isolated package build, avoiding existing output directories:
bash xllm_ops/build.sh --pkg --ops=mega_kda_decode --soc=ascend910b -j4 \
  --build-dir="$BUILD_ROOT/build" --output-dir="$BUILD_ROOT/output" \
  --build-out-dir="$BUILD_ROOT/build_out"

# CPU-only contract and Host policy checks:
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 python -m pytest -q \
  test/python_test/test_mega_kda_decode.py

# After installing the package in an isolated OPP and building the test extension:
MEGA_KDA_TEST_NPU=0 python -m pytest -q test/python_test/test_mega_kda_decode.py
```

The native tests are opt-in so CPU collection does not initialize an NPU or
consume a busy device. They include fixed20 byte-identical Graph replay, independent
CPU/NPU state carry20, changed-metadata Graph replay, padding, different pool
capacities, accepted-prefix restoration and alias rejection. Conv copies and
untouched values require byte equality, including signed zero. The independent
CPU oracle accumulates Conv, normalization, gate and recurrence in FP64, retaining
the mode-specific FP16 boundary and common BF16 accumulation-before-SiLU,
SiLU-output and beta-sigmoid boundaries. It is a mathematical oracle rather
than a bitwise model of device FP32 reductions. Independent FP32 stage checks
cover reciprocal-then-multiply gate order, beta rounding, zero/near-zero norms
and finite BF16 inputs above the FP16 range. Shape/dtype mismatch
or nonfinite outputs, golden values or derived errors fail closed.
Normal relative errors and near-zero absolute errors (split at `1e-3`) and their
exceeding fractions are reported as diagnostics. The existing pointwise smoke
gate is unchanged: output
`rtol=1e-2, atol=1e-3`, and FP32 state `rtol=1e-3, atol=1e-4`.
These smoke thresholds are not a formal model-accuracy or Triton-equivalence
acceptance report. Fresh native-versus-frozen-Triton validation and baseline
warmup-convergence qualification are still required before performance claims.

The explicit TP matrix maps GLM's 64 KDA heads to local H=64/32/16/8/4/2/1
for TP1/2/4/8/16/32/64. Every size covers ordinary decode and every integer
MTP query capacity from 1 through 17 (model MTP1 through MTP16), plus mode 3,
with CPU oracle checks, dense fixed20 Graph replay, changed-metadata replay20
and independent state carry20. Existing batch/ragged/padding cases remain.
Host policy checks also cover all seven head counts. Select these cases with
`pytest -k tp`, or a single size with `-k tp8`. They execute one rank's shape
on one device; they do not validate distributed TP collectives or model memory.
Dedicated mode 3 replay/carry/changed-metadata and numerical edge cases cover
H=8 at B=1/4/5 and H=3 at B=2. The first two H=8 cases use batched Conv;
B=5 and H=3 use the general path. To qualify both V-row variants, set
`MEGA_KDA_TEST_AIV_CORES` from the selected platform's `GetCoreNumAiv` result
and run `-k public_compat_row_dispatch`; it records launched blocks and derives
the selected 32/64 rows from that count. Shapes alone do not prove this coverage.

Only Ascend 910B is registered. No A3/A5 support or speedup is claimed.

For local H=8 and batch B<=4, the kernel batches Q/K/V Conv preprocessing.
Ordinary one-token decode also batches the four-tap input/weight loads and raw
Conv-state publication. The 32/64-row state task map and floating-point
conversion boundaries remain unchanged. Explicit UB is 83,872/141,536bytes
for ordinary decode and 70,048/127,712bytes for MTP; other shapes retain the
previous path. These native specializations do not establish model TPOT gains.
