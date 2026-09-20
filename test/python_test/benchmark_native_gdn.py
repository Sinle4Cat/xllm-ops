"""Single-card graph microbenchmark; not a model throughput claim."""

import argparse
import hashlib
import importlib.util
import json
import tarfile
from functools import partial
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import custom_ops_lib
import torch
import torch_npu
from test_mega_gdn_native import call, make_case, prefill_call, prefill_case

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--legacy-archive", type=Path, required=True)
args = parser.parse_args()
output = Path(args.output)
assert not output.exists()
root = Path(__file__).resolve().parents[2]
archive = args.legacy_archive
with tarfile.open(archive) as tf:
    source = tf.extractfile("vllm-ascend/vllm_ascend/ops/triton/fla/gdn_kv.py").read()
assert (
    hashlib.sha256(source).hexdigest()
    == "1f43193788ea5cc366668ef42cf261e4836070dd98dc62561dc23b27a8aa9cd4"
)
legacy_file = root / "benchmark-legacy-gdn-kv.py"
legacy_file.write_bytes(source)
spec = importlib.util.spec_from_file_location("legacy_gdn_kv", legacy_file)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
init_device_properties_triton()


def capture(fn):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = fn()
    return graph, result


def measure(graph):
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(args.iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / args.iterations


results = {}
for label, lengths in [
    ("mtp4", (4, 4, 4, 4)),
    ("ragged", (4, 2, 1, 0)),
    ("ordinary", (1, 1, 1, 1)),
]:
    x = make_case(lengths)
    live_ids, live_starts, live_accepted = x.ids, x.starts, x.accepted
    x.ids = torch.zeros(32, 4, dtype=torch.int32).npu()
    x.ids[:4] = live_ids
    x.starts = torch.full((33,), sum(lengths), dtype=torch.int32).npu()
    x.starts[:5] = live_starts
    x.accepted = torch.ones(32, dtype=torch.int32).npu()
    x.accepted[:4] = live_accepted
    initial_c, initial_s = x.conv_back.clone(), x.ssm_back.clone()
    layer = SimpleNamespace(
        A_log=x.a_log, dt_bias=x.dt_bias, norm=SimpleNamespace(weight=x.norm)
    )
    conv_weights = x.weights.T.contiguous()
    meta = SimpleNamespace(
        spec_state_indices_tensor=x.ids,
        spec_decode_metadata=SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=x.starts, num_accepted_tokens=x.accepted
            )
        ),
    )
    old_fn = partial(
        legacy.fused_decode_kv,
        layer,
        custom_ops_lib,
        x.qkv,
        x.b,
        x.a,
        x.z,
        meta,
        conv_weights,
        x.conv,
        x.ssm,
        mtp=True,
    )
    old_graph, old_out = capture(old_fn)
    native32, new_out = capture(partial(call, x))
    capacity = min(32, x.qkv.size(0))
    x.ids, x.starts, x.accepted = (
        x.ids[:capacity],
        x.starts[: capacity + 1],
        x.accepted[:capacity],
    )
    native_bucket, bucket_out = capture(partial(call, x))
    graphs = {
        "legacy_capacity32": old_graph,
        "native_capacity32": native32,
        "native_bucket": native_bucket,
    }
    times = {k: [] for k in graphs}
    # Crossed order, same data and bank reset outside each measured window.
    for order in [
        list(graphs),
        list(reversed(graphs)),
        list(reversed(graphs)),
        list(graphs),
    ]:
        for name in order:
            x.conv_back.copy_(initial_c)
            x.ssm_back.copy_(initial_s)
            times[name].append(measure(graphs[name]))
    results[label] = {
        "capacity": capacity,
        "lengths": lengths,
        "unit": "us/layer/rank",
        "samples": times,
        "median": {k: median(v) for k, v in times.items()},
        "legacy_ssm_scratch_bytes": 32 * 4 * 12 * 128 * 128 * 4,
        "native_ssm_scratch_bytes": 0,
    }
    if args.profile and label == "mtp4":
        trace_dir = output.with_suffix("").as_posix() + "-profile"
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
        ) as prof:
            with torch.autograd.profiler.record_function("legacy_capacity32"):
                old_graph.replay()
            with torch.autograd.profiler.record_function("native_bucket"):
                native_bucket.replay()
            torch.npu.synchronize()
            prof.step()
        results[label]["profile"] = trace_dir

for length in (128, 512):
    case = prefill_case(length)
    x, read, write, lower, full, minus, matrices = case
    initial_c, initial_s = x.conv_back.clone(), x.ssm_back.clone()
    compact_ids = torch.arange(write.numel(), dtype=torch.int32).npu()

    def compact_prefill(
        x=x,
        write=write,
        compact_ids=compact_ids,
        lower=lower,
        full=full,
        minus=minus,
        matrices=matrices,
    ):
        native_c = x.conv.index_select(0, write.long()).contiguous()
        c = native_c[:, :3].contiguous()
        s = x.ssm.index_select(0, write.long()).contiguous()
        y = custom_ops_lib.mega_gdn_prefill_op(
            x.qkv,
            x.b,
            x.a,
            x.z,
            x.weights,
            c,
            x.a_log,
            x.dt_bias,
            compact_ids,
            compact_ids,
            compact_ids,
            compact_ids,
            s,
            lower,
            full,
            minus,
            x.starts,
            x.norm,
            matrices,
        )
        native_c[:, :3].copy_(c)
        x.conv.index_copy_(0, write.long(), native_c)
        x.ssm.index_copy_(0, write.long(), s)
        return y

    graph_a, out_a = capture(compact_prefill)
    graph_b, out_b = capture(partial(prefill_call, case))
    times = {"compact_adapter": [], "native": []}
    for name, graph in [
        ("compact_adapter", graph_a),
        ("native", graph_b),
        ("native", graph_b),
        ("compact_adapter", graph_a),
    ]:
        x.conv_back.copy_(initial_c)
        x.ssm_back.copy_(initial_s)
        times[name].append(measure(graph))
    results[f"prefill_{length}"] = {
        "lengths": [length, length + 3],
        "unit": "us/layer/rank",
        "samples": times,
        "median": {k: median(v) for k, v in times.items()},
    }
output.write_text(
    json.dumps(
        {
            "kind": "operator_microbenchmark",
            "iterations": args.iterations,
            "scope": "single card; graph latency excludes capture; not model throughput",
            "legacy_source_sha256": hashlib.sha256(source).hexdigest(),
            "results": results,
        },
        indent=2,
    )
)
print(output.read_text())
