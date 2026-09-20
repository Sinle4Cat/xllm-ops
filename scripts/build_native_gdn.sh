#!/usr/bin/env bash
# Run after loading CANN and the target Python/Torch NPU environment.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-"$repo/build/native-gdn"}
mkdir -p "$out"
out=$(realpath "$out")
# Kernel-generation .done markers in the upstream build are not invalidated
# reliably by source/header edits. Require a fresh build tree for provenance.
test ! -e "$out/cann-build" || { echo "Choose a fresh output directory: $out/cann-build already exists" >&2; exit 1; }
export OPS_CPU_NUMBER=${OPS_CPU_NUMBER:-8}
export MAX_JOBS=${MAX_JOBS:-8}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-8}
test -d "$repo/third_party/pto-isa/include"
test -d "$repo/third_party/catlass/include"
export CPATH="$repo/third_party/catlass/include:${CPATH:-}"
# Standalone SDK regression oracles require the array-index convolution ABI.
# Keep it optional: the serving framework has a different tensor-index ABI
# under the same operator name.
ops='mega_gdn_native_decode;mega_gdn_native_prefill;mega_gdn_decode;mega_gdn_mtp_decode;mega_gdn_prefill_op;mega_chunk_gdn'
if [[ "${2:-}" == --legacy-oracles ]]; then
  ops="causal_conv1d;$ops"
elif [[ -n "${2:-}" ]]; then
  echo "Unknown option: $2" >&2
  exit 1
fi
cd "$repo/xllm_ops"
bash build.sh --pkg \
  --ops="$ops" \
  --soc=ascend910b --build-dir="$out/cann-build"
# The upstream incremental build does not invalidate this summary when the
# selected operator set grows. Regenerate it before packaging, then verify.
python "$repo/cmake/scripts/util/ascendc_ops_config.py" \
  -p "$out/cann-build/binary/ascend910b/bin" -s ascend910b
cmake --build "$out/cann-build" --target package -j "$CMAKE_BUILD_PARALLEL_LEVEL"
shopt -s nullglob
installers=("$out/cann-build"/cann-ops-xllm*.run)
test "${#installers[@]}" -eq 1
bash "${installers[0]}" --install-path="$out/install"
python - "$out/install" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
path = root / 'vendors/custom_xllm_math/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json'
data = json.loads(path.read_text())
assert all(name in data for name in ('MegaGdnNativeDecode', 'MegaGdnNativePrefill', 'MegaGdnMtpDecode', 'MegaGdnPrefillOp'))
PY
cd "$repo/test/python_test"
python setup_native.py build_ext --build-lib "$out/binding" --build-temp "$out/binding-tmp"
printf 'Native GDN package: %s\nBinding: %s\n' "$out/install" "$out/binding"
