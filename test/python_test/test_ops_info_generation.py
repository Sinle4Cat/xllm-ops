"""Regression coverage for ops-info generation with selected registry groups."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("default", "inner", "exclude")
GENERATED_SOURCES = {
    "default": "generate_aclnn_srcs",
    "inner": "generate_aclnn_inner_srcs",
    "exclude": "generate_exclude_proto_srcs",
}
REGISTRY_TARGETS = {
    "default": "opbuild_gen_default",
    "inner": "opbuild_gen_inner",
    "exclude": "opbuild_gen_exc",
}
INI_SUBDIRS = {"default": "", "inner": "inner", "exclude": "exc"}


def run_command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("cmake") is None or shutil.which("make") is None,
                    reason="CMake and Unix Makefiles are required")
@pytest.mark.parametrize(
    "groups",
    [("default",), ("inner",), ("exclude",), ("default", "inner"),
     ("default", "exclude"), ("inner", "exclude"), GROUPS],
    ids=["default", "inner", "exclude", "default-inner", "default-exclude",
         "inner-exclude", "all"],
)
def test_ops_info_selected_registries(tmp_path, groups):
    source = (ROOT / "cmake/func.cmake").read_text()
    match = re.search(r"(?ms)^function\(add_ops_info_target\).*?^endfunction\(\)", source)
    assert match is not None
    (tmp_path / "ops_info.cmake").write_text(match.group())
    build = tmp_path / "build"
    autogen = build / "autogen"
    identity = tmp_path / "source.identity"
    identity.write_text("ops-info-regression")
    lines = [
        "cmake_minimum_required(VERSION 3.20)",
        "project(OpsInfoRegistryTest NONE)",
        f'set(HI_PYTHON "{sys.executable}")',
        f'set(ASCENDC_CMAKE_UTIL_DIR "{ROOT / "cmake/scripts/util"}")',
        'set(ASCEND_AUTOGEN_DIR "${CMAKE_BINARY_DIR}/autogen")',
        'set(base_aclnn_binary_dir "${ASCEND_AUTOGEN_DIR}")',
        'set(CUSTOM_DIR "${CMAKE_BINARY_DIR}/custom")',
        f'set(XLLM_OPS_SOURCE_IDENTITY_DEPENDENCY "{identity}")',
        "set(VENDOR_NAME custom)",
        "add_custom_target(generate_ops_info ALL)",
    ]
    source_inis = {}
    for group in GROUPS:
        target = REGISTRY_TARGETS[group]
        if group not in groups:
            lines.append(f"add_custom_target({target})")
            continue
        ini = tmp_path / f"{group}.ini"
        ini.write_text(
            f"[{group.title()}Op]\n"
            "input0.name=x\ninput0.paramType=required\n"
            "input0.dtype=float32\ninput0.format=ND\n"
            "output0.name=y\noutput0.paramType=required\n"
            "output0.dtype=float32\noutput0.format=ND\n"
            "precision_reduce.flag=false\n"
        )
        source_inis[group] = ini
        ini_dir = autogen / INI_SUBDIRS[group]
        generated_ini = ini_dir / "aic-ascend910b-ops-info.ini"
        marker = build / f"{group}.stamp"
        lines.extend([
            f'set({GENERATED_SOURCES[group]} "{group}.cpp")',
            f'add_custom_command(OUTPUT "{marker}"',
            f'  COMMAND "${{CMAKE_COMMAND}}" -E make_directory "{ini_dir}"',
            f'  COMMAND "${{CMAKE_COMMAND}}" -E copy "{ini}" "{generated_ini}"',
            f'  COMMAND "${{CMAKE_COMMAND}}" -E touch "{marker}"',
            f'  BYPRODUCTS "{generated_ini}"',
            f'  DEPENDS "{ini}")',
            f'add_custom_target({target} DEPENDS "{marker}")',
        ])
    lines.extend([
        'include("${CMAKE_CURRENT_SOURCE_DIR}/ops_info.cmake")',
        "add_ops_info_target(COMPUTE_UNIT ascend910b)",
    ])
    (tmp_path / "CMakeLists.txt").write_text("\n".join(lines) + "\n")
    run_command(["cmake", "-S", str(tmp_path), "-B", str(build),
                 "-G", "Unix Makefiles"])
    command = ["cmake", "--build", str(build), "--parallel", "2"]
    run_command(command)
    output = autogen / "aic-ascend910b-ops-info.json"
    assert set(json.loads(output.read_text())) == {f"{g.title()}Op" for g in groups}
    packaged = build / "custom/op_impl/ai_core/tbe/config/ascend910b" / output.name
    assert packaged.read_bytes() == output.read_bytes()
    for group in set(GROUPS) - set(groups):
        assert not (autogen / INI_SUBDIRS[group] / "aic-ascend910b-ops-info.ini").exists()
    original_mtime = output.stat().st_mtime_ns
    run_command(command)
    assert output.stat().st_mtime_ns == original_mtime
    changed = groups[0]
    ini = source_inis[changed]
    ini.write_text(ini.read_text().replace("precision_reduce.flag=false",
                                         "precision_reduce.flag=true"))
    os.utime(output, (1, 1))
    run_command(command)
    assert json.loads(output.read_text())[f"{changed.title()}Op"]["precision_reduce"]["flag"] == "true"
    assert packaged.read_bytes() == output.read_bytes()
