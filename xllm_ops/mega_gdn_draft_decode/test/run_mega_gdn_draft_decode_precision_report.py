#!/usr/bin/env python3
"""Run MegaGdnDraftDecode precision cases and generate JSON/Markdown reports."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from test_mega_gdn_draft_decode_precision import (  # noqa: E402
    ALL_CASES,
    PROFILE_NAME,
    THRESHOLD,
    run_precision_case,
)


def _format_result_rows(results: list[dict[str, object]]) -> list[str]:
    rows = []
    for index, result in enumerate(results, 1):
        geometry = f"{result['num_k_heads']}×{result['num_v_heads']}"
        rows.append(
            f"| {index} | {result['name']} | {result['batch_size']} | "
            f"{result['total_tokens']} | {geometry} | "
            f"{result['MERE']:.3e} | {result['MARE']:.3e} | "
            f"{result['max_abs_err']:.3e} | {result['cosine_sim']:.10f} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    return rows


def _write_markdown(results: list[dict[str, object]], path: Path) -> None:
    passed = sum(bool(result["passed"]) for result in results)
    failed = len(results) - passed
    lines = [
        "# MegaGdnDraftDecode 算子精度验证报告",
        "",
        f"**测试时间**：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "**测试平台**：Ascend 910B（A2）",
        "**参考基线**：FP32 PyTorch 小算子拼接，保留四个 BF16 舍入合同点",
        f"**dtype profile**：{PROFILE_NAME}",
        "**精度标准**：生态算子 MERE/MARE；按 BF16 端到端合同判定",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 总用例数 | {len(results)} |",
        f"| 通过数 | {passed} |",
        f"| 失败数 | {failed} |",
        f"| 通过率 | {100.0 * passed / len(results):.2f}% |",
        "",
        "## 精度阈值",
        "",
        "通过条件：`MERE < 2^-7` 且 `MARE < 10 × 2^-7`。SSM State "
        "虽然以 FP32 存储，但每一步由 BF16 Q/K/V、g、beta 驱动，因此使用 "
        "BF16 融合链端到端阈值。",
        "",
        "| dtype profile | MERE 上限 | MARE 上限 |",
        "|---|---:|---:|",
        f"| Mixed BF16/FP32 | {THRESHOLD:.6e} | {10 * THRESHOLD:.6e} |",
    ]

    for category in ("Typical", "General", "Boundary"):
        category_results = [
            result for result in results if result["category"] == category
        ]
        lines.extend(
            [
                "",
                f"## {category} 测试结果",
                "",
                "| # | 用例 | B | T | Hk×Hv | MERE | MARE | MaxAbsErr | CosSim | 结果 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
                *_format_result_rows(category_results),
            ]
        )

    max_mere = max(float(result["MERE"]) for result in results)
    max_mare = max(float(result["MARE"]) for result in results)
    min_cosine = min(float(result["cosine_sim"]) for result in results)
    lines.extend(
        [
            "",
            "## 关键发现",
            "",
            f"1. 40 个固定混合 dtype 用例通过 {passed} 个，失败 {failed} 个。",
            f"2. 最大 MERE 为 `{max_mere:.3e}`，最大 MARE 为 `{max_mare:.3e}`。",
            f"3. 最小 CosineSim 为 `{min_cosine:.10f}`；全零用例按双零向量记为 1。",
            "4. 覆盖 1×1 到 16×64 head geometry、B=1..32、FLA/non-FLA、"
            "Prefix fork、fresh/mixed validity 和关键 gate 数值边界。",
            "5. `heads_16x64` 强制同一 AIV 连续处理多个 recurrent head，覆盖"
            "最终 State 的 MTE3→MTE2 防回归依赖。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    results = []
    for index, case in enumerate(ALL_CASES, 1):
        result = run_precision_case(case)
        results.append(result)
        print(
            f"[{'PASS' if result['passed'] else 'FAIL'}] {index:02d}/"
            f"{len(ALL_CASES)} {case.name}: MERE={result['MERE']:.3e}, "
            f"MARE={result['MARE']:.3e}, MaxAbs={result['max_abs_err']:.3e}"
        )

    json_path = SCRIPT_DIR / "mega_gdn_draft_decode_precision_report.json"
    markdown_path = SCRIPT_DIR / "mega_gdn_draft_decode_precision_report.md"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(results, markdown_path)

    passed = sum(bool(result["passed"]) for result in results)
    print(f"Total={len(results)} Passed={passed} Failed={len(results) - passed}")
    print(f"JSON={json_path}")
    print(f"Markdown={markdown_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
