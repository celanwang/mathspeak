#!/usr/bin/env python3
"""端云路由：按规则覆盖度决定公式朗读在端侧（规则引擎/端侧小模型）还是云端处理。

路由信号：对文本中每个公式段尝试 latex_speech.parse ——
- 全部可解析 -> 端侧路由（零网络开销，转换时延亚毫秒级）
- 任一不可解析 -> 云端升级（生产形态为云端大模型/规则增强版）

评测口径：
- coverage：公式段可解析比例
- escalation：升级云端的样本比例
- 模拟时延：端侧取流式评测实测转换耗时，云端按配置常量（默认 80ms）

Usage:
    .venv/bin/python router.py --sets data/test.jsonl data/ood_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import latex_speech as ls

CLOUD_LATENCY_MS = 80.0  # 云端大模型转换时延模拟值（网络+推理）


def route_text(text: str) -> tuple[str, list[str]]:
    """返回 (route, uncovered_latex)：route ∈ device|cloud。"""
    uncovered = []
    for _, _, latex in ls.find_formulas(text):
        try:
            ls.parse(latex)
        except ValueError:
            uncovered.append(latex)
    return ("cloud", uncovered) if uncovered else ("device", [])


def eval_set(path: str, device_ms: float) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n_device = 0
    formulas = 0
    covered = 0
    lat = []
    escalated_examples = []
    for r in rows:
        src = r["prompt"].split("原句：", 1)[1]
        route, uncovered = route_text(src)
        spans = ls.find_formulas(src)
        formulas += len(spans)
        for _, _, latex in spans:
            try:
                ls.parse(latex)
                covered += 1
            except ValueError:
                pass
        if route == "device":
            n_device += 1
            lat.append(device_ms * max(1, len(spans)))
        else:
            lat.append(CLOUD_LATENCY_MS)
            if len(escalated_examples) < 10:
                escalated_examples.append(uncovered[0])
    n = len(rows)
    return {
        "file": path,
        "n": n,
        "device_rate": n_device / n,
        "escalation_rate": 1 - n_device / n,
        "formula_coverage": covered / max(1, formulas),
        "avg_added_latency_ms": sum(lat) / max(1, n),
        "escalated_examples": escalated_examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+",
                    default=["data/test.jsonl", "data/ood_test.jsonl"])
    ap.add_argument("--device-ms", type=float, default=0.5,
                    help="端侧单公式转换时延（模拟值，实测见 stream_report）")
    ap.add_argument("--out", default="results/router_report.md")
    args = ap.parse_args()

    stats = [eval_set(p, args.device_ms) for p in args.sets]
    lines = [
        "# 端云路由评测报告",
        "",
        "路由信号：公式段能否被 latex_speech 规则解析全覆盖。",
        f"时延模型：端侧 {args.device_ms}ms/公式（可换成 stream_report 实测值），"
        f"云端 {CLOUD_LATENCY_MS:.0f}ms/样本（网络+大模型推理模拟）。",
        "",
        "| 数据集 | 样本数 | 端侧路由率 | 云端升级率 | 公式覆盖率 | 平均新增时延 |",
        "|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {Path(s['file']).name} | {s['n']} | {s['device_rate']:.1%} "
            f"| {s['escalation_rate']:.1%} | {s['formula_coverage']:.1%} "
            f"| {s['avg_added_latency_ms']:.1f}ms |")
        print(f"{s['file']}: 端侧 {s['device_rate']:.1%}, "
              f"覆盖率 {s['formula_coverage']:.1%}, 升级率 {s['escalation_rate']:.1%}")
    for s in stats:
        if s["escalated_examples"]:
            lines += ["" , f"## {Path(s['file']).name} 升级样例（未覆盖公式）", ""]
            for e in s["escalated_examples"]:
                lines.append(f"- `{e}`")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {args.out}")


if __name__ == "__main__":
    main()
