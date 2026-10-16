#!/usr/bin/env python3
"""Three-group evaluation on the frozen test set.

Groups (identical instruction header, temperature 0):
    A  base       ERNIE-4.5-0.3B-PT-4bit, zero-shot
    B  fewshot    ERNIE-4.5-0.3B-PT-4bit, instruction + 3 fixed train examples
    C  lora       ERNIE-4.5-0.3B-PT-bf16 + adapters/lora (produced by Phase 2)

Metrics (double counting):
    exact    exact match after whitespace/punctuation normalization
    fragment the expected spoken formula appears in the output

Usage:
    .venv/bin/python evaluate.py --groups A B C
    .venv/bin/python evaluate.py --groups A B --limit 20   # smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

BASE_4BIT = "mlx-community/ERNIE-4.5-0.3B-PT-4bit"
BASE_BF16 = "mlx-community/ERNIE-4.5-0.3B-PT-bf16"
ADAPTER = "adapters"

MAX_TOKENS = 96

PUNCT_MAP = str.maketrans({
    ",": "，", ".": "。", ":": "：", ";": "；",
    "!": "！", "?": "？", "(": "（", ")": "）",
})


def normalize(s: str) -> str:
    """去空白 + 半角标点转全角 + 小写字母归一，用于公平比对。"""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(PUNCT_MAP)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def first_line(s: str) -> str:
    """PT/LoRA 输出统一后处理：截断特殊 token 与换行后的内容。"""
    s = s.split("<|")[0]
    s = s.split("\n")[0]
    return s.strip()


def clean_output(s: str) -> str:
    """去掉 chat 模板残留的角色前缀（三组统一处理，不影响公平性）。"""
    s = first_line(s)
    s = re.sub(r"^(Assistant|助手|输出|答案|改写后的整句)\s*[:：]\s*", "", s)
    return s.strip()


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fewshot_prefix(train: list[dict]) -> str:
    """从训练集按类别特征各取 1 条，构造 3-shot 前缀（不使用 test 数据）。"""
    markers = {"fraction": "\\frac{", "radical": "\\sqrt{", "script": "^2"}
    picked = {}
    for row in train:
        for cat, mark in markers.items():
            if cat not in picked and mark in row["prompt"]:
                picked[cat] = row
    assert len(picked) == len(markers), "训练集缺少 few-shot 所需类别"
    blocks = []
    for cat in ("fraction", "radical", "script"):
        row = picked[cat]
        src = row["prompt"].split("原句：", 1)[1]
        blocks.append(f"原句：{src}\n{row['completion']}")
    return "\n".join(blocks) + "\n"


def run_group(model, tokenizer, rows: list[dict], prefix: str = "") -> list[str]:
    from mlx_lm import generate
    from mlx_lm.generate import make_sampler

    sampler = make_sampler(temp=0.0)
    outs = []
    for row in rows:
        src = row["prompt"].split("原句：", 1)[1]
        prompt = f"{prefix}原句：{src}\n" if prefix else row["prompt"] + "\n"
        text = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            sampler=sampler,
            verbose=False,
        )
        outs.append(clean_output(text))
    return outs


def score(outputs: list[str], refs: list[str], speeches: list[str]) -> dict:
    exact = frag = 0
    for out, ref, sp in zip(outputs, refs, speeches):
        n_out = normalize(out)
        if n_out == normalize(ref):
            exact += 1
        if normalize(sp) in n_out:
            frag += 1
    n = len(refs)
    return {"exact": exact / n, "fragment": frag / n, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["A", "B", "C"],
                    choices=["A", "B", "C"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条（冒烟用）")
    ap.add_argument("--out", default="results/eval_report.md")
    args = ap.parse_args()

    rows = load_jsonl(f"{args.data}/test.jsonl")
    meta = load_jsonl(f"{args.data}/test_meta.jsonl")
    train = load_jsonl(f"{args.data}/train.jsonl")
    assert [r["prompt"] for r in rows] == [m["prompt"] for m in meta], "test/meta 不对齐"
    if args.limit:
        rows, meta = rows[: args.limit], meta[: args.limit]
    refs = [m["completion"] for m in meta]
    speeches = [m["speech"] for m in meta]

    prefix = fewshot_prefix(train) if "B" in args.groups else ""
    results: dict[str, dict] = {}
    outputs: dict[str, list[str]] = {}

    from mlx_lm import load

    if "A" in args.groups or "B" in args.groups:
        model, tok = load(BASE_4BIT)
        if "A" in args.groups:
            outputs["A"] = run_group(model, tok, rows)
            results["A"] = score(outputs["A"], refs, speeches)
        if "B" in args.groups:
            outputs["B"] = run_group(model, tok, rows, prefix=prefix)
            results["B"] = score(outputs["B"], refs, speeches)
        del model
    if "C" in args.groups:
        model, tok = load(BASE_BF16, adapter_path=ADAPTER)
        outputs["C"] = run_group(model, tok, rows)
        results["C"] = score(outputs["C"], refs, speeches)
        del model

    for g, r in results.items():
        print(f"group {g}: exact={r['exact']:.1%} fragment={r['fragment']:.1%} (n={r['n']})")

    write_report(args.out, results, rows, meta, outputs)


def write_report(path: str, results: dict, rows: list[dict], meta: list[dict],
                 outputs: dict[str, list[str]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 公式朗读转换评测报告",
        "",
        "| 组 | 配置 | 完全匹配率 | 关键片段匹配率 |",
        "|---|---|---|---|",
    ]
    desc = {
        "A": "基座直调（4bit, zero-shot）",
        "B": "prompt 工程（4bit, 3-shot）",
        "C": "LoRA 后（bf16 + adapters）",
    }
    for g in ("A", "B", "C"):
        if g in results:
            r = results[g]
            lines.append(f"| {g} | {desc[g]} | {r['exact']:.1%} | {r['fragment']:.1%} |")
    lines += ["", "## Badcase（按类别归类）", ""]
    for g in ("A", "B", "C"):
        if g not in outputs:
            continue
        lines.append(f"### 组 {g}")
        lines.append("")
        bad = []
        for row, m, pred in zip(rows, meta, outputs[g]):
            if normalize(pred) != normalize(m["completion"]):
                bad.append((m["category"], row["prompt"].split("原句：", 1)[1],
                            m["completion"], pred))
        lines.append(f"错误 {len(bad)}/{len(rows)} 条")
        lines.append("")
        for cat, src, ref, out_text in bad:
            lines.append(f"- **[{cat}]** `{src}`")
            lines.append(f"  - 参考：{ref}")
            lines.append(f"  - 输出：{out_text}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
