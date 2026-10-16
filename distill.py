#!/usr/bin/env python3
"""教师蒸馏：21B 教师模型为长尾公式生成朗读样本，硬检查过滤后入库。

管线：
1. 从 ood_pool.jsonl 取候选（规则引擎已给出参考朗读）
2. 教师用 3-shot（示例来自规则引擎，展示约定）为每条原句生成朗读
3. 硬检查过滤（verify.check_row 守恒检查 + 角色前缀清洗 + 非空）
4. 通过的写入 ood_train.jsonl，作为第二轮训练数据

Usage:
    .venv/bin/python distill.py --pool data/ood_pool.jsonl \
        --out data/ood_train.jsonl --limit 0
"""

from __future__ import annotations

import argparse
import json
import random
import re

import latex_speech as ls
import verify

from evaluate import clean_output, load_jsonl

TEACHER = "mlx-community/ERNIE-4.5-21B-A3B-PT-4bit"
MAX_TOKENS = 160

FEWSHOT_LATEX = [
    ("代入得 \\lim_{x \\to 0} \\frac{\\sin x}{x}。",
     "代入得 当 x 趋向于 零 时，x 分之 sin x 的极限。"),
    ("解得 \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}。",
     "解得 矩阵，第 一 行 一、二，第 二 行 三、四。"),
    ("由此可知 50\\%。", "由此可知 百分之 五零。"),
]


def build_prompt(src: str) -> str:
    shots = "\n".join(f"原句：{a}\n{b}" for a, b in FEWSHOT_LATEX)
    return (f"将讲解中的公式改写为适合语音朗读的中文口语，只输出改写后的整句：\n"
            f"{shots}\n原句：{src}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/ood_pool.jsonl")
    ap.add_argument("--out", default="data/ood_train.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    pool = load_jsonl(args.pool)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    if args.limit:
        pool = pool[: args.limit]

    from mlx_lm import generate, load
    from mlx_lm.generate import make_sampler

    model, tok = load(TEACHER)
    sampler = make_sampler(temp=0.0)

    accepted, rejected = [], {"conservation": 0, "empty": 0, "echo": 0}
    for row in pool:
        src = row["prompt"].split("原句：", 1)[1]
        text = generate(model, tok, prompt=build_prompt(src),
                        max_tokens=MAX_TOKENS, sampler=sampler, verbose=False)
        out = clean_output(text)
        if not out:
            rejected["empty"] += 1
            continue
        if out == src or out.startswith("原句："):
            rejected["echo"] += 1
            continue
        ok, missing = verify.check_row(src, out)
        if not ok:
            rejected["conservation"] += 1
            continue
        accepted.append({"prompt": row["prompt"], "completion": out,
                         "category": row.get("category", ""),
                         "rule_ref": row.get("completion", "")})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"接受 {len(accepted)}/{len(pool)}，拒绝: {rejected}")
    from collections import Counter
    print(Counter(r["category"] for r in accepted))
    for r in accepted[:6]:
        print("---", r["category"])
        print("规则参考:", r["rule_ref"])
        print("教师产出:", r["completion"])


if __name__ == "__main__":
    main()
