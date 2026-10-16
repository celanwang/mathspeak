#!/usr/bin/env python3
"""长尾 OOD 基准集生成器：8 类训练分布之外的公式类别。

与 generate_data.py 的区别：参考朗读直接由 latex_speech（朗读规范唯一事实源）
产出，覆盖 limit/log/matrix/determinant/factorial-binom/geometry/set/vector/
percent/function 等长尾类别。用于量化"规则内 100%"之外的分布外差距，
并作为教师蒸馏第二轮的训练输入池。

产物：
    data/ood_test.jsonl   冻结 OOD 测试集（默认 200 条，10 类 × 20）
    data/ood_pool.jsonl   教师蒸馏输入池（候选 prompt，含 rule 参考）
    data/ood_meta.jsonl   ood_test 的 category/latex/rule_speech

Usage:
    .venv/bin/python gen_ood_data.py --out data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import latex_speech as ls
from generate_data import INSTRUCTION, TEMPLATES

VARS = ["x", "y", "a", "b", "c", "n"]


def gen_limit(rng):
    v = rng.choice(VARS)
    n = rng.randint(0, 5)
    f = rng.choice([f"\\frac{{\\sin {v}}}{{{v}}}", f"\\frac{{{v}^2 - {n}}}{{{v}}}",
                    f"\\frac{{\\ln(1 + {v})}}{{{v}}}"])
    return f"\\lim_{{{v} \\to {n}}} {f}"


def gen_log(rng):
    if rng.random() < 0.4:
        return f"\\ln {rng.choice(VARS)}"
    b = rng.choice(["2", "3", "10"])
    a = rng.randint(2, 9)
    return f"\\log_{{{b}}} {a}"


def gen_matrix(rng):
    kind = rng.random()
    if kind < 0.5:
        a, b, c, d = (rng.choice(VARS) for _ in range(4))
        env = rng.choice(["pmatrix", "bmatrix"])
        return f"\\begin{{{env}}} {a} & {b} \\\\ {c} & {d} \\end{{{env}}}"
    a, b, c = (rng.randint(1, 9) for _ in range(3))
    d, e, f = (rng.randint(1, 9) for _ in range(3))
    return (f"\\begin{{pmatrix}} {a} & {b} & {c} \\\\ "
            f"{d} & {e} & {f} \\end{{pmatrix}}")


def gen_det(rng):
    a, b, c, d = (rng.randint(1, 9) if rng.random() < 0.6 else rng.choice(VARS)
                  for _ in range(4))
    return f"\\begin{{vmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{vmatrix}}"


def gen_factorial(rng):
    if rng.random() < 0.5:
        return f"{rng.choice(VARS)}!"
    return f"{rng.randint(3, 9)}!"


def gen_binom(rng):
    if rng.random() < 0.5:
        return f"\\binom{{{rng.choice(VARS)}}}{{{rng.choice(VARS)}}}"
    n, k = rng.randint(4, 9), rng.randint(2, 3)
    return f"\\binom{{{n}}}{{{k}}}"


def gen_geometry(rng):
    kind = rng.randint(0, 3)
    if kind == 0:
        p = rng.choice(["A", "B", "C", "D"])
        n = rng.randint(2, 18) * 10
        return f"\\angle {p} = {n}^\\circ"
    if kind == 1:
        t1, t2 = rng.sample(["ABC", "DEF", "PQR"], 2)
        rel = rng.choice(["\\cong", "\\sim"])
        return f"\\triangle {t1} {rel} \\triangle {t2}"
    p1, p2 = rng.sample(["AB", "CD", "EF", "MN"], 2)
    rel = rng.choice(["\\perp", "\\parallel"])
    return f"{p1} {rel} {p2}"


def gen_set(rng):
    kind = rng.randint(0, 2)
    if kind == 0:
        v = rng.choice(["a", "b", "x"])
        s = rng.choice(["\\mathbb{R}", "\\mathbb{N}", "\\mathbb{Z}", "\\mathbb{Q}"])
        return f"{v} \\in {s}"
    if kind == 1:
        s1, s2 = rng.sample(["A", "B", "C"], 2)
        op = rng.choice(["\\cup", "\\cap"])
        return f"{s1} {op} {s2}"
    return "\\emptyset"


def gen_vector(rng):
    if rng.random() < 0.5:
        return f"\\vec{{{rng.choice(VARS)}}}"
    ab = rng.choice(["AB", "CD", "MN", "PQ"])
    return f"\\overrightarrow{{{ab}}}"


def gen_percent(rng):
    n = rng.choice([str(rng.randint(1, 99)), f"{rng.randint(1, 9)}0"])
    if rng.random() < 0.3:
        return f"{n}^\\circ"
    return f"{n}\\%"


def gen_function(rng):
    f = rng.choice(["f", "g", "h"])
    v = rng.choice(VARS)
    kind = rng.randint(0, 1)
    if kind == 0:
        return f"{f}({v}) = {v}^2"
    inner = rng.choice([f"{v} + 1", f"2{v}", f"{v}^2"])
    return f"{f}({inner})"


GENERATORS = {
    "limit": gen_limit,
    "log": gen_log,
    "matrix": gen_matrix,
    "determinant": gen_det,
    "factorial": gen_factorial,
    "binom": gen_binom,
    "geometry": gen_geometry,
    "set": gen_set,
    "vector": gen_vector,
    "percent": gen_percent,
    "function": gen_function,
}


def build(category: str, rng: random.Random) -> dict | None:
    latex = GENERATORS[category](rng)
    try:
        speech = ls.speech_of(ls.parse(latex))
    except ValueError:
        return None
    lt, st = rng.choice(TEMPLATES)
    if "{X}" in lt:
        x = rng.choice(["A", "B", "C", "D"])
        lt, st = lt.format(E=latex, X=x), st.format(S=speech, X=x)
    else:
        lt, st = lt.format(E=latex), st.format(S=speech)
    return {
        "prompt": f"{INSTRUCTION}\n原句：{lt}",
        "completion": st,
        "category": category,
        "latex": latex,
        "speech": speech,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--pool-size", type=int, default=800)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    n_cat = len(GENERATORS)
    per_test = args.test_size // n_cat
    per_pool = args.pool_size // n_cat

    test, pool, seen = [], [], set()
    for category in GENERATORS:
        collected, attempts = [], 0
        while len(collected) < per_test + per_pool and attempts < 20000:
            attempts += 1
            s = build(category, rng)
            if s is None or s["prompt"] in seen:
                continue
            seen.add(s["prompt"])
            collected.append(s)
        test.extend(collected[:per_test])
        pool.extend(collected[per_test:])

    rng.shuffle(test)
    rng.shuffle(pool)

    def dump(name, rows, fields):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in fields}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条")

    dump("ood_test.jsonl", test, ("prompt", "completion"))
    dump("ood_meta.jsonl", test, ("prompt", "completion", "category", "latex", "speech"))
    dump("ood_pool.jsonl", pool,
         ("prompt", "completion", "category", "latex", "speech"))
    from collections import Counter
    print("ood_test", dict(sorted(Counter(t["category"] for t in test).items())))


if __name__ == "__main__":
    main()
