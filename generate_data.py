#!/usr/bin/env python3
"""Rule-based generator for the LaTeX-to-speech rewrite dataset.

Each sample teaches the model to rewrite LaTeX formulas embedded in K12
tutoring sentences into Mandarin TTS-friendly speech, e.g.
``\\frac{1}{2}`` -> "二分之一", ``\\sqrt{x}`` -> "根号 x".

Output files (default: data/):
    train.jsonl      ~2000 samples for LoRA training
    valid.jsonl      ~200  samples for validation
    test.jsonl       200   samples, FROZEN once generated
    test_meta.jsonl  per-test category + expected spoken formula (for eval)
    spot_check.txt   100 random samples for human review

Usage:
    .venv/bin/python generate_data.py --out data --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

INSTRUCTION = "将讲解中的公式改写为适合语音朗读的中文口语，只输出改写后的整句："

CN_DIGIT = "零一二三四五六七八九"

GREEK = {
    "\\alpha": "阿尔法",
    "\\beta": "贝塔",
    "\\gamma": "伽马",
    "\\theta": "西塔",
    "\\lambda": "兰姆达",
    "\\mu": "谬",
    "\\pi": "派",
    "\\rho": "柔",
    "\\Delta": "德尔塔",
    "\\Sigma": "西格玛",
    "\\Omega": "欧米伽",
}

CHEM_NAME = {
    "H_2": "氢气",
    "O_2": "氧气",
    "N_2": "氮气",
    "C": "碳",
    "CO": "一氧化碳",
    "CO_2": "二氧化碳",
    "SO_2": "二氧化硫",
    "NO_2": "二氧化氮",
    "H_2O": "水",
    "H_2SO_4": "硫酸",
    "HCl": "盐酸",
    "NaOH": "氢氧化钠",
    "NaCl": "氯化钠",
    "CaO": "氧化钙",
    "CaCO_3": "碳酸钙",
    "Fe_2O_3": "三氧化二铁",
    "CH_4": "甲烷",
    "NH_3": "氨气",
    "C_2H_5OH": "乙醇",
}

CHEM_EQUATIONS = [
    ("2H_2 + O_2 = 2H_2O", "二 氢气 加 氧气 等于 二 水"),
    ("2CO + O_2 = 2CO_2", "二 一氧化碳 加 氧气 等于 二 二氧化碳"),
    ("C + O_2 = CO_2", "碳 加 氧气 等于 二氧化碳"),
    ("2H_2O = 2H_2 + O_2", "二 水 等于 二 氢气 加 氧气"),
]

VARS = ["x", "y", "a", "b", "m", "n", "c", "d"]


def speak_int(n: int) -> str:
    """逐位朗读整数：2026 -> 二零二六。"""
    if n < 0:
        return "负" + speak_int(-n)
    return "".join(CN_DIGIT[int(d)] for d in str(n))


def speak_number(s: str | int) -> str:
    """朗读数字文本（支持小数）：3.14 -> 三点一四。"""
    s = str(s)
    if "." in s:
        ip, fp = s.split(".", 1)
        head = speak_int(int(ip)) if ip else "零"
        return head + "点" + "".join(CN_DIGIT[int(d)] for d in fp)
    return speak_int(int(s))


def speak_token(tok: str) -> str:
    """朗读公式中的原子 token：变量原样，数字逐位。"""
    return speak_number(tok) if tok.isdigit() else tok


def speak_frac(num: str, den: str) -> str:
    """分数读法：1/2 -> 二分之一，5/6 -> 六分之五。"""
    return f"{speak_number(den)}分之{speak_number(num)}"


def pick_frac(rng: random.Random) -> tuple[str, str, str]:
    """随机生成已约分的真分数，返回 (a, b, speech)。"""
    b = rng.randint(2, 9)
    a = rng.randint(1, b - 1)
    g = math.gcd(a, b)
    a, b = a // g, b // g
    return str(a), str(b), speak_frac(str(a), str(b))


# ---------------------------------------------------------------------------
# 8 类公式模板：每个生成器返回 (latex, speech)
# ---------------------------------------------------------------------------

def gen_fraction(rng: random.Random) -> tuple[str, str]:
    kind = rng.randint(0, 2)
    if kind == 0:
        a, b, sp = pick_frac(rng)
        return f"\\frac{{{a}}}{{{b}}}", sp
    if kind == 1:
        whole = rng.randint(1, 3)
        a, b, sp = pick_frac(rng)
        latex = f"{whole}\\frac{{{a}}}{{{b}}}"
        speech = f"{speak_int(whole)}又{sp}"
        return latex, speech
    a, b, sp1 = pick_frac(rng)
    c, d, sp2 = pick_frac(rng)
    op = rng.choice(["+", "-"])
    word = "加" if op == "+" else "减"
    latex = f"\\frac{{{a}}}{{{b}}} {op} \\frac{{{c}}}{{{d}}}"
    speech = f"{sp1} {word} {sp2}"
    return latex, speech


def gen_radical(rng: random.Random) -> tuple[str, str]:
    kind = rng.randint(0, 3)
    if kind == 0:
        x = rng.choice(VARS + [str(rng.randint(2, 9))])
        return f"\\sqrt{{{x}}}", f"根号 {speak_token(x)}"
    if kind == 1:
        x = rng.choice(VARS)
        return f"\\sqrt{{{x}^2 + 1}}", f"根号下 {x} 平方 加 一"
    n = rng.choice(["3", "4", "n"])
    x = rng.choice(VARS + ["8", "27"])
    n_sp = "三" if n == "3" else ("四" if n == "4" else "n")
    return f"\\sqrt[{n}]{{{x}}}", f"{n_sp} 次根号 {speak_token(x)}"


def gen_script(rng: random.Random) -> tuple[str, str]:
    v = rng.choice(VARS)
    kind = rng.randint(0, 3)
    if kind == 0:
        return f"{v}^2", f"{v} 平方"
    if kind == 1:
        return f"{v}^{{n+1}}", f"{v} 的 n 加 一 次方"
    if kind == 2:
        return f"{v}_0", f"{v} 零"
    return f"{v}_{{n-1}}", f"{v} 下标 n 减 一"


def gen_greek(rng: random.Random) -> tuple[str, str]:
    g, name = rng.choice(list(GREEK.items()))
    kind = rng.randint(0, 2)
    if kind == 0:
        v = rng.choice(["A", "B", "C", "O"])
        return f"{g} = {v}", f"{name} 等于 {v}"
    if kind == 1:
        n = rng.randint(1, 9) * 10
        return f"{g} = {n}^\\circ", f"{name} 等于 {speak_int(n)} 度"
    g2, name2 = rng.choice([(k, v) for k, v in GREEK.items() if k != g])
    return f"{g} + {g2}", f"{name} 加 {name2}"


def gen_operator(rng: random.Random) -> tuple[str, str]:
    kind = rng.randint(0, 4)
    if kind == 0:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        if rng.random() < 0.3:
            return f"x = \\pm {a}", f"x 等于 正负 {speak_int(a)}"
        op, word = rng.choice([("\\times", "乘"), ("\\div", "除以")])
        if op == "\\div":
            q = rng.randint(2, 9)
            a = b * q
            return f"{a} \\div {b} = {q}", f"{speak_int(a)} 除以 {speak_int(b)} 等于 {speak_int(q)}"
        return f"{a} \\times {b} = {a*b}", f"{speak_int(a)} 乘 {speak_int(b)} 等于 {speak_int(a*b)}"
    if kind == 1:
        v = rng.choice(VARS)
        n = rng.randint(0, 20)
        op, word = rng.choice([
            ("\\leq", "小于等于"), ("\\geq", "大于等于"),
            ("\\neq", "不等于"), ("\\approx", "约等于"),
        ])
        return f"{v} {op} {n}", f"{v} {word} {speak_int(n)}"
    if kind == 2:
        return "x \\to \\infty", "x 趋向于 无穷大"
    if kind == 3:
        i = rng.choice(["i", "k"])
        n = rng.choice(["n", "10"])
        return f"\\sum_{{{i}=1}}^{{{n}}} {i}", f"对 {i} 从 一 到 {n} 求和"
    a = rng.randint(0, 2)
    b = a + rng.randint(1, 3)
    v = rng.choice(VARS)
    return f"\\int_{{{a}}}^{{{b}}} {v} \\, d{v}", f"{v} 对 {v} 从 {speak_int(a)} 到 {speak_int(b)} 的积分"


PHYS_FORMULAS = [
    ("v = \\frac{s}{t}", "v 等于 s 除以 t"),
    ("\\rho = \\frac{m}{V}", "柔 等于 m 除以 V"),
    ("F = ma", "F 等于 m a"),
    ("G = mg", "G 等于 m g"),
    ("I = \\frac{U}{R}", "I 等于 U 除以 R"),
    ("W = Fs", "W 等于 F s"),
    ("P = \\frac{W}{t}", "P 等于 W 除以 t"),
    ("Q = cm\\Delta t", "Q 等于 c m 德尔塔 t"),
]

VAR_UNITS = {
    "v": [("m/s", "米每秒"), ("km/h", "千米每小时")],
    "a": [("m/s^2", "米每二次方秒")],
    "m": [("kg", "千克"), ("g", "克")],
    "s": [("m", "米"), ("km", "千米"), ("cm", "厘米")],
    "t": [("s", "秒"), ("h", "小时")],
    "F": [("N", "牛")],
}


def gen_physics(rng: random.Random) -> tuple[str, str]:
    if rng.random() < 0.5:
        return rng.choice(PHYS_FORMULAS)
    var = rng.choice(list(VAR_UNITS))
    u, u_sp = rng.choice(VAR_UNITS[var])
    val = rng.choice([str(rng.randint(1, 20)), f"{rng.randint(1, 9)}.{rng.randint(1, 9)}"])
    latex = f"{var} = {val}\\,\\text{{{u}}}"
    speech = f"{var} 等于 {speak_number(val)} {u_sp}"
    return latex, speech


def gen_chemistry(rng: random.Random) -> tuple[str, str]:
    if rng.random() < 0.5:
        f = rng.choice(list(CHEM_NAME))
        return f, CHEM_NAME[f]
    return rng.choice(CHEM_EQUATIONS)


def gen_mixed(rng: random.Random) -> tuple[str, str]:
    kind = rng.randint(0, 3)
    if kind == 0:
        n = rng.choice(["2", "3"])
        inner = f"\\sqrt{{{n}}}"
        return "\\frac{" + inner + "}{2}", f"二分之根号 {speak_number(n)}"
    if kind == 1:
        v = rng.choice(VARS)
        inner = f"\\sqrt{{{v}^2 + 1}}"
        return "\\frac{" + inner + "}{2}", f"二分之根号下 {v} 平方 加 一"
    if kind == 2:
        a, b = rng.randint(1, 9), rng.randint(2, 9)
        v = rng.choice(VARS)
        return f"\\frac{{{a}}}{{{b}}} {v}^2", f"{speak_frac(str(a), str(b))} {v} 平方"
    return "\\sqrt{\\frac{1}{4}}", "根号 四分之一"


GENERATORS = {
    "fraction": gen_fraction,
    "radical": gen_radical,
    "script": gen_script,
    "greek": gen_greek,
    "operator": gen_operator,
    "physics": gen_physics,
    "chemistry": gen_chemistry,
    "mixed": gen_mixed,
}

# ---------------------------------------------------------------------------
# 讲题句式模板：{E} 为公式块，{S} 为朗读形式，{X} 为选项
# ---------------------------------------------------------------------------

TEMPLATES = [
    ("因为 {E}，所以选 {X}。", "因为 {S}，所以选 {X}。"),
    ("原式可化为 {E}，所以选 {X}。", "原式可化为 {S}，所以选 {X}。"),
    ("由 {E} 可知，答案选 {X}。", "由 {S} 可知，答案选 {X}。"),
    ("代入得 {E}。", "代入得 {S}。"),
    ("化简可得 {E}。", "化简可得 {S}。"),
    ("由此可知 {E}。", "由此可知 {S}。"),
    ("计算得到 {E}。", "计算得到 {S}。"),
    ("解得 {E}。", "解得 {S}。"),
    ("即 {E}。", "即 {S}。"),
    ("于是 {E}。", "于是 {S}。"),
    ("根据公式 {E} 可求出结果。", "根据公式 {S} 可求出结果。"),
    ("将 {E} 代入原式，即可求解。", "将 {S} 代入原式，即可求解。"),
    ("这一步要注意 {E}。", "这一步要注意 {S}。"),
    ("接下来计算 {E}。", "接下来计算 {S}。"),
    ("最终得到 {E}。", "最终得到 {S}。"),
    ("根据题意得 {E}。", "根据题意得 {S}。"),
    ("联立可得 {E}。", "联立可得 {S}。"),
    ("整理得 {E}。", "整理得 {S}。"),
    ("两边化简后得 {E}。", "两边化简后得 {S}。"),
    ("注意到 {E}。", "注意到 {S}。"),
]


def build_sample(category: str, rng: random.Random) -> dict:
    latex, speech = GENERATORS[category](rng)
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
        "speech": speech,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--valid", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--candidates", type=int, default=900)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_cat = len(GENERATORS)
    per_test = args.test // n_cat
    per_valid = args.valid // n_cat

    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []
    seen: set[str] = set()
    for category in GENERATORS:
        pool: list[dict] = []
        local_seen: set[str] = set()
        attempts = 0
        while len(pool) < args.candidates and attempts < args.candidates * 30:
            attempts += 1
            s = build_sample(category, rng)
            if s["prompt"] in local_seen:
                continue
            local_seen.add(s["prompt"])
            pool.append(s)
        need = per_test + per_valid + 50
        assert len(pool) >= need, f"{category}: 候选不足 ({len(pool)} < {need})"
        rng.shuffle(pool)
        test.extend(pool[:per_test])
        valid.extend(pool[per_test : per_test + per_valid])
        train.extend(pool[per_test + per_valid :])

    rng.shuffle(train)
    train = train[: args.train]
    rng.shuffle(valid)
    rng.shuffle(test)

    # 跨集合去重：test/valid 优先，train 不得与之重复
    seen = {s["prompt"] for s in test} | {s["prompt"] for s in valid}
    train = [s for s in train if not (s["prompt"] in seen or seen.add(s["prompt"]))]

    def dump(name: str, rows: list[dict], fields: tuple[str, ...]) -> None:
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in fields}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条")

    dump("train.jsonl", train, ("prompt", "completion"))
    dump("valid.jsonl", valid, ("prompt", "completion"))
    dump("test.jsonl", test, ("prompt", "completion"))
    dump("test_meta.jsonl", test, ("prompt", "completion", "category", "speech"))

    spot = rng.sample(train + valid + test, min(100, len(train) + len(valid) + len(test)))
    with open(out / "spot_check.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(spot, 1):
            f.write(f"--- [{i}] ({s['category']}) ---\n{s['prompt']}\n=> {s['completion']}\n\n")
    print("spot_check.txt: 100 条")

    from collections import Counter
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        dist = Counter(s["category"] for s in rows)
        print(name, dict(sorted(dist.items())))


if __name__ == "__main__":
    main()
