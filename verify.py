#!/usr/bin/env python3
"""忠实性硬检查：转换输出必须守恒（conservation check）。

思路（对应"硬检查一票否决、Judge 仅举证"的评测理念）：
从参考公式的 AST 提取守恒原子——数字、变量、希腊字母、运算符词、
化学名、单位名——模型输出的朗读脚本里每个原子的口语形式都必须出现。
丢数字、丢变量、改运算符 = 转换不可信，一票否决，无需人工/LLM 复核。

输入为 evaluate.py --dump 的转储文件；检查所有组的全部输出。

Usage:
    .venv/bin/python verify.py --dump results/outputs_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import latex_speech as ls


def collect_atoms(node: ls.Node, out: set) -> None:
    """从 AST 收集必须以口语形式出现在输出中的原子（与附着规范一致）。

    原子为 frozenset 时表示多选一（如元素符号 C 可读" C"或"碳"）。
    """
    k = node.kind
    if k == "seq":
        for seg in node.val:
            if isinstance(seg, ls.Node):
                collect_atoms(seg, out)
        return
    if k == "num":
        out.add(ls.speak_number(node.val).replace(" ", ""))
    elif k == "var":
        v = node.val
        # 元素符号：可读字母或物质名（下标一并折入）
        if node.sub is not None and node.sub.kind == "num" and node.sub.sup is None:
            key = f"{v}_{node.sub.val}"
            nm = ls.CHEM_NAME.get(key)
            out.add(frozenset((nm, v, ls.speak_number(node.sub.val))) if nm
                    else frozenset((v, ls.speak_number(node.sub.val))))
        elif len(v) == 1 and v.isupper():
            nm = ls.CHEM_NAME.get(v)
            out.add(frozenset((v, nm)) if nm else v)
        else:
            out.add(v)
        return
    elif k == "greek":
        out.add(node.val)
    elif k == "cmd0":
        out.add(node.val)
    elif k == "text":
        if node.val in ls.UNITS:
            out.add(ls.UNITS[node.val])
        else:
            out.add(node.val)
    elif k == "func":
        if node.val not in ("lim", "log", "ln"):  # 结构性函数，朗读中变换为句式
            out.add(ls.FUNCS[node.val])
    elif k == "bigop":
        out.add(node.val)
    elif k == "chem":
        out.add(node.val)
    # 上下标：与 _attach_scripts 同规范；bigop/sqrt/frac 自行处理其脚本
    if k not in ("bigop", "sqrt", "frac"):
        if node.sub is not None:
            collect_atoms(node.sub, out)
        if node.sup is not None:
            sup = node.sup
            if sup.kind == "cmd0" and sup.val == "度":
                out.add("度")
            elif sup.kind == "num" and sup.val == "2":
                out.add("平方")
            elif sup.kind == "num" and sup.val == "3":
                out.add("立方")
            else:
                collect_atoms(sup, out)
                out.add("次方")
    for c in node.children:
        if isinstance(c, ls.Node):
            collect_atoms(c, out)


def norm(s: str) -> str:
    return "".join(s.split())


def check_row(src: str, output: str) -> tuple[bool, list[str]]:
    """从原句定位公式段、解析 AST 提取原子，检查输出是否全部守恒。"""
    atoms: set[str] = set()
    for _, _, latex in ls.find_formulas(src):
        try:
            node = ls.parse(latex)
        except ValueError:
            continue
        chem = ls._chem_try(node) if node.kind == "seq" else None
        if chem:
            atoms.add(chem)
        else:
            collect_atoms(node, atoms)
    if not atoms:
        return True, []
    n_out = norm(output)
    missing = []
    for a in sorted(atoms, key=str):
        alts = tuple(a) if isinstance(a, frozenset) else (a,)
        if not any(norm(x) in n_out for x in alts):
            missing.append("/".join(alts))
    return not missing, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    payload = json.load(open(args.dump, encoding="utf-8"))
    lines = [f"# 忠实性硬检查报告", "",
             f"- 数据：`{payload['test_file']}`（n={len(payload['rows'])}）",
             "", "| 组 | 守恒通过率 | 不通过数 |", "|---|---|---|"]
    summary = {}
    for g in payload["groups"]:
        ok = bad = 0
        bad_cases = []
        for row in payload["rows"]:
            src = row["prompt"].split("原句：", 1)[1]
            passed, missing = check_row(src, row["outputs"][g])
            if passed:
                ok += 1
            else:
                bad += 1
                if len(bad_cases) < 15:
                    bad_cases.append((row["category"], row["ref"],
                                      row["outputs"][g], missing))
        total = ok + bad
        summary[g] = ok / max(1, total)
        lines.append(f"| {g} | {summary[g]:.1%} | {bad} |")
        if bad_cases:
            lines += ["", f"## 组 {g} 不守恒样例", ""]
            for cat, ref, out, missing in bad_cases:
                lines.append(f"- **[{cat}]** 缺失：{'、'.join(missing)}")
                lines.append(f"  - 参考：{ref}")
                lines.append(f"  - 输出：{out}")
    print("\n".join(l for l in lines if l.startswith("| ")))
    out_path = args.out or str(Path(args.dump).with_suffix(".verify.md"))
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out_path}")


if __name__ == "__main__":
    main()
