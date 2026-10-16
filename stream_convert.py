#!/usr/bin/env python3
"""流式增量转换器：LLM 边生成边把公式转成朗读脚本，支持乐观转换与回滚。

业务背景：讲题 LLM 流式吐字、TTS 边收边播，转换层在首包延迟的热路径上。
本组件在只有前缀的条件下增量检测公式边界、立即转换（乐观），
后续 token 推翻了先前判断时回滚重来，并统计因此产生的额外时延。

两种回滚场景
------------
1. 表达式延续：前缀里公式看似完整（如 \\frac{1}{2}），下一个 token 是
   运算符（+ - = ^ _ \\ 等），说明它其实还没结束 —— 撤回已发 tentaitve
   段，拼接原文后重新扫描。
2. $ 重配对：$...$ 先按公式发出，句末标点之前又来一个 $，说明先前配对
   错了（常见于货币/价格文本），撤回重判。

Usage:
    .venv/bin/python stream_convert.py --data data/test.jsonl \
        --out results/stream_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import latex_speech as ls

# 公式之后若出现这些字符，说明公式未结束（表达式延续）
CONT_CHARS = set("+-*/=<>&|^_,;")
SCRIPT_CHARS = set("^_")


class Segment:
    __slots__ = ("kind", "raw", "speech", "tentative", "dollar")

    def __init__(self, kind: str, raw: str, speech: str = "",
                 tentative: bool = False, dollar: bool = False):
        self.kind = kind          # text | formula
        self.raw = raw            # 原文段（回滚时拼回缓冲区）
        self.speech = speech
        self.tentative = tentative
        self.dollar = dollar


class StreamConverter:
    def __init__(self):
        self.buf = ""
        self.final: list[Segment] = []
        self.tentative: Segment | None = None
        self.rollbacks = {"continuation": 0, "dollar": 0}
        self.convert_ms: list[float] = []
        self.first_formula_wait: list[int] = []  # 首公式发出时已喂入字符数
        self._fed = 0
        self._first_formula_done = False

    # ---- 对外接口 ----

    def feed(self, chunk: str) -> list[tuple]:
        """喂入一段流式文本，返回事件列表：
        ("emit", kind, speech, final) / ("rollback", raw)"""
        self.buf += chunk
        self._fed += len(chunk)
        events = []
        while True:
            ev = self._drain_once()
            if ev is None:
                break
            events.append(ev)
        return events

    def flush(self) -> list[tuple]:
        """流结束：吐出剩余缓冲区。"""
        events = []
        if self.tentative is not None:
            self._finalize_tentative()
        if self.buf:
            seg = Segment("text", self.buf, self.buf)
            self.final.append(seg)
            events.append(("emit", "text", seg.speech, True))
            self.buf = ""
        return events

    @property
    def text(self) -> str:
        """当前已确定的完整朗读文本。"""
        return "".join(s.speech for s in self.final)

    # ---- 内部 ----

    def _finalize_tentative(self):
        if self.tentative is not None:
            self.tentative.tentative = False
            self.tentative = None

    def _drain_once(self) -> tuple | None:
        if not self.buf:
            return None
        # 1. tentative 段：空格不算数，看去空白后的下一个字符再决策
        if self.tentative is not None:
            t = self.tentative
            head = self.buf.lstrip()
            if not head:
                return None  # 只有空白，等下一个 token
            if t.dollar and head.startswith("$"):
                self._rollback("dollar")
                return ("rollback", t.raw)
            if head[0] in CONT_CHARS or head[0] in SCRIPT_CHARS \
                    or head[0] == "\\" or (head[0].isascii() and head[0].isalnum()):
                # 只有重扫后公式确实变长才回滚，否则等更多 token（延续可能没喂完）
                candidate = t.raw + self.buf
                spans = ls.find_formulas(candidate)
                if spans and spans[0][0] == 0 and spans[0][1] > len(t.raw):
                    self._rollback("continuation")
                    return ("rollback", t.raw)
                return None
            self._finalize_tentative()
            return None
        # 2. 缓冲区头部是中文/全角标点 -> 直接作为文本发出
        c = self.buf[0]
        if not c.isascii() or c in "，。；：、！？（）“”":
            seg = Segment("text", c, c)
            self.final.append(seg)
            self.buf = self.buf[1:]
            return ("emit", "text", c, True)
        # 3. ascii 起点：尝试在头部识别完整公式
        spans = ls.find_formulas(self.buf)
        if spans and spans[0][0] == 0:
            _, end, latex = spans[0]
            raw = self.buf[:end]
            try:
                node = ls.parse(latex)
                t0 = time.perf_counter()
                speech = ls.speech_of(node)
                self.convert_ms.append((time.perf_counter() - t0) * 1000)
            except ValueError:
                speech = ""
            if speech:
                seg = Segment("formula", raw, speech, tentative=True,
                              dollar=raw.startswith("$"))
                self.final.append(seg)
                self.tentative = seg
                self.buf = self.buf[end:]
                if not self._first_formula_done:
                    self._first_formula_done = True
                    self.first_formula_wait.append(self._fed)
                return ("emit", "formula", speech, False)
        # 4. 识别不出来：若首字符不是公式起点，按文本吐出一个字符；
        #    否则等待更多字符（公式边界未到）
        if not (c == "\\" or c == "$" or c.isalnum()):
            seg = Segment("text", c, c)
            self.final.append(seg)
            self.buf = self.buf[1:]
            return ("emit", "text", c, True)
        return None

    def _rollback(self, kind: str):
        t = self.tentative
        self.tentative = None
        self.final.pop()
        self.buf = t.raw + self.buf
        self.rollbacks[kind] += 1


# ---------------- 评测 ----------------

def normalize(s: str) -> str:
    return "".join(s.split())


def run_stream_eval(rows: list[dict], metas: list[dict]) -> dict:
    conv_stats = {"ok": 0, "bad": 0}
    lat_all, wait_all = [], []
    rb_total = {"continuation": 0, "dollar": 0}
    bad_cases = []
    chunk_hist = []

    for row, meta in zip(rows, metas):
        src = row["prompt"].split("原句：", 1)[1]
        ref = meta["completion"]
        conv = StreamConverter()
        # 逐字符喂入（对边界检测最严苛的模拟）
        for ch in src:
            conv.feed(ch)
        conv.flush()
        rb_total["continuation"] += conv.rollbacks["continuation"]
        rb_total["dollar"] += conv.rollbacks["dollar"]
        lat_all.extend(conv.convert_ms)
        wait_all.extend(conv.first_formula_wait)
        out = conv.text
        if normalize(out) == normalize(ref):
            conv_stats["ok"] += 1
        else:
            conv_stats["bad"] += 1
            if len(bad_cases) < 20:
                bad_cases.append((src, ref, out,
                                  conv.rollbacks["continuation"]
                                  + conv.rollbacks["dollar"]))

    def pct(xs, p):
        return sorted(xs)[min(len(xs) - 1, int(len(xs) * p))] if xs else 0.0

    n_rb = rb_total["continuation"] + rb_total["dollar"]
    return {
        "n": len(rows),
        "exact": conv_stats["ok"] / max(1, len(rows)),
        "formulas": len(lat_all),
        "rollbacks": rb_total,
        "rb_per_100_formula": n_rb / max(1, len(lat_all)) * 100,
        "convert_ms": {
            "avg": statistics.mean(lat_all) if lat_all else 0,
            "p50": pct(lat_all, 0.5),
            "p95": pct(lat_all, 0.95),
            "max": max(lat_all) if lat_all else 0,
        },
        "first_formula_wait_avg": statistics.mean(wait_all) if wait_all else 0,
        "bad_cases": bad_cases,
    }


def write_report(path: str, stats: dict):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    c = stats["convert_ms"]
    lines = [
        "# 流式增量转换评测报告",
        "",
        f"- 样本：{stats['n']} 条冻结测试集原句（逐字符喂入模拟流式）",
        f"- 转换正确率（与整句参考完全一致）：**{stats['exact']:.1%}**",
        f"- 公式段总数：{stats['formulas']}，回滚总数："
        f"{stats['rollbacks']['continuation']}（延续）+ {stats['rollbacks']['dollar']}（$ 重配对）"
        f" = {stats['rollbacks']['continuation'] + stats['rollbacks']['dollar']}"
        f"（每百公式 {stats['rb_per_100_formula']:.1f} 次）",
        f"- 单次转换耗时：avg {c['avg']:.3f}ms / p50 {c['p50']:.3f}ms / "
        f"p95 {c['p95']:.3f}ms / max {c['max']:.3f}ms",
        f"- 首公式平均等待（已喂入字符数）：{stats['first_formula_wait_avg']:.0f}",
        "",
        "> 口径说明：转换耗时只含解析+朗读生成（规则引擎）。",
        "> 首包影响 = 转换层新增时延，可与 LLM 流式生成/TTS 合成 overlap。",
        "",
        "## 不一致样例（前 20 条）",
        "",
    ]
    for src, ref, out_text, rb in stats["bad_cases"]:
        lines.append(f"- 原句：`{src}`")
        lines.append(f"  - 参考：{ref}")
        lines.append(f"  - 流式输出：{out_text}（回滚 {rb} 次）")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/stream_report.md")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{args.data}/test.jsonl", encoding="utf-8")]
    metas = [json.loads(l) for l in open(f"{args.data}/test_meta.jsonl", encoding="utf-8")]
    assert [r["prompt"] for r in rows] == [m["prompt"] for m in metas]
    if args.limit:
        rows, metas = rows[: args.limit], metas[: args.limit]

    stats = run_stream_eval(rows, metas)
    print(f"流式转换正确率: {stats['exact']:.1%} ({stats['n']} 条)")
    print(f"回滚: {stats['rollbacks']}, 转换耗时 avg {stats['convert_ms']['avg']:.3f}ms "
          f"p95 {stats['convert_ms']['p95']:.3f}ms")
    write_report(args.out, stats)


if __name__ == "__main__":
    main()
