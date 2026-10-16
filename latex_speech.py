#!/usr/bin/env python3
"""LaTeX -> 中文朗读脚本：解析器、朗读规范与 SSML 桥接（共享核心）。

本模块是"朗读规范唯一事实源" v0：规则生成器（generate_data.py）、
流式转换器（stream_convert.py）、端云路由器（router.py）、
忠实性检查（verify.py）与教师蒸馏（distill.py）全部以这里的约定为准。

朗读规范（conventions v0）
-------------------------
- 数字逐位朗读：2026 -> 二零二六；3.14 -> 三点一四
- 分数：\\frac{a}{b} -> b分之a（连写）；带分数 1\\frac{1}{2} -> 一又二分之一
- 根式：单元数 \\sqrt{x} -> 根号 x；复合 \\sqrt{x+1} -> 根号下 x 加 一；
  \\sqrt[3]{x} -> 三 次根号 x；\\sqrt{\\frac{1}{4}} -> 根号 四分之一
- 上标：^2 平方、^3 立方、^{n} 的 n 次方、^\\circ 度
- 下标：单字符直读 a_n -> a n；复合 a_{n-1} -> a 下标 n 减 一
- 多字母变量串逐字母读（ABC -> A B C）；化学式查通用名（CO_2 -> 二氧化碳）
- 括号：一般组读"括号 ... 闭括号"；单字母函数应用 f(x) -> f x
- 百分号后置：50\\% -> 百分之五零；阶乘：n! -> n 的阶乘
- 矩阵/行列式按行读：矩阵，第 一 行 a、b，第 二 行 c、d
- \\lim_{x \\to 0} body -> 当 x 趋向于 零 时，body 的极限
- \\log_{2}8 -> 以 二 为底 八 的对数；\\ln x -> x 的自然对数
- \\binom{n}{k} -> n 取 k 的组合数；\\vec{a} -> 向量 a
- 几何：\\angle A -> 角 A；\\triangle ABC -> 三角形 ABC；\\perp 垂直于
- 集合：\\in 属于；\\cup 并；\\cap 交；\\emptyset 空集；\\mathbb{R} 实数集
- 单位（\\text 内含已识别单位）：m/s 米每秒、km/h 千米每小时、m/s^2 米每二次方秒
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CN_DIGIT = "零一二三四五六七八九"
CN_ORD = "一二三四五六七八九"


def speak_int(n: int) -> str:
    if n < 0:
        return "负" + speak_int(-n)
    return "".join(CN_DIGIT[int(d)] for d in str(n))


def speak_number(s: str) -> str:
    s = str(s)
    if "." in s:
        ip, fp = s.split(".", 1)
        head = speak_int(int(ip)) if ip else "零"
        return head + "点" + "".join(CN_DIGIT[int(d)] for d in fp)
    return speak_int(int(s))


GREEK = {
    "alpha": "阿尔法", "beta": "贝塔", "gamma": "伽马", "theta": "西塔",
    "lambda": "兰姆达", "mu": "谬", "pi": "派", "rho": "柔", "phi": "fai",
    "omega": "欧米伽", "sigma": "西格玛", "Delta": "德尔塔", "Sigma": "西格玛",
    "Omega": "欧米伽", "epsilon": "艾普西龙", "psi": "普赛",
}

FUNCS = {"sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot",
         "log": "log", "ln": "ln", "lim": "lim", "max": "max", "min": "min"}

CMD0 = {
    "infty": "无穷大", "pm": "正负", "mp": "负正", "times": "乘", "div": "除以",
    "cdot": "乘", "leq": "小于等于", "le": "小于等于", "geq": "大于等于", "ge": "大于等于",
    "neq": "不等于", "approx": "约等于", "to": "趋向于", "rightarrow": "趋向于",
    "Rightarrow": "推出", "in": "属于", "notin": "不属于", "subseteq": "包含于",
    "supseteq": "包含", "cup": "并", "cap": "交", "emptyset": "空集",
    "forall": "任意", "exists": "存在", "cong": "全等于", "sim": "相似于",
    "perp": "垂直于", "parallel": "平行于", "because": "因为", "therefore": "所以",
    "ldots": "省略号", "cdots": "省略号", "circ": "度", "angle": "角", "triangle": "三角形",
}

BIGOP = {"sum": "求和", "prod": "连乘", "int": "积分"}

UNITS = {
    "m/s": "米每秒", "km/h": "千米每小时", "m/s^2": "米每二次方秒",
    "kg": "千克", "g": "克", "cm": "厘米", "mm": "毫米", "m": "米",
    "km": "千米", "s": "秒", "h": "小时", "N": "牛", "Pa": "帕",
    "J": "焦", "W": "瓦", "A": "安", "V": "伏", "Omega": "欧姆", "L": "升",
    "mL": "毫升",
}

CHEM_NAME = {
    "H_2": "氢气", "O_2": "氧气", "N_2": "氮气", "C": "碳",
    "CO": "一氧化碳", "CO_2": "二氧化碳", "SO_2": "二氧化硫", "NO_2": "二氧化氮",
    "H_2O": "水", "H_2SO_4": "硫酸", "HCl": "盐酸", "NaOH": "氢氧化钠",
    "NaCl": "氯化钠", "CaO": "氧化钙", "CaCO_3": "碳酸钙", "Fe_2O_3": "三氧化二铁",
    "CH_4": "甲烷", "NH_3": "氨气", "C_2H_5OH": "乙醇",
}

BINOP = {"+": "加", "-": "减", "*": "乘", "=": "等于", "<": "小于", ">": "大于"}
SPACING_OPS = {"\\,", "\\;", "\\:", "\\!", "\\ "}

REL_CMD = {
    "times", "div", "cdot", "leq", "le", "geq", "ge", "neq", "approx",
    "pm", "mp", "to", "rightarrow", "Rightarrow", "in", "notin",
    "subseteq", "supseteq", "cup", "cap", "cong", "sim", "perp",
    "parallel", "because", "therefore",
}
SPACING_CMDS = {"quad", "qquad", "enspace"}
MATHBB = {"R": "实数集", "N": "自然数集", "Z": "整数集", "Q": "有理数集"}
MATRIX_ENVS = {"pmatrix": "矩阵", "bmatrix": "矩阵", "vmatrix": "行列式"}


@dataclass
class Node:
    kind: str
    val: object = None
    children: list["Node"] = field(default_factory=list)
    sup: "Node | None" = None
    sub: "Node | None" = None


# ---------------- 词法 ----------------

@dataclass
class Tok:
    kind: str  # cmd lbrace rbrace lbrack rbrack caret under num letter op punct sep
    val: str


def tokenize(s: str) -> list[Tok]:
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", s[i:])
            if m:
                name = m.group(1)
                if name in SPACING_CMDS:
                    i += len(m.group(0))
                    continue
                toks.append(Tok("cmd", name))
                i += len(m.group(0))
                continue
            if s[i:i + 2] == "\\\\":
                toks.append(Tok("sep", "\\\\"))
                i += 2
                continue
            if i + 1 < n and s[i + 1] in ",;:!% ":
                toks.append(Tok("op", "\\" + s[i + 1]))
                i += 2
                continue
            raise ValueError(f"未知转义: {s[i:i+4]!r}")
        if c == "{":
            toks.append(Tok("lbrace", c))
        elif c == "}":
            toks.append(Tok("rbrace", c))
        elif c == "[":
            toks.append(Tok("lbrack", c))
        elif c == "]":
            toks.append(Tok("rbrack", c))
        elif c in "^_":
            toks.append(Tok("caret" if c == "^" else "under", c))
        elif c == "&":
            toks.append(Tok("sep", "&"))
        elif c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            m = re.match(r"\d+(?:\.\d+)?", s[i:])
            toks.append(Tok("num", m.group(0)))
            i += len(m.group(0))
            continue
        elif c.isalpha():
            toks.append(Tok("letter", c))
        elif c in "+-*/=<>!(),;:|'":
            toks.append(Tok("op", c))
        elif c in "，。；：、！？（）":
            toks.append(Tok("punct", c))
        elif c in " \t\r\n":
            pass
        elif c == "%":
            toks.append(Tok("op", "\\%"))
        elif c == "$":
            toks.append(Tok("punct", "$"))
        else:
            raise ValueError(f"未支持字符: {c!r}")
        i += 1
    return toks


class Parser:
    """seq 用 segs 表示：Node 与运算符词字符串交替，恒以 Node 开头。"""

    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.i = 0

    def peek(self) -> Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> Tok:
        t = self.peek()
        if t is None:
            raise ValueError("意外结尾")
        self.i += 1
        return t

    def _stop(self, t: Tok, stops: set[str]) -> bool:
        if t.kind in ("rbrace", "rbrack", "sep", "lbrack"):
            return t.kind != "lbrack" or "[]" in stops
        if t.kind == "cmd" and t.val == "end":
            return True
        if t.kind == "punct":
            return t.val in stops or t.val in "。，；、！？）$"
        if t.kind == "op":
            return t.val in stops
        return False

    def parse_expr(self, stops: set[str] | None = None) -> Node:
        stops = stops or set()
        segs: list = []
        while True:
            t = self.peek()
            if t is None or self._stop(t, stops):
                break
            if t.kind == "op" and t.val in SPACING_OPS:
                self.next()
                continue
            if t.kind == "op" and t.val == "\\%" and any(isinstance(x, Node) for x in segs):
                self.next()
                for j in range(len(segs) - 1, -1, -1):
                    if isinstance(segs[j], Node):
                        segs[j] = Node("percent", children=[segs[j]])
                        break
                continue
            if t.kind == "op" and t.val in BINOP and any(isinstance(x, Node) for x in segs):
                segs.append(BINOP[self.next().val])
                continue
            if t.kind == "cmd" and t.val in REL_CMD and any(isinstance(x, Node) for x in segs):
                segs.append(CMD0[self.next().val])
                continue
            segs.append(self.parse_item())
        nodes = [x for x in segs if isinstance(x, Node)]
        if not nodes:
            raise ValueError("空表达式")
        if len(nodes) == 1 and len(segs) == 1:
            return nodes[0]
        return Node("seq", val=segs)

    def parse_item(self) -> Node:
        t = self.peek()
        if t is None:
            raise ValueError("意外结尾")
        if t.kind == "op" and t.val == "-":
            self.next()
            node = Node("neg", children=[self.parse_item()])
        elif t.kind == "num":
            self.next()
            node = Node("num", t.val)
        elif t.kind == "letter":
            self.next()
            node = Node("var", t.val)
            # 单字母函数应用 f(x) -> f x（省括号）
            p = self.peek()
            if p is not None and p.kind == "op" and p.val == "(":
                self.next()
                inner = self.parse_expr({")"})
                self._expect_op(")")
                node = Node("fap", children=[node, inner])
        elif t.kind == "cmd":
            node = self.parse_cmd()
        elif t.kind == "lbrace":
            self.next()
            node = Node("group", children=[self.parse_expr({"}"})])
            self._expect("rbrace")
        elif t.kind == "op" and t.val == "(":
            self.next()
            inner = self.parse_expr({")"})
            self._expect_op(")")
            node = Node("group", children=[inner], val="paren")
        elif t.kind == "op" and t.val == "|":
            self.next()
            inner = self.parse_expr({"|"})
            self._expect_op("|")
            node = Node("group", children=[inner], val="abs")
        else:
            raise ValueError(f"无法解析 token: {t.kind} {t.val!r}")
        while True:
            t = self.peek()
            if t is None or t.kind not in ("caret", "under"):
                break
            self.next()
            arg = self.parse_script_arg()
            if t.kind == "caret":
                node.sup = arg
            else:
                node.sub = arg
        if self.peek() is not None and self.peek().kind == "op" and self.peek().val == "!":
            self.next()
            node = Node("postfix", children=[node])
        return node

    def _expect(self, kind: str) -> None:
        t = self.peek()
        if t is None or t.kind != kind:
            raise ValueError(f"缺 {kind}")
        self.next()

    def _expect_op(self, val: str) -> None:
        t = self.peek()
        if t is None or t.kind != "op" or t.val != val:
            raise ValueError(f"缺 {val}")
        self.next()

    def parse_script_arg(self) -> Node:
        t = self.peek()
        if t is None:
            raise ValueError("缺上/下标参数")
        if t.kind == "lbrace":
            self.next()
            n = self.parse_expr({"}"})
            self._expect("rbrace")
            return n
        return self.parse_item()

    def parse_raw_group(self) -> str:
        self._expect("lbrace")
        depth, out = 1, []
        while True:
            t = self.next()
            if t.kind == "lbrace":
                depth += 1
            elif t.kind == "rbrace":
                depth -= 1
                if depth == 0:
                    return "".join(out)
            out.append(t.val if t.kind != "sep" else ("&" if t.val == "&" else "\\\\"))

    def parse_cmd(self) -> Node:
        name = self.next().val
        if name in GREEK:
            return Node("greek", GREEK[name])
        if name in FUNCS:
            return Node("func", name)
        if name in BIGOP:
            return Node("bigop", BIGOP[name])
        if name in ("frac", "dfrac", "tfrac"):
            return Node("frac", children=[self.parse_group(), self.parse_group()])
        if name == "binom":
            return Node("binom", children=[self.parse_group(), self.parse_group()])
        if name == "sqrt":
            node = Node("sqrt", children=[None])
            if self.peek() is not None and self.peek().kind == "lbrack":
                self.next()
                node.sub = self.parse_expr({"]"})
                self._expect("rbrack")
            node.children[0] = self.parse_group()
            return node
        if name in ("text", "mathrm", "textbf", "textit"):
            return Node("text", self.parse_raw_group())
        if name == "vec":
            return Node("vec", children=[self.parse_group()])
        if name == "overrightarrow":
            return Node("vec", children=[self.parse_group()], val="arrow")
        if name in ("left", "right", "big", "Big", "bigg", "Bigg"):
            return Node("wrap", children=[self.parse_expr({")", "]", "|"})])
        if name == "begin":
            return self.parse_env()
        if name == "mathbb":
            g = self.parse_group()
            letter = g.val if g.kind == "var" else None
            if letter not in MATHBB:
                raise ValueError("不支持的 \\mathbb")
            return Node("cmd0", MATHBB[letter])
        if name in CMD0:
            return Node("cmd0", CMD0[name])
        raise ValueError(f"未知命令: \\{name}")

    def parse_group(self) -> Node:
        t = self.peek()
        if t is None:
            raise ValueError("缺组")
        if t.kind == "lbrace":
            self.next()
            n = self.parse_expr({"}"})
            self._expect("rbrace")
            return n
        return self.parse_item()

    def parse_env(self) -> Node:
        env_name = self.parse_raw_group()
        label = MATRIX_ENVS.get(env_name)
        if label is None:
            raise ValueError(f"不支持的环境: {env_name}")
        rows, cur = [], []
        while True:
            t = self.peek()
            if t is None:
                raise ValueError("环境未闭合")
            if t.kind == "cmd" and t.val == "end":
                self.next()
                self.parse_raw_group()
                rows.append(cur)
                break
            if t.kind == "sep":
                self.next()
                if t.val == "&":
                    continue
                rows.append(cur)
                cur = []
                continue
            cur.append(self.parse_expr({"&", "\\\\"}))
        rows = [r for r in rows if r]
        if not rows:
            raise ValueError("空矩阵")
        return Node("matrix", val=label, children=[Node("row", children=r) for r in rows])


def parse(s: str) -> Node:
    p = Parser(tokenize(s.strip()))
    node = p.parse_expr()
    if p.peek() is not None:
        raise ValueError(f"尾部未消费: {p.peek().val!r}")
    return node


# ---------------- 朗读 ----------------

def _ns(s: str) -> str:
    return s.replace(" ", "")


def _sub_from(sub: Node) -> str:
    """积分/求和下限的 'i=1' 读作 'i 从 一'。"""
    if sub.kind == "seq":
        segs = ["从" if s == "等于" else s for s in sub.val]
        return _seq_speech(segs, attach=False)
    return to_speech(sub)


def _attach_scripts(base: str, node: Node) -> str:
    """上下标附着（平方/立方/次方/度/下标）；bigop/sqrt/frac 等自行处理。"""
    out = base
    if node.sub is not None:
        sub = node.sub
        if sub.kind in ("var", "num", "greek") and sub.sup is None and sub.sub is None:
            out += f" {to_speech(sub)}"
        else:
            out += f" 下标 {to_speech(sub)}"
    if node.sup is not None:
        sup = node.sup
        if sup.kind == "cmd0" and sup.val == "度":
            out += " 度"
        elif sup.kind == "num" and sup.val == "2":
            out += " 平方"
        elif sup.kind == "num" and sup.val == "3":
            out += " 立方"
        else:
            out += f" 的 {to_speech(sup)} 次方"
    return out


def _seq_speech(segs: list, attach: bool) -> str:
    """seq 段表通用装配：运算符词 + 节点朗读；attach 时给可附着节点补上标。"""
    parts, prev_node = [], None
    for seg in segs:
        if isinstance(seg, str):
            parts.append(seg)
            prev_node = None
            continue
        if prev_node is not None and prev_node.kind == "num" and seg.kind == "frac":
            parts.append("又")
        sp = to_speech(seg)
        if attach and seg.kind in _ATTACHABLE or seg.kind == "num" and (seg.sup is not None or seg.sub is not None):
            sp = _attach_scripts(sp, seg)
        parts.append(sp)
        prev_node = seg
    return " ".join(p for p in parts if p)


def to_speech(node: Node) -> str:
    k = node.kind
    if k == "seq":
        return _seq_speech(node.val, attach=False)
    if k == "neg":
        return "负 " + to_speech(node.children[0])
    if k == "num":
        return speak_number(node.val)
    if k in ("var", "greek", "cmd0"):
        return node.val
    if k == "text":
        text = node.val
        if text in UNITS:
            return UNITS[text]
        if re.fullmatch(r"[a-zA-Z/^0-9]+", text):
            return " ".join(text)
        return text
    if k == "func":
        return FUNCS[node.val]
    if k == "fap":
        return f"{to_speech(node.children[0])} {to_speech(node.children[1])}"
    if k == "bigop":
        base = node.val
        if node.sub is not None and node.sup is not None:
            return f"对 {_sub_from(node.sub)} 到 {to_speech(node.sup)} {base}"
        if node.sub is not None:
            return f"对 {_sub_from(node.sub)} {base}"
        return base
    if k == "frac":
        den = _ns(to_speech(node.children[1]))
        num = to_speech(node.children[0])
        if " " in num:
            return f"{den} 分之 {num}"
        return f"{den}分之{num}"
    if k == "binom":
        return f"{to_speech(node.children[0])} 取 {to_speech(node.children[1])} 的组合数"
    if k == "sqrt":
        inner = node.children[0]
        sp = to_speech(inner)
        if node.sub is not None:
            return f"{to_speech(node.sub)} 次根号 {sp}"
        if inner.kind in ("var", "num", "greek", "frac"):
            return f"根号 {sp}"
        return f"根号下 {sp}"
    if k == "group":
        inner = to_speech(node.children[0])
        if node.val == "abs":
            return f"绝对值 {inner}"
        if node.val == "paren":
            return f"括号 {inner} 闭括号"
        return inner
    if k == "vec":
        return f"向量 {to_speech(node.children[0])}"
    if k == "wrap":
        return to_speech(node.children[0])
    if k == "postfix":
        return f"{to_speech(node.children[0])} 的阶乘"
    if k == "percent":
        return f"百分之 {to_speech(node.children[0])}"
    if k == "logof":
        return f"以 {node.val} 为底 {to_speech(node.children[0])} 的对数"
    if k == "lnof":
        return f"{to_speech(node.children[0])} 的自然对数"
    if k == "matrix":
        rows = []
        for i, row in enumerate(node.children):
            cells = "、".join(to_speech(c) for c in row.children)
            rows.append(f"第 {CN_ORD[i]} 行 {cells}")
        return f"{node.val}，" + "，".join(rows)
    raise ValueError(f"未知节点: {k}")


def _chem_try(node: Node) -> str | None:
    """整式为化学式（元素符号+下标数字）时查通用名。"""
    if node.kind != "seq":
        c = node
        if c.kind == "var" and c.sub is not None and c.sub.kind == "num":
            return CHEM_NAME.get(f"{c.val}_{c.sub.val}")
        if c.kind == "var" and len(c.val) <= 2 and c.val[0].isupper():
            return CHEM_NAME.get(c.val)
        return None
    # 按"大写字母(+小写字母)(+_下标数字)"切分元素符号
    units: list[str] = []
    i = 0
    segs = node.val
    while i < len(segs):
        seg = segs[i]
        if not isinstance(seg, Node) or seg.kind != "var" or not re.fullmatch(r"[A-Z]", seg.val):
            return None
        symbol = seg.val
        if seg.sup is not None:
            return None
        if seg.sub is not None:
            if seg.sub.kind == "num":
                symbol += f"_{seg.sub.val}"
                units.append(symbol)
                i += 1
                continue
            return None
        if i + 1 < len(segs) and isinstance(segs[i + 1], Node) \
                and segs[i + 1].kind == "var" and re.fullmatch(r"[a-z]", segs[i + 1].val) \
                and segs[i + 1].sub is None and segs[i + 1].sup is None:
            symbol += segs[i + 1].val
            i += 1
        units.append(symbol)
        i += 1
    return CHEM_NAME.get("".join(units))


def _transform_seq(node: Node) -> Node:
    """seq 级变换：化学式 / lim / log / ln。"""
    chem = _chem_try(node)
    if chem:
        return Node("cmd0", chem)
    segs = node.val
    nodes = [x for x in segs if isinstance(x, Node)]
    lim = next((x for x in nodes if x.kind == "func" and x.val == "lim"), None)
    if lim is not None:
        rest = [x for x in segs if x is not lim]
        body = to_speech(Node("seq", val=rest)) if rest else ""
        cond = to_speech(lim.sub) if lim.sub is not None else ""
        if cond:
            return Node("text", f"当 {cond} 时，{body} 的极限")
        return Node("text", f"{body} 的极限")
    # \int_{a}^{b} body dvar -> body 对 var 从 a 到 b 的积分
    int_idx = next((i for i, x in enumerate(segs)
                    if isinstance(x, Node) and x.kind == "bigop" and x.val == "积分"
                    and x.sub is not None and x.sup is not None), None)
    if int_idx is not None:
        intn = segs[int_idx]
        tail = segs[int_idx + 1:]
        var = None
        body_segs = tail
        if (len(tail) >= 2 and isinstance(tail[-2], Node) and tail[-2].kind == "var"
                and tail[-2].val == "d" and tail[-2].sub is None and tail[-2].sup is None
                and isinstance(tail[-1], Node) and tail[-1].kind == "var"
                and tail[-1].sub is None and tail[-1].sup is None):
            var = tail[-1].val
            body_segs = tail[:-2]
        body_sp = _seq_speech(body_segs, attach=False) if body_segs else ""
        a, b = _sub_from(intn.sub), to_speech(intn.sup)
        if var:
            return Node("text", f"{body_sp} 对 {var} 从 {a} 到 {b} 的积分")
        return Node("text", (f"对 {a} 到 {b} 的积分 {body_sp}").strip())
    # log_{b} x / ln x 合并相邻项
    out, i = [], 0
    changed = False
    while i < len(segs):
        s = segs[i]
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if isinstance(s, Node) and s.kind == "func" and s.val == "log" \
                and s.sub is not None and isinstance(nxt, Node):
            out.append(Node("logof", val=to_speech(s.sub), children=[nxt]))
            i += 2
            changed = True
            continue
        if isinstance(s, Node) and s.kind == "func" and s.val == "ln" and isinstance(nxt, Node):
            out.append(Node("lnof", children=[nxt]))
            i += 2
            changed = True
            continue
        out.append(s)
        i += 1
    if changed:
        return Node("seq", val=out)
    return node


_ATTACHABLE = ("var", "func", "group", "fap")


def speech_of(node: Node) -> str:
    if node.kind == "seq":
        node = _transform_seq(node)
        if node.kind != "seq":
            return to_speech(node)
        return _seq_speech(node.val, attach=True)
    if node.kind in _ATTACHABLE:
        return _attach_scripts(to_speech(node), node)
    return to_speech(node)


# ---------------- 公式定位 ----------------

def _consume_formula(text: str, start: int) -> int:
    n = len(text)
    i = start
    depth = 0
    while i < n:
        c = text[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", text[i:])
            if not m:
                if i + 1 < n and text[i + 1] in ",;:!% ":
                    i += 2
                    continue
                break
            if m.group(1) == "begin":
                env_end = text.find("\\end", i)
                if env_end == -1:
                    return start
                close = text.find("}", env_end)
                if close == -1:
                    return start
                return close + 1
            i += len(m.group(0))
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                k = i + 1
                while k < n and text[k] in "^_":
                    k += 1
                    if k < n and text[k] == "{":
                        d = 1
                        k += 1
                        while k < n and d:
                            if text[k] == "{":
                                d += 1
                            elif text[k] == "}":
                                d -= 1
                            k += 1
                    else:
                        k += 1
                i = k
                while i < n and text[i] in " +-*/=<>&|":
                    i += 1
                    if i < n and text[i] == "\\":
                        sub = _consume_formula(text, i)
                        if sub == i:
                            break
                        i = sub
                    elif i < n and (text[i].isalnum() or text[i] == "."):
                        while i < n and (text[i].isalnum() or text[i] in " ."):
                            i += 1
                return i
        elif depth > 0:
            i += 1
            continue
        else:
            break
    return start


def find_formulas(text: str) -> list[tuple[int, int, str]]:
    """静态文本中定位公式段。返回 [(start, end, latex)]。"""
    spans = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            j = _consume_formula(text, i)
            if j > i:
                spans.append((i, j, text[i:j]))
                i = j
                continue
        elif c == "$":
            j = text.find("$", i + 1)
            if j != -1 and j - i <= 200 and "\n" not in text[i:j]:
                spans.append((i, j + 1, text[i + 1:j]))
                i = j + 1
                continue
        elif c.isascii() and c.isalnum():
            j = i
            while j < n and (text[j].isascii() and (text[j].isalnum() or text[j] in " ^_{}().,+-*/=<>!")):
                j += 1
            run = text[i:j].strip()
            nxt = text[j] if j < n else ""
            if run and not (run.isdigit() and nxt and "一" <= nxt <= "鿿"):
                if not re.fullmatch(r"[A-Za-z]", run):
                    if (re.search(r"[\^_=+*/<>!]|\\", run)
                            or re.search(r"\d\.\d", run)
                            or (" " in run and re.search(r"[=+*/<>]|\d", run))):
                        spans.append((i, j, run))
        i += 1
    return spans


# ---------------- SSML 桥接 ----------------

POLYPHONES = {
    "长度": "chang2 du4", "周长": "zhou1 chang2", "边长": "bian1 chang2",
    "半径": "ban4 jing4", "增长": "zeng1 zhang3", "长方": "chang2 fang1",
    "重量": "zhong4 liang4", "质量": "zhi4 liang4", "重复": "chong2 fu4",
    "差值": "cha1 zhi2", "误差": "wu4 cha1", "相差": "xiang1 cha4",
    "乘以": "cheng2 yi3", "乘积": "cheng2 ji1",
}


def annotate_polyphones(text: str) -> str:
    for w in sorted(POLYPHONES, key=len, reverse=True):
        if w in text:
            text = text.replace(w, f'<sub alias="{POLYPHONES[w]}">{w}</sub>')
    return text


def to_ssml(pairs: list[tuple[str, str]], wrap: bool = True) -> str:
    """(前文文本, 公式朗读) 列表 -> SSML：公式后停顿，文本多音字标注。"""
    parts = []
    for text, sp in pairs:
        if text:
            parts.append(annotate_polyphones(text))
        if sp:
            parts.append(f"{sp}<break time='250ms'/>")
    body = "".join(parts)
    return f"<speak>{body}</speak>" if wrap else body


# ---------------- 自检 ----------------

if __name__ == "__main__":
    samples = [
        "\\frac{1}{2}", "\\frac{\\sqrt{3}}{2}", "1\\frac{1}{2}",
        "\\sqrt{x^2 + 1}", "\\sqrt[3]{8}", "\\sqrt{\\frac{1}{4}}",
        "x^2", "x^{n+1}", "a_{n-1}", "v_0", "\\alpha = 30^\\circ",
        "3 \\times 4 = 12", "x \\to \\infty", "x = \\pm 5",
        "\\sum_{i=1}^{n} i", "\\int_{0}^{1} x \\, dx",
        "m = 7\\,\\text{kg}", "v = 72\\,\\text{km/h}", "CO_2", "CaCO_3",
        "\\lim_{x \\to 0} \\frac{\\sin x}{x}", "\\log_{2}8", "\\ln x + 1",
        "n!", "\\binom{n}{k}", "\\vec{a}", "\\angle A = 60^\\circ",
        "\\triangle ABC \\cong \\triangle DEF", "A \\cup B", "a \\in \\mathbb{R}",
        "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}",
        "\\begin{vmatrix} 1 & 2 \\\\ 3 & 4 \\end{vmatrix}",
        "50\\%", "f(x) = x^2", "(a+b)^2", "|x-1|",
        "\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}",
    ]
    bad = 0
    for s in samples:
        try:
            print(f"{s}\n  => {speech_of(parse(s))}")
        except ValueError as e:
            bad += 1
            print(f"{s}\n  !! {e}")
    print(f"\n解析失败: {bad}/{len(samples)}")
