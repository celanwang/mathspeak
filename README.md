# mathspeak

LaTeX 公式朗读转换：把 K12 讲题文本中的 LaTeX 公式改写成适合 TTS 朗读的中文口语脚本。

LLM 生成的讲解文本含有 `\frac{1}{2}`、`\sqrt{x}` 这类公式，TTS 引擎会直接念出 LaTeX 源码，体验极差。
本项目基于百度开源的 **ERNIE-4.5-0.3B-PT**，在 Apple Silicon（mlx-lm）上完成 LoRA 后训练验证，
产出"基座 vs prompt 工程 vs LoRA 后"的朗读转换准确率对比。

## 效果（200 条冻结测试集，温度 0）

| 组 | 配置 | 完全匹配率 | 关键片段匹配率 |
|---|---|---|---|
| A | 基座直调（4bit, zero-shot） | 0.0% | 0.5% |
| B | prompt 工程（4bit, 3-shot） | 2.5% | 3.0% |
| C | LoRA 后（bf16 + adapters） | 100.0% | 100.0% |

> 评测明细见 `results/eval_report.md`，结论与复现过程见 `RESULT.md`。

## 快速开始

```bash
# 环境：macOS + Apple Silicon，Python 3.12（uv/venv），依赖 mlx-lm
uv venv .venv && uv pip install mlx-lm
# 国内下载模型需镜像
export HF_ENDPOINT=https://hf-mirror.com

# 1. 构造数据（2000/200/200，test 冻结）
.venv/bin/python generate_data.py --out data

# 2. LoRA 训练（M4 Max 约 2 分钟）
.venv/bin/python -m mlx_lm.lora \
  --model mlx-community/ERNIE-4.5-0.3B-PT-bf16 \
  --train --data data \
  --fine-tune-type lora --num-layers 16 \
  --batch-size 4 --iters 1500 --learning-rate 1e-5 \
  --steps-per-report 50 --steps-per-eval 200 \
  --adapter-path adapters --save-every 500

# 3. 评测（A/B/C 三组）
.venv/bin/python evaluate.py --groups A B C
```

## 数据

- `data/train.jsonl` / `valid.jsonl` / `test.jsonl`：prompt/completion 对，固定指令头 +
  含 LaTeX 原句 → 整句口语化朗读脚本（中文与数字间空格，TTS 友好）。
- 规则模板覆盖 8 类公式：分数、根式、上下标、希腊字母、运算与关系符、物理公式与单位、
  化学式、混合嵌套；嵌入 20 个讲题句式，数字/变量随机扰动。
- `data/spot_check.txt`：100 条人工抽检样本。test 集从生成起冻结，训练与调优全程不碰。

## 目录

```
generate_data.py   # 规则模板数据生成器
evaluate.py        # A/B/C 三组评测（完全匹配 + 关键片段双指标）
data/              # 数据集（test 冻结）
adapters/          # LoRA 权重（不入库）
results/           # 评测报告
PLAN.md            # 执行计划
RESULT.md          # 结果小结
```

## 简历表述

> 基于文心开源模型完成公式朗读转换后训练验证：构造 LaTeX→口语化朗读脚本数据集 2k 条，
> LoRA 微调 ERNIE-4.5-0.3B，朗读转换准确率由 X% 提升至 Y%（数字见 RESULT.md）
