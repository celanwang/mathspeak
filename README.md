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
| C | LoRA 后（bf16 + adapters，第一轮） | 100.0% | 100.0% |
| C2 | LoRA 第二轮（+长尾蒸馏数据） | 99.5% | 99.5% |

长尾 OOD 测试集（11 类训练分布外公式，198 条）：adapter 完全匹配
**7.1%（第一轮）→ 99.5%（第二轮）**。

> 关于 C 与 C2 的数字口径：第一轮的 100% 是**封闭分布内**的规则复现
> （test 与 train 同一生成器），证明任务可后训练；第二轮用 0.5% 的规则内
> 代价（1/200 条，已定位为分数分子分母反转一类错误）换取 OOD 长尾
> 7.1% → 99.5%，生产价值严格更优。评测明细见 `results/`，
> 结论与复现过程见 `RESULT.md`。

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

# 4. 生产链路组件
.venv/bin/python stream_convert.py     # 流式增量转换（乐观转换+回滚+延迟报告）
.venv/bin/python gen_ood_data.py       # 长尾 OOD 基准
.venv/bin/python distill.py            # 教师蒸馏（21B 生成 + 硬检查过滤）
.venv/bin/python router.py             # 端云路由（规则覆盖度）
.venv/bin/python verify.py --dump results/outputs_eval.json  # 忠实性硬检查
```

## 架构

```
generate_data.py    规则模板数据生成器（8 类公式 × 20 句式）
latex_speech.py     朗读规范唯一事实源：解析器 + 朗读生成 + SSML 桥接
stream_convert.py   流式增量转换器（乐观转换 + 回滚，87% 整句一致，p95 0.004ms）
gen_ood_data.py     长尾 OOD 冻结基准（11 类，参考由规范引擎产出）
distill.py          教师蒸馏管线（21B 生成 → 守恒硬检查 → 训练集）
router.py           端云路由（规则覆盖度决定端侧/云端）
verify.py           忠实性硬检查（守恒原子缺一即否决）
evaluate.py         A/B/C 三组评测（双指标 + badcase 归类 + 输出转储）
data/               数据集（test/ood_test 冻结）
adapters/           LoRA 权重（不入库）
results/            评测与路由/流式/校验报告
PLAN.md             执行计划
RESULT.md           结果小结（含第二阶段数字总表）
```

## 数据

- `data/train.jsonl` / `valid.jsonl` / `test.jsonl`：prompt/completion 对，固定指令头 +
  含 LaTeX 原句 → 整句口语化朗读脚本（中文与数字间空格，TTS 友好）。
- 规则模板覆盖 8 类公式：分数、根式、上下标、希腊字母、运算与关系符、物理公式与单位、
  化学式、混合嵌套；嵌入 20 个讲题句式，数字/变量随机扰动。
- `data/spot_check.txt`：100 条人工抽检样本。test 集从生成起冻结，训练与调优全程不碰。
- `data/ood_*.jsonl`：长尾 OOD 基准（极限/对数/矩阵/行列式/阶乘/组合数/几何/集合/
  向量/百分号/函数），参考朗读由 latex_speech 规范引擎生成。
