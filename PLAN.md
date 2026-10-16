# 文心公式朗读转换模型 · 后训练执行计划

> 给执行者（新会话）：本文档自包含全部背景与操作步骤。目标是在本机（Apple M4 Max / 48GB）完成
> 一次可复现的 LoRA 后训练验证，产出"基座 vs 微调后"的公式朗读转换准确率对比数字。

## 一、项目背景与目标

- **业务场景**：K12 学习机 AI 讲题，LLM 生成的讲解文本含 LaTeX 公式（如 `\frac{1}{2}`、`\sqrt{x}`），
  TTS 引擎（豆包/Minimax 等）会直接念出 LaTeX 源码，体验极差。
- **解决方案**：在 LLM 与 TTS 之间加"公式朗读转换层"——把 LaTeX 公式转成中文口语化朗读脚本
  （如 `\frac{1}{2}` → "二分之一"，`\sqrt{x}` → "根号 x"）。
- **本项目目标**：用百度开源的 ERNIE-4.5 小模型做 LoRA 后训练，验证"转换任务可后训练"，
  产出对比数字（基座准确率 X% → LoRA 后 Y%），用于简历个人项目板块与面试素材。
- **最终产物**：train/valid/test 数据集、LoRA adapter 权重、三组评测对比表、一页结果小结。

## 二、环境现状（Phase 0 已完成，勿重复安装）

| 项 | 值 |
|---|---|
| 机器 | Apple M4 Max，48GB 统一内存，磁盘充足 |
| uv | `~/.local/bin/uv`（已安装 Python 3.12.15） |
| 项目目录 | `~/Developer/mathspeak` |
| 虚拟环境 | `~/Developer/mathspeak/.venv`（已装 mlx-lm、transformers 5.20） |
| Python | 一律使用 `~/Developer/mathspeak/.venv/bin/python`，**不要用系统 Python 3.9** |
| 模型源 | 国内镜像：所有下载命令前加环境变量 `HF_ENDPOINT=https://hf-mirror.com` |
| 已验证模型 | `mlx-community/ERNIE-4.5-0.3B-PT-4bit`（已下载，冒烟通过。注意：基座模型对指令会跑题，属正常现象） |

快速自检（应输出"模型加载成功"）：

```bash
cd ~/Developer/mathspeak
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c "
from mlx_lm import load
model, tok = load('mlx-community/ERNIE-4.5-0.3B-PT-4bit')
print('模型加载成功')"
```

## 三、Phase 1：数据构造（最关键，预计半天）

目标产物：`data/train.jsonl`（~2000 条）、`data/valid.jsonl`（~200 条）、`data/test.jsonl`（200 条，**冻结，训练评测全程不碰**）。

### 3.1 数据 schema

```json
{"prompt": "将讲解中的公式改写为适合语音朗读的中文口语，只输出改写后的整句：\n原句：因为 \\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}，所以选 C。", "completion": "因为 二分之一 加 三分之一 等于 六分之五，所以选 C。"}
```

要点：prompt 带固定指令头（后续训练和评测统一）；completion 是整句改写（保留非公式部分原文），
数字读法遵循中文朗读习惯（分数读"几分之几"、`2026` 读"二零二六"或"两千零二十六"按场景统一）。

### 3.2 规则模板生成器 `generate_data.py`

覆盖 8 类模板，每类多个句式变体，随机扰动数字/变量/系数：

1. 分数 `\frac{a}{b}`（含带分数、连分数）
2. 根式 `\sqrt{x}`、`\sqrt[n]{x}`
3. 上下标 `x^2`、`x^{n+1}`、`a_n`、`v_0`
4. 希腊字母 `\alpha \beta \theta \pi \Delta \Sigma \Omega \lambda \mu`
5. 运算与关系符 `\pm \times \div \leq \geq \neq \approx \infty \sum \prod \int`
6. 物理公式与单位（`v=\frac{s}{t}`、`\rho=\frac{m}{V}`、`F=ma`、`m/s`、`km/h`、`kg·m/s²`）
7. 化学式（`CO_2`、`H_2SO_4`、`2H_2+O_2=2H_2O`）
8. 混合嵌套（如 `\frac{\sqrt{x^2+1}}{2}`）

把模板句嵌入 10–20 个讲题句式（"所以选 X"、"这一步要注意"、"代入得"等），生成后去重。
写生成器时注意：completion 里中文与数字之间加空格（TTS 友好）。

### 3.3 抽检

随机抽 100 条打印出来给人工过目（生成器 bug 高发区：嵌套括号、多音字"长/重/差"、单位读法）。
发现问题回改模板，不要带病进训练。

### 3.4 划分

train : valid : test ≈ 2000 : 200 : 200，按模板类别分层抽样，**test 冻结**。

### 3.5 （可选）教师模型增强

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c "
from mlx_lm import load
model, tok = load('mlx-community/ERNIE-4.5-21B-A3B-PT-4bit')  # 约12GB，48GB内存可跑
print('教师模型加载成功')"
```

用途：给规则模板没覆盖的长尾公式（矩阵、行列式、极限、对数）生成样本；给规则数据做口语化润色。
生成结果必须抽样人工核对后再入库。

## 四、Phase 2：LoRA 训练（机器跑 0.5–2 小时）

```bash
cd ~/Developer/mathspeak
# 下载全精度基座（约600MB）
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -c "
from mlx_lm import load
load('mlx-community/ERNIE-4.5-0.3B-PT-bf16')
print('bf16 模型就绪')"

# 启动 LoRA 训练
.venv/bin/python -m mlx_lm.lora \
  --model mlx-community/ERNIE-4.5-0.3B-PT-bf16 \
  --train --data data \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size 4 \
  --iters 1500 \
  --learning-rate 1e-5 \
  --steps-per-report 50 \
  --steps-per-eval 200 \
  --adapter-path adapters \
  --save-every 500
```

观察项：train loss 应持续下降（起始通常 3–6，收敛到 0.5 以下量级为正常）；
valid loss 若上升说明过拟合，提前停并用较早 checkpoint。
（mlx_lm 约定：data 目录下放置 train.jsonl / valid.jsonl / test.jsonl。）

训练完成后（可选）融合：

```bash
.venv/bin/python -m mlx_lm.fuse --model mlx-community/ERNIE-4.5-0.3B-PT-bf16 \
  --adapter-path adapters --save-path fused-model
```

## 五、Phase 3：评测（产出对比数字）

写 `evaluate.py`，对 test.jsonl 的 200 条分别跑三组：

| 组 | 配置 |
|---|---|
| A 基座直调 | `ERNIE-4.5-0.3B-PT-4bit`，只给指令头 |
| B prompt 工程 | 同上，指令头 + 3 条 few-shot 示例 |
| C LoRA 后 | `bf16 基座 + adapters`，同一指令头 |

指标双口径：
- **完全匹配率**：输出与参考 completion 完全一致（去空格/标点归一化后）；
- **关键片段匹配率**：公式部分的读法正确（可按公式段抽取比对）。

输出 markdown 对比表 + badcase 清单（按 3.2 的 8 类归类）。
若 C 组某类错误集中，回 Phase 1 补该类数据，再训一轮（正常迭代 1–2 轮）。

## 六、Phase 4：结果小结与简历表述

写 `RESULT.md`：数据规模、训练配置、loss 曲线结论、三组对比表、badcase 分析、迭代记录。

简历表述模板（数字填真实值，放个人项目板块，**不要写进实习经历**）：

> 基于文心开源模型完成公式朗读转换后训练验证：构造 LaTeX→口语化朗读脚本数据集 Xk 条，
> LoRA 微调 ERNIE-4.5-0.3B，朗读转换准确率由 X% 提升至 Y%

## 七、故障处理

| 症状 | 处理 |
|---|---|
| HuggingFace 下载超时 | 确认命令带 `HF_ENDPOINT=https://hf-mirror.com` |
| 内存压力（swap 暴涨） | 换 4bit 模型训练，或 batch-size 降到 2 |
| loss 不降 | 检查数据格式（prompt/completion 字段名）、学习率调到 5e-6 或 5e-5 |
| 生成乱码/复读 | max_tokens 调小、加 repetition penalty；检查训练数据是否有重复模板污染 |
| valid loss 上升 | 提前停训，取较早 checkpoint |

## 八、执行纪律

1. test.jsonl 从生成起冻结，任何训练、调 prompt 都不许看它；
2. 每组评测的指令头、max_tokens、温度（建议 0）保持一致，确保对比公平；
3. 所有数字必须来自真实评测输出，不许估算；
4. 每完成一个 Phase，向用户汇报产物与关键数字，用户确认后进下一阶段。
