# Tokenizer Techniques: What Exists

本文档记录截至 2026-04-20 在 parameter-golf 社区出现的所有有效 tokenizer 技术路线的实现细节和 BPB 算法。

思考层（为什么这些技术 work、何时选择哪条路线、下一步做什么）见 `how-to-think-about-them-2026-04-20.md`。

相关文档：
- `eda-findings-tokenizer.md` — per-token BPB 分析，识别 word fragment 等难预测 token
- `scylla-v2-vs-sp8192.md` — Scylla tokenizer 为什么在 BPB 指标下是 dead-end

目录：
- §0 全景一览
- §1 Gravity Tokenizer
- §2 CaseOps（无损双射路线）
- §3 Casefold（有损归一路线）
- §4 Lowercase + SP10240 + FreqGPTQ
- §5 完整 Tokenizer PR Landscape
- §6 技术对比矩阵
- §7 BPB 计算详解
- §8 关键 PR 索引

---

## 0. 全景一览

社区里发现了 5 条独立的 tokenizer 技术路线，按 BPB 效果排列：

| # | 路线 | 代表 PR | Best BPB (ALIVE) | 核心机制 | 合规风险 |
|---|---|---|---|---|---|
| 1 | **Pre-Quant TTT + CaseOps** | #1738 (alertcat) | **1.0354** | 无损双射 + 量化前 TTT (21ep) | 低 |
| 2 | **Gravity Tokenizer** | #755 (dcrow85) | **1.0321** | Ablation leverage 替换 86% vocab | 低但 byte accounting 争议 |
| 3 | **Casefold V4 + Phased TTT** | #1693 (dexhunter) | **1.0573** | 有损 casefold + multi-phase SGD TTT | 中 (Issue #1604 pending) |
| 4 | **Casefold V2 + Systems Opt** | #1585 (codemath3000) | **1.0639** | 有损 casefold + BPB scored refill | 低 |
| 5 | **Lowercase + SP10240 + FreqGPTQ** | #1707 (nothingLiva) | **1.0740** | `.casefold()` + 更大 vocab + 频率加权量化 | 低 |

这 5 条路线的 ALIVE PR 在 leaderboard_legal.md 中大部分未出现——因为 legal 过滤基于 `compliance_keywords`，而很多新 PR 带有 `gptq_eval` 或 `cache_eval` 关键词被误滤。实际合规性需要逐 PR 检查。

**关于 vocab size 的定位（重要）：**
- EDA（见 `eda-findings-tokenizer.md`）表明 `SP8192 → SP16384` 的 BPB 边际收益极小（-0.003 ~ -0.008），但 embedding 参数预算增长线性。在 16MB artifact 预算下，**SP8192 是 vanilla BPE 的 sweet spot**。
- SP10240（PR #1707）不是简单地"把 vocab 做大"——它的增益主要来自和 `.casefold()` 的协同（所有 slots 都是有效 subword，没有 case 重复占位）。**单独扩大 vocab 到 10240 不值得。**
- Scylla (TokenMonster, V=1254) 属于"不强调压缩"的 tokenizer 设计，在 BPB 指标下是 dead-end。本文不讨论这条路线。

---

## 1. Gravity Tokenizer — PR #755 (dcrow85, BPB 1.0321)

### 原理

**不改文本预处理，只改哪些 token 占 vocab slot。** 在固定 1024 vocab 下，用 "ablation leverage"（消融杠杆）替代 BPE 频率作为 token 选择标准。

**关键洞察：** 频率 ≠ 结构重要性。有些 token 出现频繁但模型不依赖（去掉后 loss 几乎不变）；有些 token 出现较少但是 "承重墙"（去掉后下游 loss 暴涨）。

### 算法

1. **候选生成**：BPE merge table → 3,058 个候选（freq ≥ 1,000，排除 byte-level）
2. **Ablation leverage scoring**：
   - 冻结 GPT-2 作为参考模型（避免 tokenizer 自引用偏差）
   - 每个候选在 100 个 FineWeb context 上评估
   - 对比 "intact text" vs "shattered text"（把目标 token 的字符拆成空格分隔）
   - `leverage = mean(loss_shattered) - mean(loss_intact)` over K=10 downstream tokens
   - 早停：30 context 后如果 mean + 2SE < 0，跳过剩余 70 context
3. **Scoring**：`score(t) = freq_norm(t)^(1-β) * leverage_norm(t)^β`，β=1.0（纯 leverage）
4. **Vocab 构建**：取 top 765 → SentencePiece Unigram (byte fallback)
5. **Retokenize** 全 FineWeb

### 结果

- **659 / 765 merge tokens 被替换**（86%）
- 压缩率 1.05 bytes/token（BPE: 2.45 bytes/token）——Gravity 比 BPE **差 2.3 倍**
- 但 per-token 预测质量的提升远超压缩损失 → 净 BPB 1.0321

### 为什么 work

"The Depth Efficiency Law"（dcrow85 的理论框架）：

- High-leverage token（语义晶体）使用全部 12 层 transformer 的计算深度
- Low-leverage byte-gas token 呈 U 型速度曲线：前几层挣扎 → 中间层空转 → 最后一层恐慌
- 12 层 BPE 模型对 byte-gas 部分实际上只有 ~3-4 层有效深度
- Gravity tokenizer 不增加层数，而是让现有层数对所有 token 都 usable

### 架构

极简（用来隔离 vocab 效果）：12L, 384d, 6H/2KV, 3× MLP, relu², seq 2048, tied embeddings, int8+zlib。**没有任何 fancy 组件**（无 TTT, 无 EMA, 无 SmearGate, 无 BigramHash）。

### Dead-ends (dcrow85 记录)

| 实验 | 结果 | 原因 |
|---|---|---|
| Gravitational lensing (attention 路由假说) | KILLED | 56% 是 RoPE positional artifact，不是 learned geometry |
| Bifurcation scoring (BPE merge tree 上的 delta-leverage) | +0.021 BPB | 去掉了 individually redundant 但 collectively essential 的连接组织 |
| Warm-start embeddings (BPE 嵌入的空间均值初始化) | +0.038 BPB | 灾难性初始不匹配（27.5 vs 5.4 BPB）——crystallization 是全模型 phase transition，不是 embedding-local |
| Breadth measure (leverage 分布的熵) | 无区分力 | std=0.15，dropped from final score |

### SP8192 scale 上的未知

Gravity 在 SP1024 上 work，因为 1024 vocab 下每个 slot 的边际价值极高。在 SP8192 下 slot 边际价值低得多，leverage scoring 的 ROI 是否仍然正？**未验证。** dcrow85 没有在 SP8192 上试过。

### HuggingFace

```bash
huggingface-cli download dcrow85/gravity-tokenizer-fineweb \
    tokenizers/gravity_beta_1.0.model --repo-type dataset --local-dir ./data
```

---

## 2. CaseOps（无损双射路线）

### 2.1 谱系与演化

```
PR #1729 (romeerp, 1.0678)        ← CaseOps 概念创始 + byte sidecar 合规先例
  → PR #1736 (dexhunter, 1.0655, AT_RISK)  ← +GatedAttn +QuantGate +Loop45 +PhasedTTT
  → PR #1738 (alertcat, 1.0354)             ← +Pre-Quant TTT (21 epochs, PR #1735 stack)
```

### 2.2 核心机制

**双射文本变换**：大写字母 → 控制符号 + 小写字母，可完全逆转。

```python
# lossless_caps_caseops_v1 (833 行源码，v1-v7 共 7 个版本)
# v2 版本使用 4 个 Private Use Area 控制字符：
TITLE    = "\uE001"   # 下一个 word TitleCase
ALLCAPS  = "\uE002"   # 下一个 word/region 全大写
CAPNEXT  = "\uE003"   # 下一个 letter 大写
ESC      = "\uE004"   # 转义

"The NASA Launch" → "<TITLE>the <ALLCAPS>nasa <TITLE>launch"
"iPhone OpenAI"   → "i<CAPNEXT>phone <TITLE>open<CAPNEXT>a<CAPNEXT>i"
```

控制 token 作为 SentencePiece `user_defined_symbols` 进入 vocab，占 4 个 slot。

### 2.3 Byte Sidecar（合规关键）

BPB 字节分母**不从 tokenizer piece 计算**，而是从预生成的 uint16 sidecar 文件读取：

```
fineweb_val_000000.bin        # token stream (post-transform)
fineweb_val_bytes_000000.bin  # per-token 原始 UTF-8 byte count
```

这保证 BPB = bits / 原始字节，不受控制符号 byte 膨胀影响。

PR #1729 (romeerp) 首创此机制，被社区引用为合规先例。`eval_val()`, `eval_val_sliding()`, `eval_val_ttt()` 三处 call site 都 patched。

### 2.4 PR #1738：当前 CaseOps 最佳 (1.0354 BPB)

**关键不是 CaseOps 本身，而是它叠在了 Pre-Quant TTT 栈上。**

Pre-Quant TTT（PR #1735, AjAnubolu, 1.0429）：
- **在量化之前**对 full-precision EMA 模型做 TTT
- 21 epochs AdamW on validation chunks → GPTQ int6 → 输出固定 artifact
- **eval 时零适应**——最终 artifact 是一个 static int6 模型
- 合规优势：不需要争论 eval-time adaptation 是否合法
- 8-GPU 并行联邦平均（每 GPU 处理 1/8 val chunks → `all_reduce(AVG)`）
- Epoch-level cosine LR (5e-4 → 5e-5 across 21 epochs)

CaseOps 在此基础上贡献 **-0.0075 BPB**（1.0429 → 1.0354）。

### 2.5 HuggingFace

```bash
huggingface-cli download romeerp/parameter-golf-caseops-v1 --repo-type dataset
```

---

## 3. Casefold（有损归一路线）

### 3.1 谱系与演化

```
PR #1578 (mikeapedia, 1.0668, INCOMPLETE)  ← 原始 casefold tokenizer 创始
  → PR #1585 (codemath3000, 1.0639)        ← +Parallel Residuals +Systems Opt
  → PR #1670 (dexhunter, 1.0597)           ← 简化 V4 重训 +Multi-Phase SGD TTT
    → PR #1693 (dexhunter, 1.0573)         ← +AttnOutGate +SmearGate
```

### 3.2 V2 vs V4 的关键区别

| 维度 | V2 (mikeapedia, #1578/#1585) | V4 (dexhunter, #1670/#1693) |
|---|---|---|
| Tokenizer 构建 | 8 步 pipeline：BPE 训练 → clean 374 slots → BPB scoring refill (3.2h Rust) → punctuation swap → Unigram 模式 | **从零在 casefold 语料上重训 SP8192 BPE**（简单得多） |
| Byte fallback 处理 | 9.47% → 2.49%（swap 25 个裸标点） | 无特殊处理 |
| Decoding 模式 | Unigram (Viterbi) | 标准 BPE |
| 复杂度 | 高（25GB 语料 + Rust builder + multi-stage） | 低（直接 SP retrain） |
| TTT | Score-first chunk SGD | Multi-Phase Global SGD (3 phases, 2000 prefix docs) |
| BPB | 1.0639 (1.0668 without CUTLASS/systems opt) | 1.0597 (#1670) → 1.0573 (#1693) |

**V4 比 V2 更好（1.0573 vs 1.0639）**，但差距主要来自 **Multi-Phase SGD TTT 栈的改进**，不是 tokenizer 本身的改进。V2 的 tokenizer 工程（BPB scoring refill + punctuation swap）比 V4 精良得多。

**推论：** 最优组合是 V2 tokenizer + V4 TTT 栈——目前没人做过这个实验。

### 3.3 V2 的 8 步 Pipeline（详细版在 CASEFOLD_TOKENIZER.md）

1. `NFKC(text).lower()` 后训练 SP8192 BPE → 零大写 token
2. Clean：删 53 L=1 单字节 + 321 undertrained token → 释放 374 slots
3. 合并 case-variant 候选频率（37.3M → 34.2M 候选）
4. 25GB 小写化语料
5. **BPB scoring refill**（Rust `vocab-builder`，3.2h）：DP encoding + `Σ(old_bits - new_bits)` 打分
   - Top tokens: `'s` (84.2B), `▁1` (82.2B), `▁2` (78.8B), `20` (60.7B), `00` (47.5B)
   - 全是数字 + 缩写——BPE 学不到但 BPB scoring 能找到
6. 转为 Unigram (Viterbi) 模式（删 token 会破坏 BPE merge chain）
7. Swap 25 个最低使用 token → 25 个裸 ASCII 标点（byte fallback 9.47% → 2.49%）
8. Retokenize with NFKC-first fix

**结果：**
- 10.38% fewer tokens（40.5M → 36.3M）
- +1,784 byte 累计误差（+0.001%，来自 NFKC decomposition + 土耳其 İ）
- Post-TTT BPB 改进 -0.0116

### 3.4 Casefold 路线的 Dead-Ends (mikeapedia §8.2)

| 方向 | 结果 | 原因 |
|---|---|---|
| Capcode / `<CAP>` 修饰符 | Dead-end | 修饰符 token +1.5B 个，slots 只省 ~15M → 100× 净惩罚 |
| Full vocab replacement (自底向上) | +1.6% BPB | 27-37% 新 token 破坏了预训练分布 |
| BPE-dropout (stochastic DP) | Dead-end | sub-1-epoch 下 tokenization 多样性换 data 多样性是亏的 |
| Length-MAX scoring (Dong&Su 2025) | Dead-end | 在 <20K vocab 下 BPE 打败它 |
| SP BPE 重训 (7 configs) | 最好 -0.57% | Stock SP 在 1024 已近最优 |

### 3.5 合规状态

- **Issue #1604** 待裁决——核心问题是 casefold 是否算 "合规的 custom tokenizer"
- mikeapedia (#1578) 状态 INCOMPLETE（但这不是合规问题，是提交流程问题）
- codemath3000 (#1585) 状态 ALIVE，3 seeds
- dexhunter (#1670, #1693) 状态 ALIVE

### 3.6 HuggingFace

```bash
# V2 数据集（mikeapedia）
huggingface-cli download Mikeapedia/fineweb10B-sp8192-casefold-v2 --repo-type dataset
# V4 没有 HF 发布——需要自己重建（但简单：直接 SP retrain on lowercased text）
```

---

## 4. Lowercase + SP10240 + FreqGPTQ — PR #1707 (nothingLiva, 1.0740)

### 4.1 核心

**最简路线**：无 TTT、无 Parallel Residuals、无 Depth Recurrence、无任何复杂组件。

- `.casefold()` 预处理（比 `.lower()` 更 aggressive：`"ß" → "ss"`）
- 重训 SP10240 tokenizer on casefolded text
- FreqGPTQ：Top-100 最频繁 token → INT8，其余 → INT6；GPTQ Hessian 2× 权重加成
- 10L, 512d, 8H/4KV, Muon + EMA, seq 2048

### 4.2 FreqGPTQ 详细

来自 nothingLiva 自己的 PR #1042：

1. **Frequency-Weighted GPTQ Calibration**：Hessian 收集时，top-100 token 的 activation 乘以 2× 权重。偏向高频 token 的量化精度。零额外 artifact 开销。
2. **Frequency-Weighted Embedding Quantization**：Top-100 token → INT8，其余 → INT6。高频 token 对 loss 贡献不成比例——precision 应该匹配 impact。
3. **Sandwich Layer (L10 → INT8)**：最后一层用 INT8 保护 LM head 前的信号质量。~0.75 MB 预算。

### 4.3 SP10240 的意义

- 之前版本（SP10240, 混合大小写）= 1.083
- 加 `.casefold()` 后 = **1.074**
- **单一改动 ΔBPB = -0.009**

SP10240 比 SP8192 多 2048 slots。在 casefold 之后，这些 slots 可以全部分配给真正有区分力的 subwords——边际价值更高。

### 4.4 HuggingFace

```bash
huggingface-cli download MissGlitterToken/sp10240_casefold --repo-type dataset
```

---

## 5. 完整 Tokenizer PR Landscape（ALIVE, BPB ≤ 1.10）

| PR# | Author | BPB | 路线 | 关键组件 |
|---:|---|---:|---|---|
| #755 | dcrow85 | 1.0321 | Gravity | Ablation leverage, SP1024, 无 TTT |
| **#1738** | alertcat | **1.0354** | **CaseOps + Pre-Quant TTT** | romeerp CaseOps + AjAnubolu 21ep PreQ-TTT |
| #1735 | AjAnubolu | 1.0429 | Pre-Quant TTT (无 tokenizer 改动) | 8-GPU parallel 21ep AdamW PreQ-TTT, SP8192 |
| #1693 | dexhunter | 1.0573 | Casefold V4 + Phased TTT | +AttnOutGate +SmearGate |
| #1670 | dexhunter | 1.0597 | Casefold V4 + Phased TTT | Multi-Phase Global SGD |
| #1585 | codemath3000 | 1.0639 | Casefold V2 + Systems | mikeapedia tokenizer + msisovic PR |
| #1736 | dexhunter | 1.0655 | CaseOps + PhasedTTT | AT_RISK |
| #1729 | romeerp | 1.0678 | CaseOps + Tapered WD | byte sidecar 合规先例 |
| #1707 | nothingLiva | 1.0740 | Lowercase + SP10240 | +FreqGPTQ, 无 TTT |

---

## 6. 技术对比矩阵

### 6.1 信息保留

| 方法 | 信息丢失 | 影响 |
|---|---|---|
| Gravity | 无（不改文本，只改 vocab 选择） | 零 |
| CaseOps | 无（双射变换，可逆） | 零 |
| Casefold V2/V4 | **丢失大小写** | 模型无法区分 "The" / "the" / "THE" |
| Lowercase + SP10240 | **丢失大小写**（更 aggressive：`ß→ss`） | 同上 + 非 ASCII 大小写归并 |

### 6.2 Vocab 效率

| 方法 | 机制 | 释放 slots | 填充策略 |
|---|---|---|---|
| Gravity | 替换低-leverage token | 659 / 765 (86%) | leverage scoring |
| CaseOps | 去除 case 重复 + 4 控制 token 占 slot | ~1,671 (净) | SP retrain |
| Casefold V2 | 去除 case 重复 + BPB refill | 374 | BPB scoring (Rust DP) |
| Casefold V4 | 去除 case 重复 | ~1,675 | SP retrain (无 refill) |
| Lowercase SP10240 | 去除 case 重复 + 扩大 vocab | 相当于新增 ~3,700 effective | SP retrain |

### 6.3 BPB 字节计数

| 方法 | 字节来源 | 误差 |
|---|---|---|
| Gravity | Tokenizer piece (标准 LUT) | 0 |
| CaseOps | **Per-token byte sidecar**（原始 UTF-8） | 0 (严格) |
| Casefold V2 | Tokenizer piece (casefold LUT) | +0.001% (NFKC + 土耳其 İ) |
| Casefold V4 | 同 V2 | 同 V2 |
| Lowercase SP10240 | Tokenizer piece | 未审计（casefold 比 lower 更 aggressive） |

### 6.4 实施成本

| 方法 | 离线 pipeline | GPU 时间 | 依赖项 | 数据可用性 |
|---|---|---|---|---|
| Gravity | 4h scoring + retokenize | 4h (1 GPU) + GPT-2 | PyTorch + GPT-2 | HF: `dcrow85/gravity-tokenizer-fineweb` |
| CaseOps | `lossless_caps.py` + retokenize + byte sidecar | ~1h | SP only | HF: `romeerp/parameter-golf-caseops-v1` |
| Casefold V2 | 8 步 pipeline + 3.2h Rust builder | 3.2h (Rust) + retokenize | Rust + SP | HF: `Mikeapedia/fineweb10B-sp8192-casefold-v2` |
| Casefold V4 | SP retrain on lowered text | ~10min + retokenize | SP only | 无 HF（需自建）|
| Lowercase SP10240 | SP retrain + retokenize | ~10min + retokenize | SP only | HF: `MissGlitterToken/sp10240_casefold` |

---

## 7. BPB 计算详解

这一节回答三个常见但容易搞错的问题：
1. BPB 的分子分母分别怎么来？
2. Casefold 丢了大小写，分母怎么还能和原文对齐？
3. CaseOps 多了 `<TITLE>` `<ALLCAPS>` 这些控制 token，分子不就爆了吗？

### 7.1 定义

$$\text{BPB} = \frac{\sum_i \text{NLL}(t_i)}{\ln(2) \cdot \sum_i \text{bytes}_i}$$

- 分子：每个 target token 位置的 cross-entropy loss，用 nats 计数 → 除以 `ln(2)` 换成 bits
- **分母：原始 FineWeb UTF-8 字节数**，不是 tokenizer 输出字节数、不是 post-transform 字节数

**Parameter-golf eval 硬约束：** 分母必须是**原文字节**。任何 tokenizer 改动都不能因为"压缩了文本"而让分母变小——否则 BPB 就是无意义的数字。

### 7.2 Baseline SP8192 的算法

```
原文: "The cat"               # 7 bytes (UTF-8)
tokenize: [▁The, ▁cat]        # 2 tokens
  - ▁The piece = " The", len(UTF-8) = 4
  - ▁cat piece = " cat", len(UTF-8) = 4
  - 但 sum(pieces) = 8 ≠ 原文 7——差 1 byte 是 SP 前导空格惯例

Eval 时：
  numerator   = NLL(▁The | <bos>) + NLL(▁cat | <bos>, ▁The)   # 两个 loss 求和
  denominator = 7 × ln(2)                                      # 原文字节
```

实践上，实现会维护一个 **byte LUT**：`base_bytes[tok_id] = len(piece.replace("▁"," ").encode("utf-8"))`，启动时 `sum(LUT over stream) ≈ shard_bytes_header` 校验（差 >1% 直接 abort）。

### 7.3 Casefold V2 的算法

```
原文: "The cat"                            # 7 bytes (原文)
preprocess: "the cat"                      # .lower(), 仍是 7 bytes (ASCII 大小写都 1 byte)
tokenize: [▁the, ▁cat]                     # 2 tokens, 都是 4-byte piece

Eval 时：
  numerator   = NLL(▁the) + NLL(▁cat)
  denominator = 7 × ln(2)                  # ← 仍然是原文 7 bytes
```

**关键不变量：** `len("T".encode()) == len("t".encode()) == 1`。ASCII 大小写字节长度完全一致，所以 `LUT[▁the]` = `LUT[▁The]` = 4。

**这就是为什么 casefold 是"几乎免费"的：**
- 分母完全不变
- 分子因为 vocab 合并后 embedding 训练更充分而下降（`▁the` 的 exposure = `The + the + tHe + THE` 之和）

**+0.001% 累计误差从哪来？**

Mikeapedia 的 casefold pipeline 在 50K val docs 上测出 +1,784 bytes / 151,082,429 = +0.001% 的分母误差，来源有两类（非 ASCII）：

| 来源 | 例子 | 字节变化 |
|---|---|---|
| NFKC 分解 | `"½"` (2B `\xc2\xbd`) → `"1⁄2"` (5B) | +3 bytes |
| Turkish İ | `"İ"` (2B `\xc4\xb0`) → `.lower()` 变 `"i\u0307"` (3B, i + 组合点) | +1 byte |

实现处理方式：`_validate_lut_bytes()` 启动时对比 `sum(LUT)` vs `shard_header_ground_truth`，差 >1% abort。+0.001% 属于可接受范围。

### 7.4 CaseOps 的算法

```
原文: "The cat"                                    # 7 bytes
encode: "<TITLE>the cat"                           # 在内部表示里多了控制字符
tokenize: [<TITLE>, ▁the, ▁cat]                    # 3 tokens

预生成 byte sidecar (uint16 per token position)：
  sidecar[<TITLE>] = 0           # 控制 token 本身不拥有原文字节
  sidecar[▁the]    = 4           # 对应原文 " The" = 4 bytes (空格+3)
  sidecar[▁cat]    = 3           # 对应原文 " cat" 对应的原文 bytes ...

Eval 时：
  numerator   = NLL(<TITLE>) + NLL(▁the) + NLL(▁cat)
  denominator = sum(sidecar) × ln(2) = 7 × ln(2)   # ← 严格等于原文
```

**两个关键设计：**
1. **Byte sidecar 从 PR #1729 (romeerp) 首创**，被社区引用为合规先例。`.bin` 文件旁边配一个 `_bytes.bin`，存每个 token 位置对应的原文字节数。eval 读 sidecar 求和，不从 tokenizer piece 计算。
2. **控制 token 分得 0 bytes** 是一种约定。另一种约定是把大写信息的 byte 成本"摊"到下一个字母 token 上。哪种都可以，只要 `sum(sidecar) == 原文字节数` 严格成立。

**Trade-off vs Casefold：**
- Casefold: 分子 2 个 NLL + 分母有 +0.001% 误差
- CaseOps: 分子 3 个 NLL（多了 `<TITLE>` 的 NLL）+ 分母严格相等

CaseOps 的控制 token 在 training 时其实也很 predictable（句首、行首规律性强），所以 `NLL(<TITLE>)` 通常不大，但确实是一个真实的 cost。这解释了为什么两条路线的 BPB 差距不大，且依赖于 TTT 栈决定。

### 7.5 Gravity 的算法

```
原文: "The cat"                            # 7 bytes
不做预处理
tokenize: [▁The, ▁cat]                     # 按 gravity vocab 分段
  - 但 vocab 和标准 SP 不同：low-leverage token 被 high-leverage 替换
  - 可能切出 [▁The, ▁c, at] 之类更细的分段（如果 ▁cat 被删了）

Eval 时：
  numerator   = sum(NLL over new segmentation)
  denominator = 7 × ln(2)                  # 仍是原文字节
```

Gravity 不改文本，所以分母定义清晰。但**压缩率下降**意味着 token 数增多（2.45 → 1.05 bytes/token，约 2.3×），分子项数也增多。

**Gravity 之所以能赢**，是因为：每个新 token 虽然覆盖更少字节，但它的 **per-token NLL 大幅下降**——leverage-selected token 对模型来说更"可预测"。净效应：分子项数增加 × 每项更小 = 总分子仍比 baseline 小。

### 7.6 三路对比总表

| 维度 | Baseline SP8192 | Casefold V2 | CaseOps | Gravity |
|---|---|---|---|---|
| 文本变换 | 无 | `NFKC().lower()`（有损） | `encode_lossless_caps()`（无损双射） | 无 |
| Token 数（151MB val） | 40.5M | 36.3M (-10%) | ~42M (+4%) | ~62M (+53%) |
| 分子项数 | baseline | -10% | +4% | +53% |
| 分子每项均值 | baseline | 略降（合并后 embedding 更充分） | 略降 + 控制 token | **大幅降**（leverage selected） |
| 分母字节来源 | piece LUT | piece LUT（lowercase piece） | byte sidecar | piece LUT |
| 分母误差 | 0 | +0.001% | 0（严格） | 0 |
| 合规依据 | N/A | Issue #43, #897 + LUT 自洽 | Issue #1017 + PR #1729 先例 | 无额外机制需要 |

**一句话总结：** 三种路线都保证分母 ≈ 原文字节；差异在于各自牺牲什么来换分子的下降。Casefold 牺牲大小写信息；CaseOps 多花几个控制 token；Gravity 牺牲压缩率换 per-token 可预测性。

---

## 8. 关键 PR 索引

每个 PR 有对应的 records 文件夹，通过 `git fetch origin pull/<PR>/head:pr-<PR>` 可重现 worktree。

| PR | Records 文件夹名 | 内容亮点 |
|---|---|---|
| #755 | `2026-03-25_GravityTokenizer_AblationLeverage/` | Gravity tokenizer 原始实现；`gravity_beta_1.0.model`；ablation leverage scoring 源码 |
| #1578 | `2026-04-12_CustomCasefoldTokenizer/` | 原始 casefold tokenizer 创始 PR (mikeapedia) |
| #1585 | `2026-04-13_SystemsOpt_CasefoldTokenizer/` | **CASEFOLD_TOKENIZER.md 652 行必读**；`tokenizer_pipeline/` 全套脚本；`verify_bytes.py` |
| #1670 | `2026-04-16_CasefoldV4_MultiPhaseGlobalSGD_PhasedTTT/` | Casefold V4 + Phased TTT 初版 |
| #1693 | `2026-04-17_CasefoldV4_AttnOutGate_PhasedTTT/` | +AttnOutGate +SmearGate，当前 casefold 路线最佳 1.0573 |
| #1707 | `2026-04-17_LowercaseTokenization_107399BPB/` | 最简路线 `train_gpt.py` (522 行)，含 FreqGPTQ 实现 |
| #1729 | `2026-04-18_PR1626_CaseOps_Taper/` | CaseOps 合规先例，byte sidecar，`lossless_caps.py`，HF 数据下载脚本 |
| #1735 | `2026-04-18_SP8192_ParallelPreQuantTTT/` | Pre-Quant TTT 原始实现 |
| #1736 | `2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/` | `lossless_caps.py` (833 行, v1-v7)，`prepare_caseops_data.py` |
| #1738 | `2026-04-19_SP8192_PreQuantTTT_CaseOps_V15/` | 当前 CaseOps SOTA 1.0354，byte sidecar 集成 |

---

## 附录：修订历史

- **2026-04-20**：初版，基于 9 个 PR worktree 源码检查
- **2026-04-21**：拆分自原 `tokenizer_techniques_deep_dive-2026-04-20.md`；本文只保留技术实现和 BPB 算法；思考层迁移到 `how-to-think-about-them-2026-04-20.md`
