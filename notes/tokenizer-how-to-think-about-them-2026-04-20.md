# Tokenizer Techniques: How to Think About Them

本文档是 `what-techniques-exist-2026-04-20.md` 的思考层配对文档。前者记录每条 tokenizer 技术路线的实现细节与 BPB 算法；本文档讨论这些技术**为什么**有效、**何时**该选哪条路线、以及**下一步**应该做什么实验。

相关文档：
- `what-techniques-exist-2026-04-20.md` — 技术实现与 BPB 算法
- `eda-findings-tokenizer.md` — per-token BPB 分析
- `scylla-v2-vs-sp8192.md` — Scylla tokenizer 在 BPB 指标下的 dead-end 诊断

目录：
- §1 Meta-Insight：Tokenizer gain 来自三个机制
- §2 Meta-Insight：BPB 对纯压缩免疫，不对预算重分配免疫
- §3 Meta-Insight：Sub-1-epoch regime 下 tokenizer 是 curriculum，不是编码器
- §4 Meta-Insight：Augment, don't replace
- §5 合法 merge 的原则
- §6 对 EDA 难预测 token 的应对
- §7 从 EDA 到下一个 tokenizer 战场
- §8 未被探索的组合
- §9 对 Tim 的实施建议（按 ROI 排序）

---

## 1. Meta-Insight：Tokenizer gain 来自三个机制

Tokenizer 改进不是单一的"压缩率提升"。每条路线的收益都可以拆成三个独立机制的组合，**不同路线只是在押注不同项**：

| 机制 | 描述 | 代表路线 |
|---|---|---|
| **Coverage gain** | 同样 N 个 token 的 context window 能覆盖更多原文字节 | `SP8192 → SP16384` / `.casefold()` 合并重复词带来的压缩 |
| **Parameter density gain** | 去掉 `The/the/THE` 这类重复 embedding row，让 vocab table 更"密"——每个 row 的训练 exposure 更充分 | Casefold、CaseOps |
| **Segmentation disambiguation** | 减少 `▁perman` / `▁pict` 这种高 NLL fragment 的出现，让模型少做"猜下一个后缀"的工作 | BPB-scored refill、Gravity leverage |

这个分解解释了一些看似矛盾的观察：
- **为什么 `SP8192 → SP10240` 单独不太值？** 它只买到 coverage gain，但 embedding 参数预算线性增长
- **为什么 casefold 在相同 vocab 下就值？** 它同时买到 coverage gain（token 数 -10%）和 parameter density gain（去掉 21% 重复 row）
- **为什么 Gravity 压缩率 -57% 还能赢？** 它牺牲 coverage gain 但 segmentation disambiguation 足够大
- **为什么 SP10240 + casefold 值？** 三个机制叠加：coverage + density + 多出来的 2048 slots 能再买一点 disambiguation

把新想法套进这三维后，一眼能看出它是一个好主意还是重复押注同一个机制。

---

## 2. Meta-Insight：BPB 对纯压缩免疫，不对预算重分配免疫

mikeapedia CASEFOLD_TOKENIZER.md §3 里有一句被严重低估的观察：

> "BPB 对压缩天然免疫——压缩 10% 同时 val_loss/token 升 10%，完全相消。"

推广到一条总原则：

> **如果 tokenizer 只是把文本压得更短，但没有让模型更好地利用固定训练预算，那么 BPB 理论上会近似相消。真正有效的 tokenizer 改动，都是在重分配有限的训练曝光和有限的 embedding rows。**

这解释了 §1 为什么叫"机制"而不是"压缩率变化"：
- Casefold 真正的收益不是"压缩了 10%"，而是"同样 embedding rows 里每行拿到更多 exposure"
- BPB refill 真正的收益不是"加了 374 个 token"，而是"让模型少看几亿个高熵 fragment token"
- FreqGPTQ 真正的收益不是"量化更准"，而是"把有限的 bit 预算倾斜到高影响力 token"

推论：**"让这件事在 10 分钟训练内更可学" 应该是 tokenizer 改动的判断标准，而不是 "文本压缩了多少"。**

---

## 3. Meta-Insight：Sub-1-epoch regime 下 tokenizer 是 curriculum，不是编码器

Parameter-golf 不是长训练场景。10 分钟 / 16MB artifact / sub-1-epoch 的设置下，tokenizer 的角色不是"长期最优编码"，而是：

> **在有限训练步数中，决定模型先学到什么、什么单位拿到足够 exposure、哪些步数被浪费在 byte-gas / fragment / case duplicates 上的 curriculum。**

这个视角能解释为什么几个看似合理的方向是 dead-end：

| 方向 | 在长训练下的直觉判断 | Sub-1-epoch 下的实际结果 |
|---|---|---|
| Full vocab replacement | "我的 vocab 更合理，早晚会学好" | +1.6% BPB——没时间让模型重新建立新 embedding 几何 |
| BPE-dropout | "增加 tokenization 多样性 = 正则化" | 亏了，因为每 dropout 一次就浪费一次稀缺的训练步 |
| Capcode 细粒度修饰符 | "信息无损，模型能学到" | 没时间学，而且每条文本多出几个 token 的成本立刻生效 |

反过来，casefold / CaseOps / Gravity / BPB refill 都是在优化"什么先被学到"：
- Casefold：让模型把 `The` 和 `the` 当同一件事，省下学这种冗余的步数
- Gravity：让 vocab 里只剩下值得模型花深度去处理的 token
- BPB refill：让分词器直接提供完整词，模型不用学"怎么拼 permanent"

---

## 4. Meta-Insight：Augment, don't replace

综合前三点，对 parameter-golf 的 tokenizer 设计有一个操作原则：

> **对小模型 + 短训练预算，tokenizer 改进的主路线不是"重新发明一种语言切分"，而是对现有 SP/BPE 做最小侵入的 augmentation：去重、补洞、精修 byte accounting，不是整体替换。**

这条原则把所有 work 和 dead-end 的路线归到同一个框架：

| 类型 | 路线 | 为什么 |
|---|---|---|
| **Augment (work)** | Casefold V2 (8 步 pipeline 全都是微修正) | 只改 case 归一和 374 个 slot 的补洞 |
| Augment (work) | CaseOps | 加 4 个控制 token，其余不变 |
| Augment (work) | Punctuation swap | 换 25 个低使用 token 为裸 ASCII 标点 |
| Augment (work) | BPB refill | 只填被 clean 释放的 374 slot |
| Augment (work) | Gravity (on SP1024) | 替换 86% 但保持 SP 接口和结构 |
| **Replace (dead-end)** | Full vocab replacement | +1.6% BPB，embedding 几何破坏 |
| Replace (dead-end) | Length-MAX scoring (Dong&Su) | 从零重设计 scoring，BPE 在 <20K vocab 下胜出 |
| Replace (dead-end) | Capcode 修饰符 | 改变 token 分布的整体形状，100× 惩罚 |

**future-you 的自检：**遇到新 tokenizer 想法时，第一个问题应该是：
- 这是在 augment 现有 tokenizer 的一个局部缺陷？
- 还是在替换它的整体逻辑？

后者在本竞赛预算下几乎必然失败。

---

## 5. 合法 merge 的原则

Casefold 告诉我们 "The / the / THE 合并可以让 BPB 降 -0.01"。自然的下一个问题：
- 为什么不把 `cow` 和 `ox` 也合并？
- 为什么不把 `run / ran / running / runs` 合并？
- 为什么不把 `color / colour` 合并？

**答：全部不行。**这一节先用一个层次论识别哪些 merge 思路值得花精力评估，再用三个必要条件筛选具体的候选，最后用两个被证实的 dead-end 背书。

### 5.0 第零问：你在 BPE 的哪一层补全？

任何 tokenizer 改造都应该先回答：你对 BPE 做的修改，是在补 BPE **看不见的维度**、修它的**贪婪错误**，还是在**替换它自己能做的事**？

BPE 的 merge 规则是 `merge(A, B) → AB` iff `count(AB)` 是当前最高的 pair。这里 `A`、`B` 必须是**字节级相邻**才能算。所以 BPE 有一个清晰的能力边界：

| BPE **能自动学到** | 为什么 |
|---|---|
| `▁the`, `▁and`, `ing`, `tion` 等高频字符串 | 字符级邻接 + 频率高 |
| `▁permanent` 整词（如果整体频率够高） | 同上 |
| `'s`, `20`, `00` 等短片段 | 原则上能学，但可能被贪婪顺序错过 |

| BPE **学不到** | 为什么 |
|---|---|
| `The ≡ the` | 两个不同的字节序列，BPE 没有"规范化"概念 |
| `color ≡ colour` | 字节序列完全不同 |
| `½ ≡ 1⁄2` | 不同 codepoint |
| `"  " ≡ " "` | 空格数不同（BPE 能学多空格 token，但学不到等价性） |

**BPE 的本质：只在原始字节流上做局部贪婪合并。任何需要"跨字节等价类"的知识，BPE 都学不到。**

基于这个能力边界，tokenizer 改造可以分成三层：

| 层次 | 你在做什么 | 例子 | 成功率 |
|---|---|---|---|
| **Layer 1: Cross-byte equivalence** | 注入 BPE 完全看不到的等价关系 | Casefold, NFKC, whitespace 归一, CaseOps | **高** — BPE 根本没有这个能力，你是在补强 |
| **Layer 2: Greedy-approximation correction** | 用全局 scoring 修正 BPE 的贪婪错误 | BPB-scored refill, Gravity leverage scoring, non-uniform Viterbi | **中** — BPE 原则上能做，但被贪婪顺序卡住；用 DP / global scoring 能改进 |
| **Layer 3: Replacing BPE's own competence** | 手动做 BPE 能做的事 | 手动 suffix markers (`<ING>`/`<S>`/`<ED>`), Capcode, full vocab replacement | **近必然死** — 在和一个已经 work 的系统竞争，且失去它的 adaptive 性质 |

**判据：只有 Layer 1 和 Layer 2 的想法才值得继续用 §5.1 的三个必要条件评估。Layer 3 的想法直接放弃。**

举几个当前路线的归类：

| 路线 | 所在 layer | 为什么 |
|---|---|---|
| Casefold V2 (预处理) | Layer 1 | 注入 `The ≡ the` 等价，BPE 看不见 |
| CaseOps | Layer 1 | 注入 case 的双射等价 |
| BPB-scored refill | Layer 2 | BPE 贪婪选 merge 时会错过 `▁permanent` 这种全局更优的 token，refill 用 DP 补回来 |
| Gravity leverage | Layer 2 | BPE 用 frequency 做 scoring；Gravity 用 leverage 做 global re-scoring |
| Non-uniform Viterbi | Layer 2 | BPE 分段是贪婪的；Viterbi 是全局最优 |
| Manual suffix merge (`<ING>`) | **Layer 3** | `-ing` 是字符级高频 pattern，BPE 已经自动在学了；你手动注入 marker 只是替换 BPE 的 adaptive 决策 |
| Capcode 修饰符 | **Layer 3** | 单字母大小写 pattern 其实也有字符级规律，硬注入 `<CAP>` 替换 BPE 决策 |
| Full vocab replacement | **Layer 3** | 整个丢掉 BPE，自己从零造 |

这三个 Layer 3 的例子都是 documented dead-end，这不是巧合。

### 5.1 合法 merge 的三个必要条件

一个 token-level merge `A, B, C → M` 要想在 BPB 指标下 work，必须同时满足：

1. **Eval-time 可还原**（bijection 或可接受近似）
   - 要么 `decode(encode(s)) == s` 严格成立（CaseOps 路线）
   - 要么归一化规则是众所周知的通用规则，且字节误差 << 0.01%（Casefold 路线，依赖 "lowercase 可逆性" 的隐含共识——严格说 casefold 并不可逆，但 Issue #43 裁定 NFKC+lowercase 属于合规 tokenizer 预处理）
2. **Embedding 语义上应该重合**
   - `The / the` 在 ideal embedding space 应该几乎重合——大小写是表面特征，不改变词义
   - `cow / ox` 在 embedding space 必须分开——它们是不同的实体
   - 违反这一条，embedding 几何会被破坏（见 §5.3 dead-end #2）
3. **合并后的 exposure gain 不被"恢复 marker"的 cost 吃掉**
   - `The → <TITLE>the` 引入 `<TITLE>`，但 `<TITLE>` 频率远低于"大写字母"的频率（因为大部分词本来就小写）
   - `running → run<ING>` 引入 `<ING>`，但英语 `-ing` 极其高频 → `<ING>` token 几乎每隔几个词就出现一次 → 分子爆炸（见 §5.3 dead-end #1）

### 5.2 一些候选 merge 的合法性判定

| 候选 merge | Layer (§5.0) | 条件 1 (可还原) | 条件 2 (语义等价) | 条件 3 (marker cost) | 判定 |
|---|---|---|---|---|---|
| `The / the / THE` | 1 | ✓ (casefold / CaseOps) | ✓ 只差 case | ✓ `<TITLE>` 低频 | **合法，已证实有效** |
| `don't / don 't / don' t` | 1 | ✓ (NFKC + 空格归一) | ✓ 只差 whitespace | N/A (无 marker) | 合法 |
| `½ / 1⁄2` | 1 | ✓ (NFKC) | ✓ 数学上相同 | N/A | 合法 |
| `running / run` (+`<ING>`) | **3** | ✓ (双射) | 部分 (时态不同) | **✗ `<ING>` 极高频** | **dead-end（Layer 3：BPE 已在学 `ing`）** |
| `color / colour` | — | ✗ (无 bijection rule) | ✓ 语义相同 | N/A | 不合法（字节数不同且无可逆映射） |
| `cow / ox` | — | ✗ (无 marker 可恢复) | ✗ 不同实体 | N/A | **不合法** |
| `run / ran` | — | ✗ | ✗ 时态不同 | N/A | 不合法 |

### 5.3 两个被证实的 dead-end

**Dead-end #1: Capcode / `<CAP>` 修饰符 token** (mikeapedia §8.2)

思路：不做 casefold，而是在每个大写字母前注入 `<CAP>` 控制 token（类似 CaseOps 但更细粒度）。

结果：
- 修饰符 token 新增 ~1.5B 个 token（因为英文大写字母出现太频繁）
- slots 只省 ~15M
- **净 +100× 惩罚** → 分子爆炸

**教训：** 这直接对应 §5.1 条件 3 的违反。引入的 marker 频率必须远低于它"解决"的重复度，否则 ROI 是负的。

**Dead-end #2: Full vocab replacement（自底向上造 vocab）**

思路：丢掉 BPE 训出的 vocab，自己设计一个"更合理"的 vocab。

结果：**+1.6% BPB 退化**。

原因：
- 替换 27-37% 的 token 破坏了 embedding space 的几何结构
- Transformer 的后续 projection matrix 是对 BPE vocab 联合训练的，对新 vocab 失调
- 即使新 vocab 在"直觉上"更合理，实际训练不充分

**教训：** 这对应 §5.1 条件 2。合并/替换只能在 "embedding 本应重合" 的 token 之间做——casefold work 是在**纠正** BPE 的过度区分，而不是**引入**新的跨语义合并。

### 5.4 形态学 merge 的诱惑与陷阱

"如果 The/the 可以合并，为什么 running/run 不行？" 是个自然的问题。这个问题的根本答案在 §5.0——**手动 morphological merge 是 Layer 3，本质上在重复 BPE 已经在做的事**。

**BPE 已经在隐式做 morphological merging：** 当 `-ing` 足够高频时，BPE 的 merge 过程会自动产生 `▁runn + ing` 这样的切分（或者直接 `▁running` 作为单 token，取决于整体频率）。所以"`run + <ING>`" 这种 suffix merging，BPE 已经做了——只是 BPE 的版本是**内生的、频率自适应的**：

- 高频整词（`▁running`, `▁being`, `▁nothing`）→ BPE 直接合并成单 token
- 中频可分解词（`▁jogging` = `▁jogg + ing`）→ BPE 保持 suffix 分离以节省 slot
- 低频词（`▁perpetuating`）→ BPE 可能切得更碎，但这恰恰是因为它不值得占 slot

手动注入 `<ING>` / `<S>` / `<ED>` 控制 token 的问题不是"想法错了"，而是**用硬规则替换了 BPE 的 adaptive 决策**：
- 一旦注入 `<ING>` 就是**强制**的——无论上下文如何，都得走 `stem + <ING>` 这条路
- 高频整词（`▁running`）本来是 1 个 token，变成 `▁runn + <ING>` = 2 个 token
- BPE 原本的 adaptive compression 被破坏

再叠加 §5.1 条件 3 的 marker cost 问题，结果必然是分子爆炸：

英语 suffix 的频率（FineWeb 粗估）：
- `-s`: ~6% of tokens
- `-ing`: ~2%
- `-ed`: ~3%
- `-ly`: ~1.5%

如果把 `runs/running/runned/running` 全合并成 `run + <S>/<ING>/<ED>`，会：
- 省下几百个 vocab slot（`running, runs, runner, runned, ...`）
- 但引入 ~12% 的 val token 是 `<S>/<ING>/<ED>/<LY>`
- val token 数增加 ~12% → 分子同比例爆炸
- 即使 `run` 的 embedding 训练再充分，也弥补不了

**一个可能的例外（Layer 2，不是 Layer 3）：** 如果不做 rule-based suffix injection，而是用 BPB scoring 去**经验性地**找 BPE 贪婪错过的整词——比如合并 `perpetuating / perpetuated / perpetuates` 成完整词放进 vocab——这属于 Layer 2，合法且可能有价值。这些恰好是 EDA 里 `▁perpet` word-fragment 的延伸，mikeapedia 的 Rust `vocab-builder` 框架就是为这类修正设计的。**目前没人做过这种 selective morphological refill。**

**本质区别：**
- Layer 3（死）：我规定 `-ing` 变成 `<ING>`，所有词一视同仁
- Layer 2（活）：我发现 `▁perpetuating` 整词值得进 vocab（因为它的 BPB scoring 高），我加进去；`▁running` 不用我加（BPE 已经自己加了）

### 5.5 原则性的反问清单

下次想到一个"是否可以把 X 和 Y 合并"或"是否应该改 tokenizer"的想法时，按这四个问题自检：

**第零问（层次判定，必答）：**
0. **这个改造是 Layer 1 (cross-byte equivalence)、Layer 2 (greedy correction)、还是 Layer 3 (replacing BPE)？**
   - Layer 1 或 2 → 继续往下
   - Layer 3 → 立即放弃（你在和 BPE 的 adaptive 机制竞争）

**第一至第三问（只对 Layer 1/2 适用）：**
1. **Eval 时能不能把合并后的 token 还原成原文字节？** 能 → 继续；不能 → 放弃
2. **X 和 Y 在理想 embedding space 里是不是应该几乎重合？** 是 → 继续；不是 → 放弃
3. **如果需要引入恢复 marker，marker 的频率是不是远低于被合并词对的频率？** 是 → 继续；否 → 大概率 dead-end

**四个都 yes 才值得做实验。** 现有路线的检查结果：

| 方法 | Q0 | Q1 | Q2 | Q3 | 判定 |
|---|---|---|---|---|---|
| Casefold | Layer 1 | ✓ | ✓ | N/A | **做** |
| CaseOps | Layer 1 | ✓ | ✓ | ✓ | **做** |
| BPB refill | Layer 2 | ✓ | N/A | N/A | **做** |
| Gravity | Layer 2 | ✓ | N/A | N/A | **做** |
| `<ING>` marker | **Layer 3** | — | — | — | 不做 |
| Capcode | **Layer 3** | — | — | — | 不做 |
| Full vocab replace | **Layer 3** | — | — | — | 不做 |
| cow / ox merge | Layer 1 (形式上) | ✗ | ✗ | — | 不做 |

---

## 6. 对 EDA 难预测 token 的应对

对接 `eda-findings-tokenizer.md` §B 的发现：当前 sp8192 baseline 在某些 token 类上 BPB 异常高。把每一类问题和现有 tokenizer 技术对应起来：

### 6.1 问题分类与解法

| EDA 发现 | BPB / NLL | 根因 | 对应解法 | ROI |
|---|---|---|---|---|
| `<0xD2>` / `<0xC6>` 字节碎片 | BPB 6-9 | SP byte fallback 把非 ASCII 切成单字节 | Casefold V2 step 7 (punctuation swap → 9.47% → 2.49%) | 中（对 ASCII 标点有效，对 CJK 不解决） |
| CJK 字符 | BPB 极高 | `character_coverage=0.9995` 把 CJK 丢给 byte fallback | 1) 改成 0.9999（占用英文 slot）<br>2) codepoint-level fallback（自实现） | **低**（FineWeb 里 CJK <0.1%） |
| `▁z`, `J`, `(` 单字符高熵前缀 | BPB 4-5 | BPE 贪婪把常见单字符留作独立 token | Casefold V2 step 2（删 53 L=1 单字节 token） | 中（包含在 casefold pipeline） |
| `▁perman`, `▁pict`, `▁somet` word-fragment | NLL 8-9 | BPE 合并顺序产生"差一步就完整词"的残片 | **BPB-scored refill**（mikeapedia Rust vocab-builder） | **高**（EDA 最大痛点恰好是这个） |
| 标点 BPB (1.97) 是普通词的 1.85× | BPB 高 | 标点后分布极宽（句子是否结束、下一词类型） | 无纯 tokenizer 解 → 靠 depth recurrence / longer context | 间接 |
| 数字 BPB (2.13) 最高 | BPB 最高 | 数字后继值域无限 | 无 tokenizer 解（本质熵高） | 放弃 |
| 首字母大写 BPB (2.09) | BPB 高 | 专有名词需要世界知识 | Casefold 合并后 `▁the ≠ ▁The` 差异消失，`▁J` 的"人名性"被稀释 | 中（casefold 顺带解决） |

### 6.2 零成本改动：BPE → Unigram Viterbi decoding

EDA 没有点出来但很重要的一点：**BPE 是贪婪分段（从左到右最长匹配），Unigram 是 Viterbi 全局最优分段。**

同一个 vocab，Unigram 分段会倾向于把 `▁perman|ent` 这种"本该是一个词"的序列切成 `▁permanent`（如果 `▁permanent` 在 vocab 里），而 BPE 可能因为 merge 顺序切成前者。

Mikeapedia 的 casefold V2 把 model 从 BPE 转成 Unigram 模式——部分动机是"删 token 会破坏 BPE merge chain"，但另一个好处正是这个 Viterbi 红利。

**这是零 vocab 变化、零训练成本的改动。** mikeapedia §8.3 Future Work #3 甚至提到更激进的版本：`score(t) = log(freq/N) / √byte_len`，估计额外 0.3-0.7% BPB 改进。

**Actionable：** 自己造 tokenizer 时，训完 BPE 后导出 `.model` 时把 `model_type=UNIGRAM` 设好，零成本拿到收益。如果用 mikeapedia 的 artifacts，已经是 Unigram 了。

### 6.3 不值得投入的方向

基于 EDA，以下是**已知但不值得做**的方向：

- **给数字加 "digit marker"**：数字 BPB 高是因为后继值域无限，不是 tokenizer 问题。无 tokenizer 解。
- **给专有名词加 "NamedEntity marker"**：需要外部 NER，违反 compliance（training 时不能用 external model 标注），且引入的 marker 会让分子爆炸（参考 §5.3 Capcode dead-end）。
- **把 CJK 全部塞进 vocab**：FineWeb 占比 <0.1%，对 val BPB 贡献 <0.001。占用英文 slot 的 cost 远大于收益。

---

## 7. 从 EDA 到下一个 tokenizer 战场

### 7.1 Hard-token 有"重要性加权"问题

EDA 里按 BPB 降序排出来的 "最难 token" 列表容易让人产生选择偏差。实际应该按 **loss contribution** 排，而不是按 per-token BPB 排：

$$\text{contribution}(t) = \text{frequency}(t) \times \text{avg\_bytes}(t) \times \text{avg\_BPB}(t)$$

按这个指标排，结论截然不同：

| Token 类 | per-token BPB | Frequency | 对总 loss 贡献 | 是否值得修 |
|---|---|---|---|---|
| Word fragment (`▁perman` 等) | 8-9 (NLL) / 高 | 中高 | **高** | **值得** |
| 标点 | 1.97 | 11% | **高** | 间接（非 tokenizer 问题） |
| 首字母大写 | 2.09 | 11% | **高** | 值得（casefold 顺带） |
| 数字 | 2.13 | 4% | 中 | 放弃（本质熵） |
| CJK / 非 ASCII fallback | 6-9 | **<0.1%** | **极低** | 不值得 |
| 单字符 `▁z`, `J` | 4-5 | 低 | 低 | 自动包含在 casefold 中 |

**原则：** 优化目标不应该是"降低 top-BPB token 列表里第一个的 BPB"，而应该是"降低对总 BPB 贡献前 k 大的 token 类"。

### 7.2 下一战场不是 case，是 fragment

Casefold 已经有充分 artifacts 覆盖。但 EDA 显示即使用上 casefold，残余最大痛点其实是 **word fragment ambiguity**：

EDA NLL 排序 top-5 全是 word fragment：
- `▁perman` (9.57) → permanent / permanently / permanence
- `▁pict` (9.11) → picture / pictorial / pictured
- `▁somet` (8.95) → something / sometimes / somewhat
- `▁appar` (8.87) → apparently / apparatus / apparent
- `▁perpet` (8.82) → perpetual / perpetuate / perpetrator

**含义：** 下一阶段 tokenizer 创新不一定继续来自 normalization（case/whitespace/NFKC），而更可能来自 **补全高歧义 fragment**——targeted refill / segmentation scoring。

这是 BPB-scored refill 的教科书目标。mikeapedia 的 Rust `vocab-builder`：
1. 枚举候选（包括当前 fragment 的合法扩展：`▁permanent, ▁permanently, ▁permanence, ...`）
2. 用 DP encoding 评估："如果加入 `▁permanent` 替代 `▁perman + ent`，corpus bits 降多少？"
3. 填入 slot

Casefold V2 pipeline 的 step 5 产出的 top refill token 是 `'s`, `▁1`, `▁2`, `20`, `00`——这说明 BPE 确实漏了一些高价值 token。EDA 发现的 word fragment 是**同一类漏洞的另一种表现**。

**值得投入的新实验方向：**
- **Fragment-driven refill**：不是像 mikeapedia 那样让 DP 自己找 top refill，而是以 EDA NLL 排序 top-100 fragment 为起点，枚举它们的词汇学完整扩展作为候选池
- **Selective morphological refill**：只合并低频长尾形态学变体（见 §5.4）
- **Non-uniform Viterbi scores**：mikeapedia §8.3 Future Work #3，`score(t) = log(freq/N) / √byte_len`，估计 0.3-0.7% BPB 改进

### 7.3 下一个 tokenizer 研究问题

把 §1-§4 的 meta-insights 和 §6-§7.2 的 EDA 结果组合起来，以下是社区没人答过但值得答的问题：

| 问题 | 为什么值得答 | 实施成本 |
|---|---|---|
| Casefold 对 per-token-class NLL 分布的影响是什么？哪类 token 受益最多？ | 回答"casefold 为什么 work"的机制层问题；mikeapedia 自己没做这个消融 | 1-2 天（只需要 eval 现有 artifacts） |
| EDA 里 top-100 word fragment 用 targeted refill 能降多少 BPB？ | 如果能，这是 casefold 之外的一个正交 gain | 3-5 天（需要跑 Rust vocab-builder 的定制版） |
| Gravity leverage scoring 在 casefold 语料上还有区分力吗？在 SP8192 上还 ROI 正吗？ | 回答 Gravity 能否叠加到 casefold 之上 | 4h + retrain |
| Casefold 之后的最优 vocab size 是多少？(SP8192? SP10240? SP12288?) | SP10240 的成功暗示 casefold 后 optimal vocab > 8192 | 1-2 天（sweep 几个 size） |

---

## 8. 未被探索的组合

### 8.1 Gravity + Casefold（理论最优组合）

Gravity 替换 low-leverage token → Casefold 去除 case 重复。两者正交。但需要验证：
- Gravity 的 leverage scoring 在 casefold 语料上是否仍有区分力？
- Casefold 后 case-variant token 不再存在——Gravity 需要在 casefold 后的候选池上重新 scoring

**估计实施成本：** 2-3 天（已有两个 pipeline，需要串联）

**未知风险：** Gravity 在 SP1024 上 work，但 SP8192 下 slot 边际价值低得多，leverage scoring ROI 可能下降（dcrow85 未验证）。

### 8.2 Casefold V2 tokenizer + V4 TTT 栈

V2 tokenizer 有 BPB scoring refill + punctuation swap（技术上更好），V4 有 Multi-Phase SGD TTT（效果更好）。两者从未在同一个 PR 里合并。

**估计实施成本：** 1 天（两边 artifacts 都有 HF 发布）

### 8.3 CaseOps + Gravity

CaseOps 去除 case 重复但保留信息（控制 token）。Gravity 在此基础上替换 low-leverage token。需要先确认：控制 token (`<TITLE>`, `<ALLCAPS>` 等) 的 ablation leverage 是高还是低？如果高，Gravity 会保留它们——这是理想情况。

### 8.4 FreqGPTQ 应用到任何 casefold 路线

nothingLiva 的 FreqGPTQ（top-100 2× Hessian boost + INT8）是零开销改动，理论上可叠加到任何栈。目前只在 SP10240 + vanilla 架构上验证。

### 8.5 Selective morphological merge

只合并低频长尾形态学变体（如 `perpetuating/perpetuated/perpetuates`），避开高频后缀 `-s/-ing/-ed`。需要用 BPB scoring 做 per-suffix ROI 评估。**目前没人做过。**

### 8.6 Byte-weighted pretraining loss

mikeapedia §8.3 Future Work #1 提出：用 byte 数加权训练 loss（长 token 贡献更多字节到 BPB 分母 → 梯度应对齐）。"Implemented but not ablated"——可能已经在某个未提交版本里了。

### 8.7 Non-uniform Viterbi scores

mikeapedia §8.3 Future Work #3 提出：给 Unigram 模型设置 `score(t) = log(freq/N) / √byte_len`。估计 0.3-0.7% BPB 改进。零 vocab 变化——只改 segmentation。**可应用于所有 tokenizer 包括 stock SP。**

### 8.8 Vocab size sweep on casefold

SP8192 对 vanilla BPE 可能是最优的（PR #1334 验证），但 casefold 改变了 compression/model-capacity tradeoff。最优 vocab size 可能需要重新搜索——mikeapedia 的 Rust builder 支持 `--checkpoints` 产出 3K/4K/5K/6K 中间 vocab。SP10240 的成功（PR #1707）暗示 casefold 后的 optimal vocab > 8192。

---

## 9. 对 Tim 的实施建议（按 ROI 排序）

**前提：** 当前 sp8192 baseline @ 1.1146，目标超过 codemath3000 #1585 的 1.0639（真正稳固的 SOTA，排除 AT_RISK 的 #1736）。

### Path 1：最快见效（1-2 天，低风险）

**直接复用 HF artifacts + FreqGPTQ，不造 tokenizer。**

```bash
# 推荐 mikeapedia 的 V2：含 BPB-scored refill，对 EDA 的 word fragment 痛点效果最大
huggingface-cli download Mikeapedia/fineweb10B-sp8192-casefold-v2 --repo-type dataset
huggingface-cli download MissGlitterToken/sp10240_casefold --repo-type dataset
huggingface-cli download romeerp/parameter-golf-caseops-v1 --repo-type dataset
```

工作量：
- 在 baseline 上换数据路径 + tokenizer path
- 加 FreqGPTQ（PR #1707 `train_gpt.py`，几十行）

**预期 BPB：** ~1.08x（无 TTT）→ ~1.07x（加标准 TTT）

**为什么从这开始：** EDA 已经显示 word fragment (`▁perman`, `▁pict`, `▁somet` 等 NLL 8-9) 是最大痛点（§6.1, §7.2），而 mikeapedia 的 V2 artifacts 已经通过 BPB-scored refill 解决了这个问题。不需要自己跑 3.2h Rust pipeline。

### Path 2：中等投入（2-3 天，中等风险）

**Casefold V2 data + Pre-Quant TTT stack（对标 PR #1738, 1.0354）。**

- 从 HF 下载 mikeapedia 数据集
- 集成 PR #1735 的 pre-quant TTT：21 epoch AdamW on val chunks，epoch-level cosine LR (5e-4 → 5e-5)，8-GPU 联邦平均
- 合规优势：eval 时零适应，artifact 是 static int6 模型，不需要争论 score-first TTT 是否合法

**预期 BPB：** ~1.05x（alertcat #1738 的 1.0354 做得到，是 CaseOps；Casefold 数据上对应 ~1.055）

### Path 3：独特贡献（3-5 天，用 EDA 变成可发表的 writeup）

**"Tokenizer 归一化对 per-token NLL 分布的影响" writeup + 可选 Gravity+Casefold 探索。**

- 在现有的 sp8192 baseline 和一个 casefold 版本之间做 per-token-class NLL 对比（用 EDA 框架的现有工具）
  - 这是 mikeapedia 的 CASEFOLD_TOKENIZER.md 里**没有的**分析维度
  - 输出：word fragment / CJK fallback / 标点 / 数字 四类的 NLL shift 细分
- 如果时间允许，在 casefold 语料上跑 Gravity leverage scoring（4h 单卡 + GPT-2），验证 leverage 信号在 casefold 后是否仍有区分力
- 即使最终 BPB 没到 frontier，这个 writeup 的价值：
  1. 是社区里没人做过的 per-class 消融
  2. 可以作为 non-record PR 贡献
  3. 回答了 "为什么 casefold work" 的机制问题（§7.3 第 1 个研究问题）

### 不建议（dead-end 或 negative ROI）

| 不要做 | 原因 | 出处 |
|---|---|---|
| 从零训 casefold tokenizer | HF 上已有完整 artifacts | mikeapedia |
| Capcode / `<CAP>` 每字母修饰符 | 100× marker 惩罚 | §5.3 dead-end #1 |
| 自底向上 full replacement vocab | +1.6% BPB 退化 | §5.3 dead-end #2 |
| 跨语义 merge（cow/ox, color/colour 等） | 破坏 embedding 几何 + 不可逆 | §5.2 |
| 高频 suffix merge（-s/-ing/-ed）| marker 频率过高，分子爆炸 | §5.4 |
| BPE-dropout | sub-1-epoch 下是亏的 | mikeapedia §8.2 |
| 在 #1736 PhasedTTT 合规裁决前投入 | AT_RISK，可能被 ban | Issue #1017 相关 |
| 扩大 vocab 到 12288/16384（不加 casefold） | 边际 -0.003 换线性 embedding 成本 | `eda-findings-tokenizer.md` |
| 走 Scylla tokenizer 路线 | 不强调压缩 = BPB 指标下天然吃亏 | `scylla-v2-vs-sp8192.md` |
| 给数字 / 专有名词加 marker | §6.3 | |

### Decision tree

```
是否有 8-GPU 可用？
├── 否 → Path 1（单卡也能做）
└── 是 →
    是否愿意接受 "artifact = post-TTT static model" 的训练 pipeline 复杂度？
    ├── 否 → Path 1 + 标准 legal TTT
    └── 是 → Path 2 (Pre-Quant TTT，预期 1.05x)

如果 Path 1/2 完成后还有 3+ 天：
    → Path 3（EDA writeup，最低风险最高社区价值）
```

---

## 附录：修订历史

- **2026-04-21**：从原 `tokenizer_techniques_deep_dive-2026-04-20.md` 拆出；本文是思考层，`tokenizer-what-techniques-exist-2026-04-20.md` 是实现层。新增 §1-§4 meta-insights，§7 "从 EDA 到下一个 tokenizer 战场"。
- **2026-04-21 (later)**：§5 重构——新增 §5.0 BPE 能力边界与三层论（Layer 1 cross-byte equivalence / Layer 2 greedy-approximation correction / Layer 3 replacing BPE's competence）。§5.4 形态学 merge 新增"为什么 BPE 已经在做这件事"的解释。§5.5 反问清单加入 Q0 层次判定。
