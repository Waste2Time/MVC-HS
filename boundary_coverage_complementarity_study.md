# Boundary Coverage Complementarity Study

## 1. Study 目标

这套 phenomenon study 的目的不是证明：

> 困难样本挖掘一定显著提升最终 ACC / NMI / ARI。

而是证明一个更容易稳定成立、也更适合包装“对所有样本做线搜索式困难样本生成”的故事：

> **Hardest-only mining 会把生成监督集中在少数最难边界区域，导致边界覆盖不完整；而更广泛的 anchor 来源，尤其是 stratified all-anchor generation，可以在相同 anchor budget 下产生更完整、更均衡、更贴合 prototype 决策边界的 hard samples。**

换句话说，这个 study 要证明的不是：

> 所有样本本身都是困难样本。

而是：

> **所有样本都有可能通过几何式生成贡献边界外延信息。**

---

## 2. 故事主线

现有困难样本挖掘方法通常只关注当前损失较大或 margin 较低的一小部分 hardest samples。  
这种做法虽然可以找到局部最难样本，但容易导致训练信号集中在少数边界段，无法完整刻画类间边界结构。

困难样本生成方法的另一种解释是：

> 不是只使用当前已经困难的样本，而是让更多普通样本也通过几何变换生成有训练价值的边界样本。

因此，本文的 phenomenon study 从“边界覆盖”的角度重新解释 broad-anchor hard generation 的必要性。

---

## 3. 核心假设

本 study 基于三个假设：

### H1. Hardest-only anchors 覆盖局部但不完整

低 margin / 高 loss 样本通常集中在最模糊的边界区域。  
只使用这些样本作为 anchors，会导致生成的 hard samples 高度集中在少数局部边界段。

### H2. Medium / easy / tail anchors 能补充边界外延

虽然 medium 或 easy samples 本身并不困难，但它们携带类内径向结构和类肩部信息。  
经过 line-search / rotation-style 几何生成后，它们可以贡献额外的边界监督。

### H3. Broad-anchor generation 的价值在于边界覆盖，而不一定直接体现为全局指标暴涨

因此，该 study 的主指标应是：

- boundary segment coverage
- boundary coverage entropy
- boundary adherence
- generated sample validity
- anchor source diversity

而不是只看 ACC / NMI / ARI。

---

## 4. Synthetic 数据设计

### 4.1 数据结构

构造三类二维 Student-t 分布：

- Cluster 1：左侧
- Cluster 2：右侧
- Cluster 3：上方

建议中心：

```text
C1 = (-2.0, 0.0)
C2 = ( 2.0, 0.0)
C3 = ( 0.0, 2.8)
```

每类采样 300~500 个样本。

使用 Student-t 分布，而不是高斯分布：

```text
df = 3 或 4
```

原因：

- t 分布有重尾；
- 能自然产生类尾部样本；
- 更适合构造“easy 本身不难，但可生成边界外延信息”的故事。

---

### 4.2 人为设置三种边界难度

故意让三组类间边界难度不同：

| Boundary Pair | 设计难度 | 目的 |
|---|---|---|
| C1-C2 | 最难 | 让 hardest anchors 主要集中在这里 |
| C1-C3 | 中等难 | 由 medium anchors 补充覆盖 |
| C2-C3 | 轻度难 | 由 broader anchors / tail anchors 补充覆盖 |

这样，Only-H 会天然覆盖最难边界，但覆盖不完整；  
M+H 和 Stratified-All 会覆盖更多边界段。

---

### 4.3 双视图设计

#### View 1
对底层语义空间做线性变换 + 小噪声：

```text
X1 = A1 * S + noise
```

#### View 2
对底层语义空间做另一组线性或弱非线性变换。  
在最难边界 C1-C2 附近加入更明显的局部扰动；  
在 C1-C3 / C2-C3 附近加入较弱扰动。

这样可以人为制造：

- hardest boundary region
- medium boundary region
- mild boundary region

---

## 5. Anchor 分桶

对训练后的初始 latent 或 warm-up latent 计算 clustering margin：

```math
m_i = q_{i,(1)} - q_{i,(2)}
```

按 margin 将样本分成三桶：

### H bucket: hard anchors
最低 margin 的 25%。

### M bucket: medium anchors
margin 位于中间区间的样本，例如 25%~60%。

### E/T bucket: easy / tail anchors
高 margin 样本，或者类内尾部样本。

其中 tail anchors 可以通过到所属 prototype 的径向距离定义：

```math
r_i = ||z_i - p_{c_i}||
```

类内距离较大的 high-margin 样本可视为 tail-easy anchors。

---

## 6. 对比策略

本 study 建议至少比较四种策略。

### 6.1 Only-H

只从 H bucket 中选 anchors。

```text
anchor budget = B
source = hard bucket only
```

预期：

- boundary adherence 高；
- 但覆盖集中；
- boundary coverage entropy 低；
- source diversity 低。

---

### 6.2 M+H

从 M bucket 和 H bucket 中选 anchors。

```text
anchor budget = B
source = medium + hard
```

预期：

- 覆盖比 Only-H 更完整；
- 仍偏向边界区域；
- 可能已经接近 All 的覆盖。

---

### 6.3 Stratified-All

这是最关键的策略。

在相同 anchor budget 下，分层采样：

```text
1/3 from H bucket
1/3 from M bucket
1/3 from E/T bucket
```

总 anchor 数量与 Only-H 一样。

这个策略的意义是排除质疑：

> All 只是因为 anchor 更多，所以覆盖更完整。

通过相同 budget 的 stratified sampling，可以证明：

> 覆盖提升来自 anchor 来源结构，而不是数量优势。

---

### 6.4 Random-budget

随机选取与 Only-H 相同数量的 anchors。

```text
anchor budget = B
source = random
```

作用：

- 作为负对照；
- 用于证明不是“随便选更多/随机选一些样本”就能得到结构化边界覆盖。

---

## 7. Line-search / Generation 机制

可采用类似中文硕士论文或几何式困难样本生成的思路。

对于每个 anchor `z_i`，找到当前 prototype `p_c1` 和竞争 prototype `p_c2`。

定义竞争方向：

```math
u_i = (p_{c_2} - p_{c_1}) / ||p_{c_2} - p_{c_1}||
```

沿该方向寻找 prototype decision boundary 附近的点：

```math
z_i(alpha) = z_i + alpha u_i
```

然后构造：

```math
z_i^+ = z_i + (alpha^* - epsilon) u_i
z_i^- = z_i + (alpha^* + epsilon) u_i
```

其中：

- `z_i^+`：边界内侧 hard positive
- `z_i^-`：边界外侧 hard negative

phenomenon study 的重点不在于证明该生成方式最优，而在于分析不同 anchor 来源导致的边界覆盖差异。

---

## 8. 主指标设计

不要把 ACC / NMI / ARI 放成主指标。  
主指标要围绕“边界覆盖互补性”设计。

---

### 8.1 Boundary Segment Coverage

将 prototype decision boundary 划分成若干 segment / bins。

例如：

- 按 prototype pair 划分：C1-C2, C1-C3, C2-C3
- 每个 pair 内再按位置或角度划分若干 bins

如果生成样本落入某个边界 bin 附近，则认为该 bin 被覆盖。

定义：

```math
Coverage_seg = #covered boundary bins / #all boundary bins
```

### 预期结果

| Policy | Segment Coverage |
|---|---|
| Only-H | 低 |
| M+H | 中 |
| Stratified-All | 最高 |
| Random-budget | 中等或不稳定 |

---

### 8.2 Boundary Coverage Entropy

统计生成样本在 boundary bins 上的分布：

```math
H = - sum_b p_b log p_b
```

归一化到 `[0,1]`。

### 预期结果

- Only-H：低熵，集中在最难边界；
- M+H：中等熵；
- Stratified-All：最高熵；
- Random-budget：可能较高，但需要结合 adherence 判断。

---

### 8.3 Boundary Adherence

计算生成样本到最近 prototype decision boundary 的平均距离：

```math
D_bd = mean_i dist(z_i^g, B)
```

越小表示生成样本越贴近决策边界。

### 预期结果

| Policy | Boundary Adherence |
|---|---|
| Only-H | 好 |
| M+H | 好 |
| Stratified-All | 好 |
| Random-budget | 较差或波动大 |

这个指标用于区分：

- 结构性覆盖；
- 随机散布。

Random 可能覆盖广，但不一定贴边界。

---

### 8.4 Generated Sample Validity

生成点应同时满足：

1. 靠近 prototype decision boundary；
2. 不处于极低密度区域。

定义：

```math
Validity = 1[D_bd < tau_b] * 1[rho(z_g) > tau_rho]
```

也可以统计平均 validity rate。

### 预期结果

- Only-H：validity 高，但 coverage 低；
- Stratified-All：validity 高，coverage 高；
- Random-budget：coverage 可能不低，但 validity 较低。

---

### 8.5 Anchor Source Diversity

统计 anchors 来自不同桶的比例：

- H bucket
- M bucket
- E/T bucket

计算来源熵：

```math
H_src = - sum_s p_s log p_s
```

### 预期结果

- Only-H：最低；
- M+H：中等；
- Stratified-All：最高；
- Random-budget：可能较高，但不保证 boundary adherence。

---

### 8.6 Coverage Gini / Concentration

为了证明 Only-H 生成点集中在少数边界段，可以计算 boundary bin 分布的 Gini coefficient。

### 预期结果

- Only-H：Gini 高，说明高度集中；
- Stratified-All：Gini 低，说明覆盖均衡。

---

## 9. 辅助指标

这些指标可以放在附表或补充分析中。

### 9.1 Global clustering metrics

- ACC
- NMI
- ARI

作用：说明生成策略没有破坏全局聚类结构。

### 9.2 Boundary clustering metrics

- Boundary ACC
- Boundary NMI
- Boundary ARI

可以报告，但不要作为主故事核心。  
如果提升明显，可以作为额外正结果；如果没有提升，也不会影响主 story。

### 9.3 Boundary margin

```math
m_i = q_{i,(1)} - q_{i,(2)}
```

可用于说明边界可分性是否改善。  
但如果 baseline 已经 margin 饱和，不要强行用它支撑故事。

---

## 10. 主图设计

### Figure: Boundary Coverage Complementarity

建议画四个子图：

1. Only-H
2. M+H
3. Stratified-All
4. Random-budget

每个子图包括：

- 灰色点：所有样本
- 红色点：oracle boundary samples
- 蓝圈：generated hard samples
- 星号：prototypes
- 虚线或浅色曲线：prototype decision boundary

### 理想视觉效果

#### Only-H
蓝圈集中在最难的一条或一小段边界附近。

#### M+H
蓝圈开始覆盖更多边界段。

#### Stratified-All
蓝圈形成更完整、更均衡的边界壳层。

#### Random-budget
蓝圈可能分布较广，但部分偏离边界，结构性不如 Stratified-All。

---

## 11. 主表设计

建议主表只放机制指标。

| Policy | Seg. Cov. ↑ | Cov. Entropy ↑ | Boundary Adherence ↓ | Validity ↑ | Source Diversity ↑ | Gini ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Only-H | low | low | good | high | low | high |
| M+H | mid | mid | good | high | mid | mid |
| Stratified-All | high | high | good | high | high | low |
| Random-budget | mid/high | mid/high | worse | mid | high | mid |

其中最想要的结论是：

> Stratified-All 在 coverage 和 balance 上优于 Only-H，同时在 adherence / validity 上优于 Random-budget。

---

## 12. 预期现象

### 12.1 Only-H 的 hard samples 很准但窄

Only-H 生成样本通常贴近最难边界，因此 boundary adherence 可以很好。  
但其覆盖集中，segment coverage 和 entropy 较低。

解释：

> Only-H 能强化最难局部边界，但无法完整刻画类间边界结构。

---

### 12.2 M+H 提供中等互补

M+H 引入 medium anchors 后，覆盖范围明显扩大。  
它能覆盖一些 Only-H 忽略的中等边界段。

解释：

> Medium samples 虽然不是 hardest，但能补充 hardest-only mining 的局部性缺陷。

---

### 12.3 Stratified-All 形成完整边界壳层

Stratified-All 在相同 anchor budget 下，通过分层采样覆盖 hard、medium、easy/tail 区域。

理想现象：

- segment coverage 最高；
- coverage entropy 最高；
- source diversity 最高；
- boundary adherence 仍然较好；
- validity 不低。

解释：

> Easy / tail samples 自身不是困难样本，但通过几何式生成可以贡献边界外延监督。

---

### 12.4 Random-budget 缺乏结构性

Random-budget 可能 source diversity 高，也可能覆盖一些边界段。  
但它的 boundary adherence 或 validity 不如 Stratified-All。

解释：

> 不是随机多选 anchors 就能形成有效边界监督；anchor 来源需要和类结构、边界结构相关。

---

## 13. 论文中可以使用的结论表述

### English

> The phenomenon study shows that hardest-only mining provides highly localized supervision around the most ambiguous boundary regions, but fails to cover the full boundary geometry. In contrast, broad-anchor generation, especially stratified all-anchor generation, produces a more complete and balanced boundary shell while maintaining high boundary adherence. This suggests that easy and medium anchors are not useless; although they are not hard by themselves, they can generate informative boundary samples through geometry-aware transformation.

### 中文

> 现象研究表明，仅基于 hardest anchors 的困难样本挖掘会将监督信号集中在少数最模糊的边界区域，难以完整覆盖类间边界结构。相比之下，基于更广泛 anchor 来源的生成策略，尤其是分层 all-anchor 生成，能够形成更完整且更均衡的边界壳层，同时保持较高的边界贴合度。这说明 easy 和 medium samples 并非无用；它们自身虽然不是困难样本，但通过几何式生成可以贡献有效的边界监督。

---

## 14. 成功标准

这套 study 的成功标准不是 Boundary ARI 大幅提升，而是以下现象成立：

1. Only-H 的 segment coverage / entropy 明显低于 Stratified-All；
2. Stratified-All 的 source diversity 明显高于 Only-H；
3. Stratified-All 的 boundary adherence 明显优于 Random-budget；
4. Stratified-All 的 validity 不低于 Only-H 太多，最好接近；
5. 可视化中 Stratified-All 形成更完整的 boundary shell。

只要这些成立，就可以支撑完整故事：

> Hardest-only mining suffers from incomplete boundary coverage; broad-anchor generation complements this limitation by using non-hard anchors to generate boundary-relevant hard samples.

---

## 15. 实验注意事项

### 15.1 不要让 baseline 过强

如果 boundary margin 已经接近 1，说明数据太容易或 assignment 太尖锐。  
这会让性能指标无法体现差异。

可以调整：

- 提高 softmax temperature；
- 增大簇重叠；
- 增大 Student-t 重尾；
- 增强边界扰动；
- 降低训练 epoch；
- 减弱 compactness loss。

### 15.2 控制 anchor budget

Only-H、Stratified-All、Random-budget 必须使用相同 anchor 数量。  
否则会被质疑 All 只是因为样本更多。

### 15.3 主结论不要依赖 ACC

ACC / NMI / ARI 可以作为辅助结果。  
主结论应依赖边界覆盖、边界贴合、有效性和来源多样性。

### 15.4 Random 对照必须存在

Random-budget 用于排除：

> 只要 anchor 多样化就能成功。

如果 Stratified-All 同时做到高 coverage 和高 adherence，就能证明其结构性优势。

---

## 16. 最终一句话

这套 phenomenon study 的核心包装是：

> **不是所有样本本身都是困难样本，而是所有样本都可能通过几何式生成贡献边界外延信息；hardest-only mining 只强化局部最难边界，而 stratified all-anchor generation 能补全更完整、更均衡、更有效的边界监督。**
