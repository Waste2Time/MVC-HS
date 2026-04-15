# Phenomenon Study 方案：在合成数据上验证“困难样本挖掘/生成”是否有助于边界强化

## 1. 研究目标

本 phenomenon study 不追求完整主实验指标，而是尝试回答一个更基础的问题：

> 当多视图聚类的主要困难集中在 **簇边界附近的样本** 时，加入困难样本挖掘/生成模块，是否比纯 baseline 更容易提升边界区域的聚类质量？

这里的逻辑是：

- baseline 主要学习整体结构；
- hard sample mining / generation 主要服务于 **边界强化**；
- 因此，在一个“困难主要来自边界”的合成数据上，这类方法应该更容易体现优势。

---

## 2. 核心假设

本实验要验证的现象假设是：

1. 大多数样本位于簇核心区，容易聚类；
2. 少量样本位于类别边界附近，是聚类错误的主要来源；
3. 如果对这些边界样本进行困难样本挖掘/生成并加入训练，则模型应主要改善：
   - 边界区域的聚类指标；
   - 边界样本的分配 margin；
   - 边界样本的跨视图一致性。

因此，本实验不是证明“hard mining 对任何数据都更好”，而是证明：

> **在边界困难主导的数据上，hard sample mining 更有价值。**

---

## 3. 合成数据设计

## 3.1 数据目标

构造一个最小但有代表性的多视图聚类场景，使其同时包含：

- 易样本（core samples）
- 边界样本（boundary samples）
- 跨视图冲突边界样本（boundary-conflict samples）
- 少量伪困难样本（pseudo-hard samples）

---

## 3.2 为什么改用 t 分布

这里不使用高斯分布，而改用 **Student-t 分布** 生成底层样本，原因是：

1. **重尾分布更接近真实数据中的“局部极端样本”现象**  
   高斯簇过于理想，边界和核心分布太“干净”；t 分布会自然产生更多远离簇中心的样本。

2. **更容易生成“边界模糊但非纯噪声”的样本**  
   这类样本更适合用来观察 hard mining 是否真的在强化边界。

3. **更容易出现局部难样本，而不必额外手工加入大量异常值**  
   这样 synthetic 数据会比高斯更自然。

因此，这个 study 更适合用 t 分布来构造“边界困难主导”的数据情景。

---

## 3.3 数据生成方式

### 底层语义空间

先在 2D 语义空间中生成 3 个 **二维 Student-t 簇**：

- Cluster 1 center: `(-2, 0)`
- Cluster 2 center: `(2, 0)`
- Cluster 3 center: `(0, 2.8)`

每类各采样 `N=400`，总样本约 `1200`。

### 采样方式
每个簇的样本可写为：

\[
s_i = \mu_k + L_k \cdot t_\nu
\]

其中：

- \(\mu_k\)：第 \(k\) 个簇中心
- \(L_k\)：控制簇形状的线性变换矩阵
- \(t_\nu\)：自由度为 \(\nu\) 的二维 Student-t 随机变量

### 参数建议
- 自由度 `df = 4` 或 `df = 5`
- `df` 不宜太小，否则重尾过强，容易让 outlier 主导
- `df` 不宜太大，否则又接近高斯

推荐：
- 主实验：`df = 4`
- 稳定性检查：`df = 3, 5, 8`

---

## 3.4 View 1

对底层语义空间施加线性变换并加小噪声：

\[
x_i^1 = A_1 s_i + \epsilon_i^1
\]

其中：

- \(A_1\) 为旋转/缩放矩阵
- \(\epsilon_i^1 \sim \mathcal{N}(0, \sigma_1^2 I)\)

建议：
- `sigma1 = 0.08 ~ 0.12`

---

## 3.5 View 2

对底层语义空间施加另一组线性/弱非线性变换：

\[
x_i^2 = A_2 s_i + \epsilon_i^2
\]

并仅在 **边界带区域** 内加入定向扰动，使部分样本在 view 2 中更偏向竞争簇，从而制造跨视图冲突。

建议操作：

1. 先根据底层语义空间到最近和第二近簇中心的距离差定义边界带；
2. 对边界带样本，以概率 `p_conflict = 0.3 ~ 0.5` 加入定向扰动；
3. 扰动方向沿最易混淆簇对的法向方向。

---

## 3.6 四类样本定义（oracle，仅用于分析，不用于训练）

### A. easy-consistent
- 远离真实边界
- 两视图语义一致
- 非噪声

### B. boundary-consistent
- 位于真实边界带附近
- 但两视图语义仍一致

### C. boundary-conflict
- 位于真实边界带附近
- view 1 与 view 2 对其更偏向不同簇

### D. pseudo-hard
两种方式任选其一或同时保留：

#### D1: outlier pseudo-hard
利用 t 分布天然重尾特性，从簇尾部选取少量密度较低、但不属于边界冲突的样本

#### D2: view-corrupted pseudo-hard
对一小部分样本只在一个视图上加入额外扰动，使其看起来“难”，但并不对应真实边界竞争

---

## 4. 方法对比

## 4.1 Baseline
原始 baseline，不加困难样本挖掘/生成模块。

## 4.2 Baseline + Hard Mining
在 baseline 基础上，加入一版“类似中文硕士论文思路”的困难样本模块，例如：

- 基于当前聚类分配定义困难样本；
- 沿特定方向构造 hard positive / hard negative；
- 将生成的困难样本加入对比学习或边界约束损失。

注意：这版不要求是最终论文方法，只需要是一个“有 hard sample 机制”的增强版。

---

## 5. 实验流程

## 5.1 训练流程
1. 在相同 synthetic 数据上分别训练：
   - Baseline
   - Baseline + Hard Mining
2. 每个方法运行 `5~10` 个随机种子；
3. 固定训练轮数、batch size、学习率等超参数。

---

## 5.2 分析阶段
训练后，固定模型表示，分析以下内容：

- 全体样本聚类指标
- 边界样本聚类指标
- 边界样本的 margin
- 边界样本的跨视图 disagreement
- 困难样本模块是否真正主要命中边界样本

---

## 6. 评价指标

## 6.1 全局聚类指标
- ACC
- NMI
- ARI

## 6.2 边界区域指标（重点）
只在 oracle 边界带样本上计算：
- boundary-ACC
- boundary-NMI
- boundary-ARI

这是本实验最关键的指标。

---

## 6.3 边界判别指标

### Clustering margin
\[
m_i = q_{i,(1)} - q_{i,(2)}
\]

关注边界样本的平均 margin 是否提高。

### Cross-view disagreement
\[
d_i = JS(q_i^1, q_i^2)
\]

关注边界样本的平均 disagreement 是否下降。

---

## 6.4 困难样本命中率
如果 hard module 会选择/生成样本，则统计：

- 被选中样本中，属于 B/C 的比例；
- 被选中样本中，属于 D 的比例。

这可以辅助判断 hard module 到底是在强化边界，还是在放大噪声。

---

## 7. 必做图：二维散点图

本 study 至少要做一张散点图。

## 图名建议
**Latent Scatter Plot with Boundary and Selected Hard Samples**

## 作图内容
在训练后的二维 latent 空间（或 PCA/TSNE/UMAP 到二维）中画出：

- 普通样本：浅色点
- oracle 边界样本（B+C）：高亮颜色
- 被 hard module 选中/生成的样本：特殊 marker（如 `x` 或空心圈）
- prototype：星号

## 这张图希望展示什么
如果 hard mining 真有效，应看到：

1. 被选中/生成的困难样本主要集中在簇边界附近；
2. 相比 baseline，hard 版本的边界过渡区更清晰；
3. hard 版本在边界附近的 prototype 分隔更明显。

---

## 8. 推荐的作图清单

### Figure 1. Synthetic latent scatter
- 展示 A/B/C/D 分布和被选中 hard samples 位置

### Figure 2. Boundary ARI comparison
- Baseline vs Baseline+Hard

### Figure 3. Boundary margin comparison
- Baseline vs Baseline+Hard

### Figure 4. Boundary disagreement comparison
- Baseline vs Baseline+Hard

---

## 9. 期望观察到的现象

如果“困难样本挖掘/生成确实服务于边界强化”，应观察到：

### 现象 1
相比 baseline，加入 hard module 后：
- 全局 ACC/NMI/ARI 可能只有小幅变化；
- 但 boundary-ARI / boundary-NMI 提升更明显。

### 现象 2
边界样本的平均 clustering margin 提高，即：
- hard module 让边界样本的 top-1 与 top-2 竞争关系更清晰。

### 现象 3
边界样本的跨视图 disagreement 下降，即：
- hard module 促进了边界区域的多视图语义对齐。

### 现象 4
被选中/生成的 hard samples 主要落在 B/C 区域，而不是 D 区域。
这说明 hard module 确实在做“边界强化”，而不是在放大伪困难噪声。

---

## 10. 结果解释模板

若实验结果符合预期，可以给出如下解释：

> 在该 synthetic setting 中，聚类困难主要集中于 prototype 边界附近的样本。相比纯 baseline，加入困难样本挖掘/生成模块后，模型在边界区域获得了更高的聚类质量和更强的 prototype separation。这表明困难样本模块的主要作用并非改善所有样本，而是通过强化边界附近的学习信号，提升 cluster boundary 的判别性。

---

## 11. 失败情况与解释

如果 hard module 没有明显优于 baseline，也要分析原因：

### 情况 A：全局与边界指标都无提升
说明：
- baseline 已足够拟合该 synthetic 数据；
- 边界不是真正瓶颈；
- 或 hard sample 构造方式无效。

### 情况 B：全局略升，但边界无明显改善
说明：
- hard module 可能只是起到普通数据增强作用；
- 尚未真正对边界样本起作用。

### 情况 C：边界指标变差
说明：
- hard module 选错了样本；
- 或生成样本偏离了真实边界流形；
- 或强化了 pseudo-hard 噪声。

---

## 12. 本 phenomenon study 的作用边界

该 study 最终想证明的是：

> 在一个“困难集中于边界”的可控场景下，困难样本挖掘/生成确实有潜力改善聚类边界质量。

而不是证明：

- 所有 hard mining 方法都有效；
- 对任意真实数据集都有效；
- 对任意噪声都鲁棒。

它只是为后续真实数据实验提供一个“边界强化机制可行”的现象证据。

---

## 13. 最小可执行版本

如果时间有限，至少完成以下最小版本：

1. 用 3 个二维 Student-t 簇构造双视图 synthetic 数据；
2. 训练 Baseline 与 Baseline+Hard；
3. 计算：
   - 全局 ARI
   - boundary-ARI
   - 边界样本平均 margin
4. 画一张散点图，显示：
   - 样本分布
   - prototype
   - 被 hard module 选中的点

只要这四步做出来，这个 phenomenon study 就已经可以写进论文的早期版本。

---

## 14.备注

train.py 的启动命令可以参考/scripts/alpha_beta_tau_Search/Caltech-5V.sh

其中 tau<0 时就是 baseline ， tau>0 时就是困难样本挖掘。可以给 tau 一个特别大的数字，以减轻困难样本挖掘对 baseline 的影响。还要注意，启动实验前需要新增一个对应的 config.yaml

这个 phenomenon study 需要你在项目文件夹下新增一个文件夹进行处理 不要修改原始文件夹下的文件