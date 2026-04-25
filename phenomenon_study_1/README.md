# phenomenon_study_1（四阶段版本）

按你的要求，这一版把实验拆成 **4 个独立 `.py` 文件**，分别对应四个阶段。

## 实验脚本

1. `stage1_boundary_identification.py`  
   最简二分类同方差高斯。定义真实边界困难样本，比较：
   - 欧氏 prototype 困难度（到真实类中心距离）
   - 决策边界 margin 困难度（分类 margin 反向）

2. `stage2_reweight_training.py`  
   在训练中对 top-k 困难样本重加权，比较两种困难度定义带来的：
   - overall accuracy
   - boundary accuracy
   - hard sample precision

3. `stage3_overlap_sweep.py`  
   连续调节两类均值距离（overlap 从易到难），记录两种方法在：
   - hard precision
   - boundary accuracy gain
   - overall accuracy gain

4. `stage4_heteroscedastic_stability.py`  
   从同方差扩展到异方差高斯，测试现象稳定性，继续比较两种困难样本方式。


5. `stage2_mlp_capacity_control.py`  
   在阶段二基础上加入小型 MLP（`32,16`）与线性 LR 对照，分别做 baseline / eu-hard / margin-hard 重加权训练。
   其中 margin-hard 改为**按各自模型**（LR/MLP）概率 margin 单独定义，并输出均值±标准差及 paired 优势统计，直接回答：
   - hard sample 价值是否被模型能力压制；
   - margin-hard 相对 eu-hard 的边界增益在 LR 与 MLP 下是否一致。

---

## 快速运行示例

```bash
python phenomenon_study_1/stage1_boundary_identification.py
python phenomenon_study_1/stage2_reweight_training.py
python phenomenon_study_1/stage3_overlap_sweep.py
python phenomenon_study_1/stage4_heteroscedastic_stability.py
python phenomenon_study_1/stage2_mlp_capacity_control.py
```

输出将分别写入：

- `phenomenon_study_1/outputs/stage1/`
- `phenomenon_study_1/outputs/stage2/`
- `phenomenon_study_1/outputs/stage3/`
- `phenomenon_study_1/outputs/stage4/`
- `phenomenon_study_1/outputs/stage2_capacity_control/`

每个阶段都会输出 `trials.json` 与 `summary.json`，阶段 3/4 还会输出趋势图。

---

## 与你的四个假设的对应

- 阶段一：验证“边界主导型困难”下 margin 困难度识别更准。
- 阶段二：验证 margin 挖掘出的 hard 样本训练价值更高（尤其 boundary 区域）。
- 阶段三：验证随着 overlap 增强，margin 相对欧氏的优势扩大。
- 阶段四：验证在轻微分布不规则（异方差）下现象仍成立。

如果你希望，我下一步可以再补一个 `run_all_stages.py` 用于一键串联四阶段并自动汇总总报告。

# phenomenon_study / Stage5

本目录是根据 `方案2.pdf` 新增的 stage5 现象研究实现：

> 验证“只对 hardest 子集做 line-search”与“所有样本都可作为 anchor 做 line-search”在边界覆盖与边界质量上的差异。

## 脚本

- `stage5_all_anchor_line_search.py`

## 运行

```bash
python phenomenon_study/stage5_all_anchor_line_search.py
```

可选参数（示例）：

```bash
python phenomenon_study/stage5_all_anchor_line_search.py \
  --seeds 0 1 2 3 4 \
  --n-per-cluster 400 \
  --df 4 \
  --p-conflict 0.4 \
  --anchor-ratio 0.2
```

## 输出

默认输出到 `phenomenon_study/stage5_outputs/`：

- `stage5_trials.json`：每个 seed、每种策略（Only-H / M+H / All / Random）明细
- `stage5_summary.json`：按策略聚合均值
- `stage5_scatter_three_policies.png`：必做散点三联图（Only-H / M+H / All）

## 当前 Stage5 实现与方案2指标对齐

当前实现覆盖：
- 全局与边界指标：ACC/NMI/ARI + boundary-ACC/NMI/ARI
- 边界 margin
- 覆盖率：class coverage / pair coverage / radial coverage
- anchor 策略对比：Only-H / M+H / All / Random

后续可继续补齐：
- disagreement 指标
- 训练过程曲线（loss / prototype drift / assignment flip）
- 更严格的 line-search 轨迹约束版本
