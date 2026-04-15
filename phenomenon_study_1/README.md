# phenomenon_study_1（独立于原工程）

该目录是按 `phenomenon_study_1.md` 的要求新增的**独立实验目录**，不修改原始训练代码与配置文件。

## 你可以直接运行

```bash
python phenomenon_study_1/run_study.py
```

默认会执行：
- 3 个二维 Student-t 簇（每类 400 样本，`df=4`）；
- 双视图构造（view1/view2）与边界冲突注入；
- Baseline vs Baseline+Hard（简化 hard mining 版本）；
- 多随机种子统计；
- 输出 boundary-ARI、boundary-margin、boundary-disagreement 等指标；
- 生成散点图和柱状图。

输出目录：`phenomenon_study_1/outputs/`

---

## 文件说明

- `run_study.py`：主实验脚本（数据生成 + 两个方法 + 指标 + 画图）。
- `outputs/trials.json`：每个 seed 的详细结果。
- `outputs/summary.json`：多 seed 平均结果。
- `outputs/figure_1_latent_scatter.png`：必做散点图。
- `outputs/figure_2_3_4_boundary_metrics.png`：边界指标对比图。

---

## 与原项目对接时可能缺失/不完整的部分（建议补齐）

你在任务里提到“项目可能不完整”。结合当前仓库状态，以下部分很可能缺失或需要明确：

1. **`phenomenon_study_1.md` 备注里的 `train.py` 不存在**  
   当前仓库入口是 `main.py`，若你想完全复用原训练管线，需要给出等价命令映射（例如 `train.py` 与 `main.py` 参数对应关系）。

2. **缺少 synthetic 数据集接入主工程的数据加载路径**  
   目前 `dataset.py` 主要是既有数据集接口。若要直接在原模型上做该 study，需补：
   - synthetic 数据生成脚本；
   - 对应 `config/*.yaml`（你备注中也提到了）；
   - `MultiViewDataset` 对新数据集名的加载分支。

3. **“hard mining” 在原工程中的开关约定需要文档化**  
   备注里提到 `tau<0` baseline、`tau>0` hard，但当前 README 未写。建议补到主 README，避免复现实验时歧义。

4. **边界样本 oracle 标注与评估脚本缺失**  
   现有工程常规输出 ACC/NMI/ARI；若论文要强调 boundary 指标，需要补独立评估脚本（本目录已给出简化版实现思路）。

---

## 说明

本目录的 `Baseline+Hard` 是一个“现象验证版” hard 机制（不是替代你论文最终方法），目标是快速验证：
- hard 机制是否更偏向命中边界样本；
- boundary 指标是否比全局指标更明显受益。

如果你希望我下一步把该 study **接到原 `main.py + config` 训练流**（完全走项目模型），我可以在不破坏主流程的前提下继续补齐：
- synthetic dataset loader
- 新增 config 文件
- 一键运行脚本
- 统一结果汇总脚本
