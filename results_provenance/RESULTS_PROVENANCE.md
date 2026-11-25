# 结果溯源（Results Provenance）

**论文成绩对应的代码版本：本仓库当前代码（第一版公开发布）。** 本目录内的结果文件即论文全部成绩的最终产出，每文件 896 行、唯一键无重复、每行带来源列（`source`）可逐行溯源。

## 一、版本说明

| 版本 | 说明 |
|---|---|
| **本仓库代码（第一版公开发布）** | 论文全部成绩由本版本产出；`results_provenance/` 内的结果文件即其最终产出 |

## 二、论文成绩与结果文件对照

| 论文位置 | 成绩 | 结果文件（本目录） | MD5 |
|---|---|---|---|
| 主结果：hybrid（RRF 混合检索）896 全量 | **F1=0.7034**，P=0.5822，R=0.8881（TP=262 FP=188 FN=33 TN=413） | `hybrid_results_896.csv` | aff8f94bc37df0f8154e45094688da70 |
| 对照臂：baseline（单路向量检索）896 全量 | **F1=0.6762**，P=0.5485，R=0.8814（TP=260 FP=214 FN=35 TN=387） | `baseline_results_896.csv` | dd91b790ff055f705dd54e781093be9f |
| 多模型消融：GLM-5.2（验证层）896 全量 | **F1=0.7478**，P=0.6532，R=0.8746（TP=258 FP=137 FN=37 TN=464） | `glm-5_2_results_896.csv` | 0887066dfce737b4bed3ab083236bd9d |
| 样本清单（可复现抽样） | 896 全量清单（seed=42） | `manifest_n896_seed42.json` | 66245de8554f70b45dd3986f80b6b4e1 |

## 三、最终结果解读

1. **主结论**：在 896 对 BioRED 全量测试上，RRF 混合检索（hybrid）F1=0.7034，较单路向量检索（baseline）的 0.6762 高 **+2.72pp**；精确率（0.5822 vs 0.5485，+3.37pp）与召回率（0.8881 vs 0.8814，+0.67pp）同步占优，提升主要来自精确率——混合召回为图谱提供了更高质量的文献证据，显著减少了误导性判断。
2. **多模型消融（GLM-5.2，验证层）**：固定图谱证据、仅更换验证模型时，GLM-5.2 达 F1=0.7478、P=0.6532，精确率为三臂最高，表明 GRACE 的图谱证据对不同验证模型均有效，且 GLM-5.2 的输出 100% 通过格式校验门。
3. **错误结构**：残余错误以"强共现 gold-NO 对"为主（三臂共同的盲区），属基准难点而非管线缺陷。

## 四、复现说明

- 抽样：`manifest_n896_seed42.json`（全量 896 对，seed=42）
- 结果文件每行含 `source` 列标记该行预测的产出批次，供逐行审计
- **指标重算**（任一结果 CSV 均可）：

```python
import csv
rows = list(csv.DictReader(open("hybrid_results_896.csv", encoding="utf-8")))
tp = sum(1 for r in rows if r["predicted"] == "yes" and r["true_label"] == "yes")
fp = sum(1 for r in rows if r["predicted"] == "yes" and r["true_label"] == "no")
fn = sum(1 for r in rows if r["predicted"] != "yes" and r["true_label"] == "yes")
P = tp / (tp + fp); R = tp / (tp + fn); F1 = 2 * P * R / (P + R)
print(f"P={P:.4f} R={R:.4f} F1={F1:.4f}")   # 预期: P=0.5822 R=0.8881 F1=0.7034
```

- 亦可直接运行仓库的 `python test_biored_rrf_hybrid.py report`（两臂 P/R/F1、混淆矩阵、McNemar 精确检验与逐样本对照）
- 各结果文件的 MD5 即本版本成绩的提交标识；建议使用者以 git 提交本仓库并打 tag（如 `results-v1`）固化
