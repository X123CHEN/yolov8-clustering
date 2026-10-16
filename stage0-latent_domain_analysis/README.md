# 源域潜在域数量分析

本模块从 India/Japan/United_States 原始训练清单重新开始，忽略旧风格分组和缓存。
完整协议见上级目录的《潜在域K选择与硬聚类执行方案.md》第 10 节。

当前阶段已完成：用户确认 K=2，正式训练聚类及验证分配已导出。训练簇为 5,913 / 6,829 张，验证簇为 738 / 854 张。数据清单见 [domain_splits/README.md](domain_splits/README.md)。初轮选择与稳健性记录中的 `final_k=null` 是当时的阶段状态，当前确认记录见 `results/final_k_decision.json`。

## 数据与表示

- 三个原始 source train 清单，共 12,742 张，保留原始国家比例。
- 固定 `../model_data/yolov8_s.pth`；YOLOv8-s backbone.dark3（P3/feat1）。
- RGB，直接 BICUBIC resize 640×640，float32 /255，无增强。
- 通道均值与样本标准差（correction=1），256 维；全量 source train Z-score。
- 国家/检测标签仅作分析；目标域不读取。源域 val 路径仅检查训练交叉。

## 环境和命令

在项目根目录执行。本次 `.venv` 基于 `yolo_dg` 的 system-site-packages，单独安装 scikit-learn 1.5.2 及依赖，不改原训练环境。

```bash
/data4/czh/anaconda3/envs/yolo_dg/bin/python -u latent_domain_analysis/extract_features.py --device cuda:0 --batch-size 16 --workers 4
latent_domain_analysis/.venv/bin/python -u latent_domain_analysis/select_k.py
```

完成特征缓存带 SHA256 校验；为避免误覆盖，提取器拒绝覆盖完成的缓存。
对不同实验设置应使用新的目录，不能混合不同提取器或预处理的缓存。
原 Conv 中多余的硬编码设备调用仅在独立提取进程内替换，原始代码不变。

每个 K：五个种子，每次 10 次初始化；固定 5,000 图片计算 Silhouette；CH/DB/占比用全量。
初始化稳定性为五组标签的十个两两 ARI；GMM diag，3 次初始化，仅输出 BIC 参考。
线程限制为 4，避免统计阶段占满共享服务器。

## 结果与人工确认

- `features/`：原始风格向量、逐图片标注索引、提取过程/权重/清单哈希。
- `results/k_selection_metrics.csv`：各 K 汇总，包含指标均值与标准差、占比和 GMM 收敛情况。
- `results/k_selection_runs.csv`、`stability_pairs.csv`：完整逐次和两两比较结果。
- `results/candidates/k*.npz`：每个 K 最低 inertia 运行的候选标签和中心，尚非最终标签。
- `figures/k_selection_overview.png`：汇总曲线；另有单项曲线及每个 K 的 silhouette profile。

可以生成任意已评估 K 的候选解释图（不设置最终 K）：

```bash
latent_domain_analysis/.venv/bin/python latent_domain_analysis/analyze_clusters.py --candidate 2
```

候选分析单独存入 `results/candidate_analysis/k2/` 和 `figures/candidate_analysis/k2/`。
包括簇占比、国家矩阵（两种归一化）、类别实例分布、PCA、每簇距中心最近的 9 张图片。
PCA 仅用于解释，不参与聚类或 K 的选择。

人工审阅并确认后，在 `config.py` 明确设置 `FINAL_K`，然后运行：

```bash
latent_domain_analysis/.venv/bin/python latent_domain_analysis/analyze_clusters.py
```

未设置 FINAL_K 时命令必须拒绝执行。最终重新拟合需重现审阅过的候选标签。
中心在标准化特征空间中，应用时必须同时使用 `results/scaler_mean.npy` 和 `scaler_scale.npy`。
最终编号为 0～K-1，不自动衔接任何专家/对抗/软聚类训练。

## 已知数据问题

Japan_train 第 1983 行有一个零宽度 D20 框。保留图片及原标注，仅在有效实例比例中排除该框；CSV 同时保存原始与有效计数。
路径审计检查每个清单内及三国之间的重复，以及各国 train/val 的路径交叉；不等同于图像内容去重。

## 补充稳健性实验

保留原初始化稳定性结果，另行执行两种扰动，使用已有特征和原标准化参数，不读取图片或重新提取特征：

```bash
latent_domain_analysis/.venv/bin/python -u latent_domain_analysis/robustness_checks.py
latent_domain_analysis/.venv/bin/python latent_domain_analysis/verify_robustness.py
```

1. 种子 100～109，各无放回抽取 5,000 张评估图片。每组索引由 K=2～8 共用，固定原候选标签，不重新拟合。报告 Silhouette mean±std、每组排名和 K2−K3 配对差值。
2. 种子 200～209，各无放回抽取 round(0.8×12742)=10,194 张拟合图片。固定 K-means seed=0、n_init=10；所有 K 共用各组索引。每次重新拟合后为全部图片分配标签，计算运行之间的 45 对 ARI/K、与原全量划分的 ARI 和簇占比。
3. 重采样拟合的 Silhouette 另用原种子 3407 的固定 5,000 张图片评分，保持评估子集不变，用于观察拟合扰动后的 K 排名。它不是独立留出集性能。

结果与全部抽样索引、70 次拟合的中心/全量标签保存到 `results/robustness/`；图表保存到 `figures/robustness/`。
簇占比先按最大重叠匹配原候选编号再展示，避免编号交换造成伪变化；ARI 直接比较原标签，不需要编号对齐。
所有标准差采用 ddof=1，只描述波动，不作置信区间。两种扰动分别统计；重叠子集、共享图片和两两 ARI 都存在依赖性。
补充实验不自动设置 FINAL_K，不覆盖原候选或初轮指标。完成后拒绝直接覆盖同目录的已完成实验。

## 正式 K=2 清单导出

已执行以下步骤，原始 RDD2022 仅作读取：

```bash
latent_domain_analysis/.venv/bin/python latent_domain_analysis/analyze_clusters.py
/data4/czh/anaconda3/envs/yolo_dg/bin/python latent_domain_analysis/extract_val_features.py --device cuda:0 --batch-size 16 --workers 4
latent_domain_analysis/.venv/bin/python latent_domain_analysis/export_domain_splits.py
latent_domain_analysis/.venv/bin/python latent_domain_analysis/verify_domain_splits.py
```

完成的验证特征及 TXT 导出有覆盖保护，不必重复运行生成命令。验证集只使用训练集 scaler/中心分配。单簇 TXT 保留原路径及原检测标注；combined_train.txt 按参考循环配对并使用 &，combined_val.txt 是不重复的单图片验证并集；all_train.txt 是额外的普通训练并集。
