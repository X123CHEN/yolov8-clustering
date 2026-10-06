# 潜在域数量确定与硬聚类阶段执行方案

## 1. 当前阶段目标

本阶段只完成以下内容：

1. 基于固定的共享 YOLOv8 特征提取器，为 Japan、India、US 三个源域训练图像提取风格特征；
2. 在不使用任何目标域数据的前提下，低成本确定潜在域数量 \(K\)；
3. 基于最终确定的 \(K\) 完成离线硬聚类，生成每张源域图像的 latent-domain 标签；
4. 对最终聚类结果进行必要的定量分析和可视化，判断潜在域划分是否合理。

**本阶段到硬聚类结果分析为止，不执行后续 MAMDG 验证，也不实现 Online Clustering、Soft Assignment、K 类判别器或 SupCon。**


## 2. 不允许修改的内容

1. 不修改、删除、移动或重构原项目已有文件与目录，包括但不限于 `train.py`、`nets/`、`utils/` 等，尽量不要修改，可以进行调用或者小幅改动。
2. 所有新增功能必须放在新的独立目录中，例如：

```text
latent_domain_analysis/
├── config.py
├── extract_features.py
├── select_k.py
├── analyze_clusters.py
├── README.md
├── features/
├── results/
└── figures/
```

3. Japan、India、US 必须使用**同一个固定共享 YOLOv8 backbone** 提取特征，不能按国家使用不同专家模型。
4. 确定 \(K\) 的整个过程只能使用 source data，不得使用 Norway、Czech、China D、China M 的图像、标签、检测结果或统计信息。
5. 第一版风格表示固定采用指定浅层特征图的 channel-wise mean 和 standard deviation：
   \[
   s_i=[\mu(F_i),\sigma(F_i)]
   \]
   不得擅自加入新的 style encoder、attention、CLIP、多尺度特征拼接等结构。
6. 聚类阶段不使用 Mosaic、MixUp、HSV 随机增强、随机翻转、随机裁剪等随机训练增强。
7. 当前阶段禁止实现或运行后续 MAMDG、专家权重训练、域判别器、Soft Clustering、Online Clustering、SupCon。

## 3. Step 1：提取固定 Style Features

固定使用本项目 `model_data/yolov8_s.pth` 作为统一特征提取器 \(F_0\)，不替换为国家专家或其他检测权重。模型规格为 YOLOv8-s；骨干权重必须完整加载，提取时使用 `eval()`、关闭梯度。

数据：
\[
D_S=Japan\cup India\cup US
\]

流程：

```text
Image
  ↓
Fixed YOLOv8 Backbone
  ↓
P3 = feat1 = backbone.dark3 输出（128×80×80）
  ↓
channel-wise mean / std
  ↓
Style Vector
```

得到：
\[
s_i=[\mu(F_i),\sigma(F_i)]
\]

特征只提取一次，并保存，例如：

```text
features/
├── style_features.npy
└── metadata.csv
```

`metadata.csv` 至少保留：
- image path / image id
- original country
- 原始检测标签索引信息

其中 `country` 和病害类别信息只用于后续分析，**不能参与聚类计算**。

## 4. Step 2：低成本确定潜在域数量 K

对保存好的 style vectors 先进行 Z-score standardization。

候选范围初步设为：
\[
K=2,3,\ldots,8
\]

所有候选 \(K\) 只在固定 style vectors 上运行 K-means，不重新训练 YOLO。

### 4.1 K 选择原则

主判据：
- **Silhouette Score：越大越好**

辅助交叉验证：
- **Calinski-Harabasz Index（CH）：越大越好**
- **Davies-Bouldin Index（DB）：越小越好**
- **Clustering Stability：越高越好**

独立统计参考：
- **GMM + BIC：BIC 越小越好**

只有当 Silhouette、CH、DB、Stability 出现明显冲突时，再增加：
- **Gap Statistic**

不要自行设计加权综合评分，例如 `0.4*Silhouette + 0.2*CH + ...`。

最终 \(K\) 以 Silhouette 为主要依据，并结合 CH、DB、Stability、cluster occupancy 和 GMM+BIC 进行人工确认。

## 5. Step 3：K 选择结果必须可视化

必须输出可直接判断为什么选择某个 \(K\) 的图表，例如：

```text
figures/
├── silhouette_vs_k.png
├── ch_vs_k.png
├── db_vs_k.png
├── stability_vs_k.png
├── gmm_bic_vs_k.png
└── silhouette_profile_kX.png
```

如启用 Gap Statistic，再输出：

```text
gap_vs_k.png
```

同时保存数值结果：

```text
results/k_selection_metrics.csv
```

至少包含：

| K | Silhouette | CH | DB | Stability | GMM-BIC | Cluster Occupancy |
|---|---:|---:|---:|---:|---:|---|

**不要仅凭 PCA、t-SNE 或 UMAP 的二维图主观确定 K。**

## 6. Step 4：人工确认最终 K 后生成 Hard Latent Domains

`select_k.py` 只负责提供选择依据，不要自动把结果直接传入后续训练。

人工查看图表和统计结果后，在配置中明确设置：

```python
FINAL_K = ...
```

随后使用全部 source style vectors，以最终 \(K\) 重新运行 K-means，得到：
\[
C_1,C_2,\ldots,C_K
\]

以及每张图像的硬潜在域标签：
\[
\hat d_i\in\{0,\ldots,K-1\}
\]

保存：

```text
results/
├── final_cluster_labels.csv
├── cluster_centers.npy
└── cluster_composition.csv
```

## 7. Step 5：最终聚类结果分析

### 7.1 Cluster Occupancy

统计每个 cluster：
- 样本数量；
- 样本占比。

用于检查是否存在极小簇或明显异常簇。

### 7.2 Country × Latent Domain

统计 Japan、India、US 在各潜在域中的分布。

重点观察：
- 一个 cluster 是否包含多个国家；
- 同一个国家是否分散到多个 cluster。

用于判断 latent domain 是否只是原 country domain 的简单复现。

### 7.3 病害类别分布

对每个 cluster 统计：

\[
P(D00|C_k),\quad
P(D10|C_k),\quad
P(D20|C_k),\quad
P(D40|C_k)
\]

建议按照 cluster 内各类别病害实例数量计算比例。

主要用于检查聚类结果是否过度绑定病害类别，避免 latent domain 实际变成 semantic class clustering。

输出：

```text
results/damage_distribution.csv
figures/damage_distribution_by_cluster.png
```

### 7.4 每个 Cluster 的代表图片

对 cluster \(C_k\)，按照样本到 cluster center 的距离排序，选择距离最近的一组图片作为代表样本。

代表图片必须由距离自动确定，不进行人工挑选。

输出例如：

```text
figures/
├── cluster_1_representatives.png
├── cluster_2_representatives.png
└── ...
```

### 7.5 二维可视化

可使用 PCA 作为第一版主要可视化，必要时再补充 UMAP。

至少展示：
- latent cluster 分布；
- original country 与 latent cluster 的关系。

二维可视化只用于解释，不作为最终 \(K\) 的唯一依据。

## 8. 当前阶段建议输出文件

```text
latent_domain_analysis/
├── features/
│   ├── style_features.npy
│   └── metadata.csv
│
├── results/
│   ├── k_selection_metrics.csv
│   ├── final_cluster_labels.csv
│   ├── cluster_centers.npy
│   ├── cluster_composition.csv
│   └── damage_distribution.csv
│
└── figures/
    ├── silhouette_vs_k.png
    ├── ch_vs_k.png
    ├── db_vs_k.png
    ├── stability_vs_k.png
    ├── gmm_bic_vs_k.png
    ├── silhouette_profile_kX.png
    ├── pca_clusters.png
    ├── country_cluster_matrix.png
    ├── damage_distribution_by_cluster.png
    └── cluster_X_representatives.png
```

## 9. 当前阶段结束条件

完成以下内容后立即停止：

- 已提取固定 source style features；
- 已完成 \(K=2\sim8\) 的 K-means 低成本比较；
- 已输出 Silhouette、CH、DB、Stability、GMM+BIC 的数值和可视化；
- 必要时完成 Gap Statistic；
- 已人工确定最终 \(K\)；
- 已生成最终 Hard Latent Domain labels；
- 已完成 cluster occupancy、Country × Cluster、病害类别分布和代表图片分析；
- 已保存最终 cluster centers。

**此时不要继续执行专家训练或 MAMDG 验证。**

下一阶段需要先人工检查以上结果，再决定是否进入 Hard-Latent-Domain MAMDG 实验。

## 10. 本次执行约定（用户确认后落实）

### 10.1 数据与已有实验的关系

- 从 `datasets/RDD2022/ImageSets/India_train.txt`、`Japan_train.txt`、`United_States_train.txt` 读取原始源域训练图片，分别为 2,578、6,320、3,844 张，共 12,742 张。
- 保留各国原始样本数量，不做国家均衡采样。国家与病害标签只用于审计、解释和统计，不参与特征聚类。
- 不读取 `README_style.md`、`260708_style_train.txt` 或旧风格标签、中心、专家、特征缓存作为新实验输入。
- 仅源域训练集参与标准化拟合、K 选择及最终聚类；K 选择阶段源域验证清单仅用于路径交叉检查。人工确认 K 后允许用训练参数分配源域验证图片，目标域不参与任何环节。
- 全部代码与实验输出位于 `latent_domain_analysis/`；除本方案文档外，不修改原项目代码。

### 10.2 确定性特征表示

- 使用同一个 `model_data/yolov8_s.pth`；保存权重哈希、源代码哈希及清单哈希。
- RGB 转换后直接以 BICUBIC 缩放到 640×640，不使用 letterbox 或随机增强；像素转 float32 并除以 255。
- 固定取骨干 `dark3` 输出，称为 P3/feat1，不取 Neck 中同名的 P3 张量。
- 空间维度上的 128 个均值与 128 个标准差拼接为 256 维；标准差采用 `correction=1`。
- 固定评估模式、禁用梯度、使用 float32，不进行模型训练。
- 原网络存在冗余硬编码 CUDA 调用，允许只在提取进程内适配设备，保持卷积、BN、激活计算及原文件不变。

### 10.3 K 选择与低成本规则

- 256 维原始特征只提取一次；Z-score 参数只在全部源域训练特征上拟合，并保存均值及尺度。
- K=2～8；每个 K 使用种子 `[0,1,2,3,4]`，每次 K-means 使用 k-means++、`n_init=10`、`max_iter=300`、`tol=1e-4`、Lloyd 算法。
- Silhouette 使用种子 3407 固定抽取的 5,000 张训练图片，所有 K 和所有重复共享同一组索引，以标准化空间欧氏距离计算。保存索引；若有簇在子集中缺失，应统一增加评估样本而非单独调整某个 K。
- CH、DB、簇占比在全部 12,742 张训练特征上计算。Silhouette、CH、DB 报告五次重复的均值和样本标准差。
- Stability 定义为同一个 K 的五组全量训练标签之间两两 ARI，共 10 对；报告均值和样本标准差。这只衡量初始化稳定性，不等同于重采样稳定性。
- GMM 固定 `covariance_type='diag'`、`n_init=3`、`max_iter=300`、`random_state=0`、`reg_covar=1e-6`，报告 BIC 和收敛状态。允许 GMM 作为统计分析例外；其软概率不用于最终标签或后续训练。
- 每个 K 以五次运行中 inertia 最低的一次作为候选展示划分；完整保存各次结果，禁止只展示最有利的一次 Silhouette。
- 指标明显冲突时补充 Gap；候选差异接近随机波动时可补充重采样稳定性。调整须记载原因，并对所有候选采用一致规则。
- 不要求存在唯一最佳 K；若当前表示下缺少清晰分簇证据，应如实报告。聚类指标不能代替后续检测泛化效果验证。

### 10.4 确认、输出和异常处理

- 首次执行到 K 比较报告及候选分析为止，`FINAL_K=None`。人工确认前只生成明确标为 candidate 的分析，不能生成或声明最终潜在域。
- 人工确认 K 后，用该 K 的候选种子和相同参数在全部训练特征上重新拟合，校验与所审阅候选的一致性；若不一致，先报告再处理。
- 标签编号统一为 0～K-1。保存标准化参数、标准化空间中心、图片到标签映射、配置和版本、选择 K 的理由及代表图片的排序距离。
- 数据审计发现 Japan_train.txt 第 1983 行有一个零宽度 D20 框。保持全部图片和原标注记录；病害统计同时报告原始实例数和有效实例数，有效比例排除该退化框，不更改聚类样本或输入特征。
- 未确认最终 K 前，最终聚类结束条件尚未满足；确认后完成最终分析即停止，不训练专家或域判别器。

### 10.5 经用户确认增加的稳健性验证

保留第 10.3 节的初始化稳定性作为独立结果，新增以下两项，全部使用现有 256 维源域特征：

1. **评估子集敏感性**：固定种子 100～109，分别从 12,742 张图片中无放回抽取 5,000 张，各 K 共用每组索引；固定原候选划分，不重新聚类。输出 Silhouette 的均值、样本标准差、范围、每组排名及 K2−K3 配对差值。
2. **80% subsampling stability**：固定种子 200～209，各抽取 10,194 张图片，不进行国家均衡；每组索引供全部 K 使用。固定原标准化参数和 K-means seed=0、n_init=10、max_iter=300、tol=1e-4；每次只在抽出的图片上拟合中心，再为全部 12,742 张图片分配标签。在共同图片集合上计算运行间 ARI（45 对/K）和相对原始全量划分的 ARI；报告簇占比变化。
3. 为观察拟合扰动后的 K 排序，每次重拟合统一使用初轮种子 3407 的固定 5,000 图片评估子集计算 Silhouette；该结果与第 1 项分表报告，不混合两种波动。CH/DB 仍在全量预测标签上描述性计算。
4. 占比展示通过最大重叠匹配原候选编号，ARI 无需对齐。保存每次中心、全量标签、抽样索引和哈希，便于复核。
5. 以上均是敏感性分析，并非独立留出集性能。标准差不解释为置信区间；不得预设 K=2 必须获胜，若排序变化或划分不稳定应如实报告。
6. 所有补充结果写入 `latent_domain_analysis/results/robustness/` 与 `figures/robustness/`，不改变原数据、原候选、原指标或 FINAL_K。

### 10.6 人工确认 K=2 后的正式清单导出（已完成）

- 用户确认 K=2 后，设置 FINAL_K=2，以已审阅候选种子 2 重新拟合，重现原训练标签及中心。训练簇为 5,913 / 6,829 张。
- 对原 India/Japan/United_States 验证集共 1,592 张图片，使用训练时相同模型和预处理重新提取特征，只应用训练集 scaler/中心，不在验证集拟合。验证簇为 738 / 854 张。
- 新增产物均放在 `latent_domain_analysis/`。原 RDD2022 图片、标注和 ImageSets 仅读取；导出前后核验原 ImageSets 16 个 TXT 哈希一致。
- `domain_splits/Cluster_0~1_train/val.txt` 每行保持原图片路径及原检测框，单簇清单没有重复或遗漏。
- `combined_train.txt` 参照 251030_train.txt，按 Cluster_0、Cluster_1 顺序循环配对到最大长度，以 & 分隔，共 6,829 行；簇 0 的前 916 张图片各出现两次，其余各一次，没有漏图。
- `combined_val.txt` 参照 251030_val.txt，采用一图一行的验证并集，共 1,592 行，不重复。额外提供 `all_train.txt` 的唯一训练并集，共 12,742 行。
- TXT 保留原路径字符串，CSV 使用规范真实路径；校验按解析符号链接后的路径对应图片，并检查原标注字段一致。
- 详细规则及校验见 `domain_splits/README.md`、`split_summary.json`、`verification.json`。本阶段结束，两个专家训练及完整两域方法的适配留待后续任务。
