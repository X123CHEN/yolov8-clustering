# K=2 潜在域数据清单

用户已确认 K=2。本目录只保存新 TXT 和审计记录；原 RDD2022 图片、标注、ImageSets 不改动、不复制、不移动。

## 文件与数量

| 文件 | 行数 | 格式/用途 |
|---|---:|---|
| Cluster_0_train.txt | 5,913 | 每行一张图片及原检测框；簇 0 专家训练 |
| Cluster_1_train.txt | 6,829 | 每行一张图片及原检测框；簇 1 专家训练 |
| Cluster_0_val.txt | 738 | 簇 0 验证清单，每行一张图片 |
| Cluster_1_val.txt | 854 | 簇 1 验证清单，每行一张图片 |
| combined_train.txt | 6,829 | 每行 Cluster_0 记录 & Cluster_1 记录，供后续两域训练 |
| combined_val.txt | 1,592 | 两个验证簇的唯一记录并集，每行一张图片，无 & |
| all_train.txt | 12,742 | 原训练图片唯一并集，每行一张；普通检测训练/评估使用 |
| split_summary.json | — | 聚类数量、国家组成、合并规则、重复次数与源文件哈希 |
| verification.json | — | 独立覆盖、标注、标签/中心、合并规则及源文件校验 |

## 聚类与验证分配

训练集固定为 India 2,578、Japan 6,320、United_States 3,844 张，重拟合 K=2（seed=2、n_init=10），标签及中心重现原已审阅候选。

验证集固定为 India 322、Japan 790、United_States 480 张，重新用同一 `model_data/yolov8_s.pth`、P3/feat1 均值+标准差提取 256 维特征。复用训练集的 Z-score 参数和中心按最近中心分配，不在验证集拟合标准化或 K-means，不重新划分 train/val。

| 簇 | India train / val | Japan train / val | United_States train / val |
|---|---:|---:|---:|
| 0 | 306 / 42 | 5,055 / 631 | 552 / 65 |
| 1 | 2,272 / 280 | 1,265 / 159 | 3,292 / 415 |

## 合并规则

已逐行验证参考 `ImageSets/251030_train.txt` 使用 Japan、India、United_States 顺序，按最大长度循环较短国家清单并以 & 分隔。本目录对应改为 Cluster_0、Cluster_1 顺序，按 6,829 行确定性循环，不 shuffle、不截断：

```text
Cluster_0_train[i % 5913] & Cluster_1_train[i % 6829]
```

导出的分隔符为 &，不添加额外空格。簇 0 的前 916 张图片各出现两次，其余 4,997 张各出现一次；簇 1 的全部图片各一次。配对清单共出现 13,658 次图片引用，但包含的唯一训练图片仍为 12,742 张，没有遗漏。专家训练应使用各自单簇清单，避免把配对清单当成普通单图片格式。

参考 `251030_val.txt` 是无 & 的 1,592 张唯一图片并集，因此 combined_val.txt 沿用单图片格式，确定性按 Cluster_0 后 Cluster_1 拼接，不做重复采样。all_train.txt 额外提供单图片训练并集，保持 India、Japan、United_States 源清单顺序。

TXT 每条记录原样保留来源清单的图片路径和检测框字段，未追加簇编号或改动类别 ID。CSV 中图片路径为解析符号链接后的规范路径，可能与 TXT 中的原始路径字符串不同；校验按真实路径对应同一图片。已知零宽度原标注同样保留，未擅自清洗数据。

## 相关结果与复核

- `../results/final_cluster_labels.csv`：训练图片正式标签。
- `../results/final_val_cluster_labels.csv`：验证图片标签、两中心平方距离。
- `../results/cluster_centers.npy` 与 `scaler_mean.npy` / `scaler_scale.npy`：训练集参数，必须成套应用。
- `../features/validation/`：新提取的验证特征、图片映射和提取器审计。
- `../results/final_k_decision.json`：确认记录。

在项目根目录运行复核：

```bash
latent_domain_analysis/.venv/bin/python latent_domain_analysis/verify_domain_splits.py
```

重新导出脚本拒绝覆盖已完成的导出；部分运行如中断，只允许复用与本次预期完全一致的 TXT。

后续先使用各簇 TXT 训练两个专家，再适配完整方法的数据加载和专家/判别器分支为两域。当前原训练代码仍存在固定三分支的实现，不能只更换 combined_train.txt 路径就直接训练。此步骤未执行专家或完整方法训练。
