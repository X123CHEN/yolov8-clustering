# YOLOv8 风格软聚类对抗实验

本项目在 YOLOv8 检测器上实现基于风格伪域的对抗训练，用于比较硬路由、完整软路由、Top-2 路由和置信度加权 Top-2 路由。三个风格域由源域图像的 `feat1` 风格向量经 K-means 得到，并复用已经训练好的 `Style_0/1/2` 单域专家。

## 核心流程

```text
图像 -> 离线聚类距离 -> 风格路由权重
                         ├─ Style_0 专家 + 判别器 D0
Student feat1 ----------├─ Style_1 专家 + 判别器 D1
                         └─ Style_2 专家 + 判别器 D2
        检测损失 + 加权对抗损失 + 原有 Triplet 损失
```

聚类和对抗统一使用 `feat1_80x80`（YOLOv8-s 下为 `128×80×80`）。专家始终冻结；路由权重只作用于风格对抗损失，不降低检测损失。

## 四种实验模式

在 [`dg_config.py`](dg_config.py) 中修改 `STYLE_EXPERIMENT_NAME`：

| 实验 | 配置值 | 路由方式 |
|---|---|---|
| E2 | `style_hard` | 原硬标签 one-hot，1 个分支 |
| E3 | `exp3_soft_dense` | 完整软概率，3 个分支 |
| E4 | `exp4_soft_top2` | 最大两个概率重新归一化 |
| E5 | `exp5_soft_top2_conf` | Top-2 × 熵置信度 |

主消融阶段建议固定：

```python
STYLE_SOFT_TEMPERATURE = 1.0
STYLE_CONFIDENCE_FLOOR = 0.2
STYLE_CONFIDENCE_POWER = 1.0
STYLE_ADV_LOSS_WEIGHT = 0.7
STYLE_ADV_WARMUP_ENABLED = False
```

先依次运行 E2–E5，再对最佳模式扫描温度、置信度和对抗权重。详细建议已写在 `dg_config.py` 中。

## 快速开始

环境使用现有 `yolo_dg`：

```bash
cd /data4/czh/yolov8-cluster/yolov8-hard-clustering
/data4/czh/anaconda3/envs/yolo_dg/bin/python train_style_adv.py
```

训练前检查：

1. 在 `train_style_adv.py` 中确认 `CUDA_VISIBLE_DEVICES`。
2. 确认 `260708_style_train.txt` 和 `260708_style_val.txt` 路径有效。
3. 确认三个专家位于 `results/Pre_train/Style_0~2/best_epoch_weights.pth`。
4. 启动摘要中的训练/验证硬标签一致率必须为 `1.0`。

当前训练入口默认使用 640 输入、100 epochs、batch size 4，并关闭 Mosaic/MixUp，因为合成图像没有对应的离线聚类距离。

## 聚类缓存

`style_cache/` 中包含：

- `train_style_distances.npz`、`val_style_distances.npz`：路径、硬标签和三中心平方距离；
- `cluster_params.npz`：可移植 scaler 参数和聚类中心；
- `cluster_metadata.json`：距离尺度、专家映射和一致性记录；
- `scaler.pkl`、`kmeans.pkl`：原始聚类对象备份。

如需从原 pretrain 产物重新生成缓存：

```bash
/data4/czh/anaconda3/envs/yolo_dg/bin/python tools/build_style_routing_cache.py
```

该脚本不会重新拟合聚类模型；若软标签与原硬标签不一致会立即终止。

## 输出与日志

输出目录由 `train_style_adv.py` 自动生成，当前根目录为：

```text
logs/Style_soft/260712/<模式>_tau<温度>_floor<下限>_adv<权重>/
```

主要文件：

- `best_epoch_weights.pth`、`last_epoch_weights.pth`：检测器权重；
- `routing_statistics.csv`：概率、熵、置信度、分支质量和分支损失；
- `style_experiment_config.json`：本次实际配置；
- `config_snapshot.py`、`cluster_metadata.json`：配置与聚类快照。

修改输出根目录时，调整 `train_style_adv.py` 中给 `args.results_path` 赋值的位置；命令行 `--results_path` 当前会被该自动目录覆盖。

## 关键文件

| 文件 | 作用 |
|---|---|
| `train_style_adv.py` | 统一训练入口、模型和日志初始化 |
| `dg_config.py` | 模式切换及消融参数 |
| `utils/style_router.py` | hard/dense/Top-2/E5 路由 |
| `utils/style_assignment.py` | 软概率、熵和置信度 |
| `utils/style_adv_trainer.py` | 逐样本加权的对抗损失 |
| `utils/style_cache.py` | 严格路径匹配和缓存校验 |
| `utils/utils_fit.py` | 检测、Triplet 与对抗训练循环 |
| `nets/maddg.py` | 三个风格判别器 |

## 测试与评估

运行路由及反向传播测试：

```bash
/data4/czh/anaconda3/envs/yolo_dg/bin/python -m unittest tests/test_style_router.py -v
```

训练完成后，在 `yolo.py` 中设置 `model_path`，再使用 `get_map.py` 评估。四个主实验必须使用相同测试集、置信度和 NMS 参数。

> 注意：对抗层已由原来的 `feat3_20x20` 改为 `feat1_80x80`，旧硬聚类结果不再是严格同条件基线；E2 需要重新训练后再与 E3/E4/E5 比较。
