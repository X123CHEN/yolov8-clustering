# Style Cluster Split

This tool splits source-domain txt files into pseudo style domains using YOLOv8 shallow feature statistics.

本工具用于把原始国家域 txt 按“图像风格”重新划分成伪域 txt。核心流程是：

```text
国家原始 txt
  -> 读取图片路径
  -> 使用 yolov8_s.pth 的 Backbone 提取浅层 feat1
  -> 对 feat1 计算通道均值和通道标准差
  -> 全部训练样本全局聚类
  -> 输出 Style_0/Style_1/Style_2 对应 txt
```

注意：脚本只生成新的 txt 和统计文件，不移动、不复制、不修改原始图片，也不会修改原始 txt 每一行的图片路径和标注内容。

The original country txt directory in this project is:

```text
/data4/czh/yolov8-cluster/yolov8-pretrain/datasets/RDD2022/ImageSets
```

三国原始 txt 所在目录就是上面这个 `ImageSets` 目录。

Use these source files:

```text
India_train.txt
Japan_train.txt
United_States_train.txt
India_val.txt
Japan_val.txt
United_States_val.txt
```

说明：

- 项目中美国数据文件名是 `United_States_train.txt / United_States_val.txt`，不是 `USA_train.txt / USA_val.txt`。
- `India_train.txt / Japan_train.txt / United_States_train.txt` 是训练集输入。
- `India_val.txt / Japan_val.txt / United_States_val.txt` 是验证集输入。
- 不要把 `251030_train.txt / 251030_val.txt` 作为本任务输入，因为这两个是组合后的总 txt。

Run in the `yolo_dg` environment:

```bash
cd /data4/czh/yolov8-cluster/yolov8-pretrain

/data4/czh/anaconda3/envs/yolo_dg/bin/python tools/style_cluster/split_style_domains.py \
  --train_txts datasets/RDD2022/ImageSets/India_train.txt datasets/RDD2022/ImageSets/Japan_train.txt datasets/RDD2022/ImageSets/United_States_train.txt \
  --train_names India Japan USA \
  --val_txts datasets/RDD2022/ImageSets/India_val.txt datasets/RDD2022/ImageSets/Japan_val.txt datasets/RDD2022/ImageSets/United_States_val.txt \
  --val_names India Japan USA \
  --weights model_data/yolov8_s.pth \
  --phi s \
  --input_shape 640 640 \
  --k 3 \
  --batch_size 8 \
  --device cpu \
  --output_dir pseudo_style_txt \
  --save_features
```

参数中文说明：

- `--train_txts`：三国原始训练 txt 路径，顺序要和 `--train_names` 对齐。
- `--train_names`：输出文件中使用的国家名；这里把 `United_States` 简写成 `USA`，所以会生成 `USA_style0_train.txt` 这类文件。
- `--val_txts`：三国原始验证 txt 路径，可选；提供后会生成 `Style_0_val.txt` 等验证集划分结果。
- `--val_names`：验证集对应国家名，顺序要和 `--val_txts` 对齐。
- `--weights`：固定特征提取器权重，当前使用 `model_data/yolov8_s.pth`。
- `--phi`：YOLOv8 模型规模，当前权重对应 `s`。
- `--input_shape 640 640`：输入图片统一 resize 到 `640x640`。
- `--k 3`：聚类成 3 个风格伪域，输出 `Style_0/Style_1/Style_2`。
- `--batch_size 8`：特征提取 batch size；CPU 太慢或内存不足时可以改小。
- `--device cpu`：当前环境 CUDA 不稳定，建议先用 CPU 跑通。
- `--output_dir pseudo_style_txt`：输出目录。
- `--save_features`：保存中间风格向量 `.npy` 和图片路径顺序文件，便于复现和检查。

主要输出目录：

```text
pseudo_style_txt/
  country_split/      # 每个国家内部按 style 划分后的 txt
  combined_style/     # 最终用于训练风格域专家的 Style_0/1/2 txt
  info/               # 聚类标签、统计、scaler/kmeans 等信息
  features/           # 中间风格特征，只有开启 --save_features 才会保存
```

最重要的最终输出文件是：

```text
pseudo_style_txt/combined_style/Style_0_train.txt
pseudo_style_txt/combined_style/Style_1_train.txt
pseudo_style_txt/combined_style/Style_2_train.txt
```

如果传入验证集，还会生成：

```text
pseudo_style_txt/combined_style/Style_0_val.txt
pseudo_style_txt/combined_style/Style_1_val.txt
pseudo_style_txt/combined_style/Style_2_val.txt
```

The script prefers `sklearn.preprocessing.StandardScaler` and `sklearn.cluster.KMeans`. If sklearn is not installed in the active environment, it uses an internal numpy implementation of StandardScaler and K-means++ so the task can still run in `yolo_dg`.

环境说明：

- `yolo_dg` 当前没有安装 `sklearn`，脚本会自动使用内置 numpy 版 `StandardScaler + KMeans++`。
- 如果后续在 `yolo_dg` 中安装了 `sklearn`，脚本会自动优先使用 sklearn 实现。
- 脚本内部只 patch 本任务运行时的 Backbone 前向，避免 CPU 模式下被原项目里的硬编码 CUDA 调用卡住；不会修改原始训练代码。
