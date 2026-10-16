# 任务文档：基于 YOLOv8 浅层特征的风格伪域划分与 txt 重组

## 1. 任务目标

在现有 `HH276/yolov8-pytorch-master` 项目基础上，实现一个“基于图像级风格特征的伪域划分工具”。

当前原方案按照国家划分源域，例如：

```text
India_train.txt
Japan_train.txt
USA_train.txt
```

现在需要改为：  
先使用 `yolov8_s.pth` 作为固定特征提取器，从每张训练图片中提取 YOLOv8 的浅层特征 `feat1`，即空间尺寸约为 `80×80` 的特征图。然后计算该特征图的通道均值和通道标准差，构成图像级风格向量，再使用 K-means++ 对所有源域训练样本进行全局聚类，得到风格伪域标签。

最终输出：

```text
Style_0_train.txt
Style_1_train.txt
Style_2_train.txt
```

后续这三个文件将分别用于训练三个风格域的单域预训练权重。

---

## 2. 核心要求

### 2.1 不使用组合后的总 txt 作为输入

本任务不要直接读取：

```text
251030_train.txt
251030_val.txt
```

因为后续需要对单个风格域的数据分别训练预训练权重，所以必须先从原始国家单域 txt 文件开始处理。

输入应为：

```text
India_train.txt
Japan_train.txt
USA_train.txt
```

验证集如果存在，则输入为：

```text
India_val.txt
Japan_val.txt
USA_val.txt
```

### 2.2 不改变原始图片和标注

不要移动、复制或修改原始图片。

不要修改 txt 文件中每一行的图片路径和标注信息。

每一行原本是什么格式，输出文件中也必须保持原格式。

例如原始行：

```text
/path/to/image.jpg 10,20,100,200,0 150,160,300,350,2
```

如果这张图片被分到 `Style_1`，则这整行应原样写入：

```text
Style_1_train.txt
```

### 2.3 聚类中心必须全局拟合

虽然输入 txt 按国家分别读取，但是 K-means++ 不能分别在每个国家内部单独拟合。

错误做法：

```text
India_train.txt 单独 K-means
Japan_train.txt 单独 K-means
USA_train.txt 单独 K-means
```

这样会导致 India 的 `Style_0`、Japan 的 `Style_0`、USA 的 `Style_0` 不是同一种风格。

正确做法：

```text
India_train.txt
Japan_train.txt
USA_train.txt
        ↓
分别提取特征
        ↓
在内存中合并所有训练特征
        ↓
全局 StandardScaler fit
        ↓
全局 K-means++ fit
        ↓
再分别回写每个国家内部的风格划分 txt
        ↓
最后组合为 Style_0_train.txt / Style_1_train.txt / Style_2_train.txt
```

---

## 3. 推荐新增文件

建议新增一个脚本：

```text
tools/style_cluster/split_style_domains.py
```

如果项目当前没有 `tools/` 目录，请新建：

```text
tools/
    style_cluster/
        split_style_domains.py
```

同时建议新增一个 README 或说明文件：

```text
tools/style_cluster/README_style_cluster.md
```

可选地新增配置文件：

```text
tools/style_cluster/style_cluster_config.py
```

如果实现更简单，也可以只写一个可通过命令行参数控制的脚本。

---

## 4. 输入参数设计

脚本应支持以下命令行参数：

```bash
python tools/style_cluster/split_style_domains.py \
    --train_txts India_train.txt Japan_train.txt USA_train.txt \
    --train_names India Japan USA \
    --val_txts India_val.txt Japan_val.txt USA_val.txt \
    --val_names India Japan USA \
    --weights model_data/yolov8_s.pth \
    --classes_path model_data/rdd_classes.txt \
    --phi s \
    --input_shape 640 640 \
    --k 3 \
    --batch_size 8 \
    --device cuda \
    --output_dir pseudo_style_txt \
    --save_features
```

其中：

| 参数 | 含义 |
|---|---|
| `--train_txts` | 原始国家训练集 txt 列表 |
| `--train_names` | 对应国家名，顺序必须和 `--train_txts` 一致 |
| `--val_txts` | 原始国家验证集 txt 列表，可选 |
| `--val_names` | 对应验证集国家名，可选 |
| `--weights` | 用于提取特征的 YOLOv8 权重，默认 `model_data/yolov8_s.pth` |
| `--classes_path` | 类别文件路径，按当前项目实际类别文件填写 |
| `--phi` | YOLOv8 模型版本，默认 `s` |
| `--input_shape` | 输入尺寸，默认 `640 640` |
| `--k` | 聚类数量，默认 `3` |
| `--batch_size` | 特征提取 batch size |
| `--device` | `cuda` 或 `cpu` |
| `--output_dir` | 输出目录 |
| `--save_features` | 是否保存中间特征文件 |

注意：  
如果当前项目中的 YOLOv8 模型构建不需要 `classes_path`，则可以不使用该参数；但脚本应尽量兼容当前项目结构。

---

## 5. 输出目录结构

脚本运行后建议生成如下目录：

```text
pseudo_style_txt/
    country_split/
        India_style0_train.txt
        India_style1_train.txt
        India_style2_train.txt

        Japan_style0_train.txt
        Japan_style1_train.txt
        Japan_style2_train.txt

        USA_style0_train.txt
        USA_style1_train.txt
        USA_style2_train.txt

        India_style0_val.txt
        India_style1_val.txt
        India_style2_val.txt

        Japan_style0_val.txt
        Japan_style1_val.txt
        Japan_style2_val.txt

        USA_style0_val.txt
        USA_style1_val.txt
        USA_style2_val.txt

    combined_style/
        Style_0_train.txt
        Style_1_train.txt
        Style_2_train.txt

        Style_0_val.txt
        Style_1_val.txt
        Style_2_val.txt

    info/
        train_pseudo_domain.csv
        val_pseudo_domain.csv
        cluster_summary_train.csv
        cluster_summary_val.csv
        scaler.pkl
        kmeans.pkl

    features/
        India_train_features.npy
        Japan_train_features.npy
        USA_train_features.npy
        all_train_features.npy

        India_val_features.npy
        Japan_val_features.npy
        USA_val_features.npy
        all_val_features.npy
```

其中：

### `country_split/`

保存每个国家内部被分到不同风格域的数据。

例如：

```text
India_style0_train.txt
India_style1_train.txt
India_style2_train.txt
```

这一步用于检查 India 内部样本被如何重新划分。

### `combined_style/`

保存最终用于训练单风格域专家的数据。

例如：

```text
Style_0_train.txt
Style_1_train.txt
Style_2_train.txt
```

其中：

```text
Style_0_train.txt =
India_style0_train.txt
+ Japan_style0_train.txt
+ USA_style0_train.txt
```

### `info/`

保存聚类信息、映射关系和统计结果。

### `features/`

保存中间风格向量，便于后续复现和分析。

---

## 6. 具体实现流程

### Step 1：读取原始国家 txt 文件

读取：

```text
India_train.txt
Japan_train.txt
USA_train.txt
```

对于每一行，保留：

```python
{
    "country": "India",
    "image_path": "/path/to/image.jpg",
    "line": "/path/to/image.jpg 10,20,100,200,0 ..."
}
```

解析规则：

```python
image_path = line.strip().split()[0]
line = line.strip()
```

不要改动 `line` 的内容。

如果某一行为空，跳过。

如果图片路径不存在，记录 warning，并跳过该样本，不要让整个脚本崩溃。

---

### Step 2：加载 YOLOv8 模型和 `yolov8_s.pth`

使用当前项目中的 YOLOv8 网络定义加载模型。

要求：

```python
model.eval()
```

并在特征提取时使用：

```python
with torch.no_grad():
    ...
```

权重路径默认：

```text
model_data/yolov8_s.pth
```

如果权重加载存在 key 不完全匹配的情况，允许使用 `strict=False`，但必须在日志中打印缺失和多余的 key。

---

### Step 3：提取 `feat1(80×80)`

目标是提取 YOLOv8 backbone 或 neck 输出的浅层特征 `feat1`。

对于输入：

```text
640×640
```

期望 `feat1` 的空间尺寸约为：

```text
80×80
```

如果当前项目的 backbone forward 返回：

```python
feat1, feat2, feat3 = model.backbone(images)
```

则直接使用：

```python
F = feat1
```

如果当前模型结构中命名不同，请 Codex 自动检查 `nets/` 目录下的模型定义，找到三个尺度特征输出中空间尺寸最大的那个，即 `80×80` 特征层，将其作为 `feat1`。

如果无法直接从原 forward 获取 `feat1`，可通过以下方式之一实现：

1. 修改或封装 backbone forward，使其返回三个尺度特征；
2. 使用 forward hook 捕获空间尺寸为 `80×80` 的特征图；
3. 新增一个 `FeatureExtractor` 包装类，只用于本任务，不影响原训练代码。

要求不要破坏原始训练和预测流程。

---

### Step 4：图像预处理

输入图片统一处理为：

```text
640×640
```

优先复用项目已有的图像预处理函数，例如：

```python
cvtColor
resize_image
preprocess_input
```

如果项目中已有训练或预测预处理流程，应尽量保持一致。

预处理后张量形状应为：

```python
[B, 3, 640, 640]
```

---

### Step 5：计算图像级风格向量

假设提取到的特征为：

```python
feat1.shape = [B, C, H, W]
```

其中理论上：

```python
H = 80
W = 80
```

对每张图计算通道均值和通道标准差：

```python
mean = feat1.mean(dim=[2, 3])
std = feat1.std(dim=[2, 3])
style = torch.cat([mean, std], dim=1)
```

得到：

```python
style.shape = [B, 2C]
```

每张图对应一个风格向量。

如果实际 `feat1` 的尺寸不是 `80×80`，但它是三个尺度特征中空间尺寸最大的浅层特征，也可以接受；但脚本必须打印实际特征 shape，例如：

```text
Using feature map shape: [B, C, 80, 80]
```

---

### Step 6：分别保存国家训练集特征

分别保存：

```text
features/India_train_features.npy
features/Japan_train_features.npy
features/USA_train_features.npy
```

同时保存对应的图片路径顺序：

```text
features/India_train_paths.txt
features/Japan_train_paths.txt
features/USA_train_paths.txt
```

确保 `.npy` 特征顺序和 paths 文件顺序完全一致。

---

### Step 7：合并所有训练特征

在内存中合并：

```python
all_train_features = np.concatenate([
    India_train_features,
    Japan_train_features,
    USA_train_features
], axis=0)
```

保存：

```text
features/all_train_features.npy
```

注意：  
这里的合并只是为了全局拟合 `StandardScaler` 和 `KMeans`，不是使用 `251030_train.txt`。

---

### Step 8：全局标准化

使用 `sklearn.preprocessing.StandardScaler`：

```python
scaler = StandardScaler()
all_train_features_norm = scaler.fit_transform(all_train_features)
```

保存：

```text
info/scaler.pkl
```

不要对每个国家单独 fit scaler。

原因：  
如果每个国家单独标准化，会削弱国家间和风格间的真实特征差异。

---

### Step 9：全局 K-means++ 聚类

使用 `sklearn.cluster.KMeans`：

```python
kmeans = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=10,
    random_state=0
)
labels = kmeans.fit_predict(all_train_features_norm)
```

默认：

```python
K = 3
```

保存：

```text
info/kmeans.pkl
```

要求：

1. 使用全体源域训练特征拟合；
2. 不要对每个国家分别拟合；
3. 设置固定 `random_state`，保证可复现；
4. 聚类标签为 `0, 1, 2`，对应 `Style_0, Style_1, Style_2`。

---

### Step 10：把全局 labels 映射回每个国家

因为合并特征时保留了样本顺序，所以需要将 `labels` 切回各国家。

例如：

```python
n_india = len(India_train_features)
n_japan = len(Japan_train_features)
n_usa = len(USA_train_features)

India_labels = labels[:n_india]
Japan_labels = labels[n_india:n_india+n_japan]
USA_labels = labels[n_india+n_japan:]
```

然后为每张图片生成记录：

```csv
image_path,country,pseudo_domain,split
/path/to/india_001.jpg,India,0,train
/path/to/japan_001.jpg,Japan,2,train
/path/to/usa_001.jpg,USA,1,train
```

保存为：

```text
info/train_pseudo_domain.csv
```

---

### Step 11：生成国家内部风格划分 txt

对每个国家分别生成：

```text
country_split/India_style0_train.txt
country_split/India_style1_train.txt
country_split/India_style2_train.txt

country_split/Japan_style0_train.txt
country_split/Japan_style1_train.txt
country_split/Japan_style2_train.txt

country_split/USA_style0_train.txt
country_split/USA_style1_train.txt
country_split/USA_style2_train.txt
```

写入规则：

如果某个样本的 `pseudo_domain = 0`，则把它的原始整行写入：

```text
<country>_style0_train.txt
```

如果 `pseudo_domain = 1`，则写入：

```text
<country>_style1_train.txt
```

如果 `pseudo_domain = 2`，则写入：

```text
<country>_style2_train.txt
```

必须保持每一行原样。

---

### Step 12：组合相同风格域的数据

组合规则：

```text
Style_0_train.txt =
India_style0_train.txt
+ Japan_style0_train.txt
+ USA_style0_train.txt
```

```text
Style_1_train.txt =
India_style1_train.txt
+ Japan_style1_train.txt
+ USA_style1_train.txt
```

```text
Style_2_train.txt =
India_style2_train.txt
+ Japan_style2_train.txt
+ USA_style2_train.txt
```

输出到：

```text
combined_style/Style_0_train.txt
combined_style/Style_1_train.txt
combined_style/Style_2_train.txt
```

这三个文件是后续训练单风格域预训练权重的主要输入。

---

## 7. 验证集处理规则

如果提供了：

```text
India_val.txt
Japan_val.txt
USA_val.txt
```

则也需要划分验证集。

但是验证集不能重新 fit `StandardScaler`，也不能重新 fit `KMeans`。

正确流程：

```python
val_features_norm = scaler.transform(val_features)
val_labels = kmeans.predict(val_features_norm)
```

然后同样生成：

```text
country_split/India_style0_val.txt
country_split/India_style1_val.txt
country_split/India_style2_val.txt

country_split/Japan_style0_val.txt
country_split/Japan_style1_val.txt
country_split/Japan_style2_val.txt

country_split/USA_style0_val.txt
country_split/USA_style1_val.txt
country_split/USA_style2_val.txt
```

再组合：

```text
combined_style/Style_0_val.txt
combined_style/Style_1_val.txt
combined_style/Style_2_val.txt
```

同时保存：

```text
info/val_pseudo_domain.csv
```

---

## 8. 统计信息与检查

脚本运行完成后，必须输出聚类统计信息。

### 8.1 每个 Style 的样本数量

生成：

```text
info/cluster_summary_train.csv
```

内容示例：

```csv
pseudo_domain,total,India,Japan,USA
Style_0,4200,1200,1800,1200
Style_1,3900,900,1600,1400
Style_2,5000,500,2500,2000
```

如果验证集存在，也生成：

```text
info/cluster_summary_val.csv
```

### 8.2 控制台打印 summary

运行结束时打印：

```text
===== Train Cluster Summary =====
Style_0: total=4200, India=1200, Japan=1800, USA=1200
Style_1: total=3900, India=900, Japan=1600, USA=1400
Style_2: total=5000, India=500, Japan=2500, USA=2000

Output files:
pseudo_style_txt/combined_style/Style_0_train.txt
pseudo_style_txt/combined_style/Style_1_train.txt
pseudo_style_txt/combined_style/Style_2_train.txt
```

### 8.3 异常提醒

如果某个 Style 的样本数过少，例如低于总样本的 5%，打印 warning：

```text
WARNING: Style_2 contains only 3.8% of training samples. K may be too large or features may be unstable.
```

如果某个国家几乎全部进入同一个 Style，例如超过 90%，打印 warning：

```text
WARNING: 92% of India samples are assigned to Style_0. Please check whether clustering degenerates to country/domain-specific bias.
```

---

## 9. 可选：类别分布统计

如果 txt 标注格式为：

```text
image_path x1,y1,x2,y2,class x1,y1,x2,y2,class ...
```

则可以解析每个 bbox 的类别 id，统计每个 Style 内的类别分布。

输出：

```text
info/class_distribution_train.csv
```

示例：

```csv
pseudo_domain,class_0,class_1,class_2,class_3
Style_0,1200,800,300,600
Style_1,900,700,500,400
Style_2,1500,600,200,700
```

如果实现时间有限，这一步可以作为可选功能，但推荐实现。

---

## 10. 训练阶段如何使用输出结果

本任务只负责生成风格划分 txt，不需要真正训练模型。

后续训练逻辑如下：

原来按国家训练单域权重：

```text
India_train.txt  → India_pretrained.pth
Japan_train.txt  → Japan_pretrained.pth
USA_train.txt    → USA_pretrained.pth
```

现在改为按风格域训练单域权重：

```text
Style_0_train.txt → Style_0_pretrained.pth
Style_1_train.txt → Style_1_pretrained.pth
Style_2_train.txt → Style_2_pretrained.pth
```

如果训练脚本需要 train/val 对应关系，则使用：

```text
Style_0_train.txt + Style_0_val.txt
Style_1_train.txt + Style_1_val.txt
Style_2_train.txt + Style_2_val.txt
```

后续对抗训练阶段中，原来的国家域标签：

```text
India / Japan / USA
```

应替换为：

```text
Style_0 / Style_1 / Style_2
```

---

## 11. 重要实现约束

1. 不要使用 `251030_train.txt` 和 `251030_val.txt` 作为本脚本输入。
2. 输入必须从原始国家单域 txt 开始。
3. 不要移动或复制原始图片。
4. 不要修改原始标注格式。
5. 每一行输出时必须保持原样。
6. K-means++ 必须在所有源域训练特征上全局拟合。
7. 验证集只能使用训练阶段保存的 scaler 和 kmeans 进行 `transform` 和 `predict`。
8. 默认提取 `feat1`，即输入 `640×640` 时空间尺寸约为 `80×80` 的浅层特征。
9. 默认风格向量为：

```python
style = concat(channel_mean(feat1), channel_std(feat1))
```

10. 默认聚类数量：

```python
K = 3
```

11. 默认使用：

```python
StandardScaler + KMeans(init="k-means++")
```

12. 保证脚本可复现，固定随机种子：

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
KMeans(random_state=0)
```

---

## 12. 建议代码结构

可以将脚本拆成以下函数：

```python
def parse_txt(txt_path, country_name):
    """
    读取 txt 文件，返回样本列表。
    每个样本包含 country、image_path、line。
    """
```

```python
def build_model(weights, phi, num_classes, device):
    """
    构建 YOLOv8 模型，加载 yolov8_s.pth 权重。
    """
```

```python
def preprocess_image(image_path, input_shape):
    """
    读取图片并进行 resize/preprocess，返回 tensor。
    """
```

```python
def extract_feat1(model, images):
    """
    提取 feat1，即 80x80 浅层特征。
    如果原模型无法直接返回 feat1，则使用 hook 或包装类。
    """
```

```python
def compute_style_vector(feat1):
    """
    feat1: [B, C, H, W]
    return: [B, 2C]
    """
```

```python
def extract_features_for_samples(samples, model, batch_size, device):
    """
    对样本列表提取风格向量。
    返回 features 和有效样本列表。
    """
```

```python
def fit_cluster(all_train_features, k):
    """
    fit StandardScaler 和 KMeans++。
    返回 scaler, kmeans, train_labels。
    """
```

```python
def write_country_split_files(samples, labels, output_dir, split, k):
    """
    写出 India_style0_train.txt 等国家内部风格划分文件。
    """
```

```python
def combine_style_files(country_names, output_dir, split, k):
    """
    合并成 Style_0_train.txt 等最终训练文件。
    """
```

```python
def save_pseudo_domain_csv(samples, labels, output_path, split):
    """
    保存 image_path,country,pseudo_domain,split 映射。
    """
```

```python
def save_cluster_summary(samples, labels, output_path, k):
    """
    保存每个风格域的国家来源统计。
    """
```

---

## 13. 运行示例

训练集划分：

```bash
python tools/style_cluster/split_style_domains.py \
    --train_txts India_train.txt Japan_train.txt USA_train.txt \
    --train_names India Japan USA \
    --weights model_data/yolov8_s.pth \
    --phi s \
    --input_shape 640 640 \
    --k 3 \
    --batch_size 8 \
    --device cuda \
    --output_dir pseudo_style_txt \
    --save_features
```

训练集 + 验证集划分：

```bash
python tools/style_cluster/split_style_domains.py \
    --train_txts India_train.txt Japan_train.txt USA_train.txt \
    --train_names India Japan USA \
    --val_txts India_val.txt Japan_val.txt USA_val.txt \
    --val_names India Japan USA \
    --weights model_data/yolov8_s.pth \
    --phi s \
    --input_shape 640 640 \
    --k 3 \
    --batch_size 8 \
    --device cuda \
    --output_dir pseudo_style_txt \
    --save_features
```

---

## 14. 验收标准

任务完成后，应满足以下条件：

### 14.1 文件生成正确

至少生成：

```text
pseudo_style_txt/combined_style/Style_0_train.txt
pseudo_style_txt/combined_style/Style_1_train.txt
pseudo_style_txt/combined_style/Style_2_train.txt
```

如果提供验证集，还应生成：

```text
pseudo_style_txt/combined_style/Style_0_val.txt
pseudo_style_txt/combined_style/Style_1_val.txt
pseudo_style_txt/combined_style/Style_2_val.txt
```

### 14.2 行数守恒

对于训练集：

```text
len(India_train.txt) + len(Japan_train.txt) + len(USA_train.txt)
=
len(Style_0_train.txt) + len(Style_1_train.txt) + len(Style_2_train.txt)
```

如果有图片路径不存在并被跳过，需要在日志中说明跳过数量。

### 14.3 行内容不变

输出 txt 中每一行必须来自原始 txt，不能改变图片路径和 bbox 标注。

### 14.4 聚类信息可追踪

必须生成：

```text
info/train_pseudo_domain.csv
info/cluster_summary_train.csv
info/scaler.pkl
info/kmeans.pkl
```

### 14.5 验证集不重新拟合

如果处理验证集，验证集必须使用训练集 fit 好的 `scaler.pkl` 和 `kmeans.pkl` 进行预测。

### 14.6 不影响原项目训练代码

新增脚本不应破坏原项目的 `train.py`、`predict.py`、`yolo.py` 等原有功能。

如果必须修改模型 forward 以获取 `feat1`，应尽量通过新增包装类或可选参数实现，避免影响原训练流程。

---

## 15. 后续扩展项

第一版只需要实现：

```text
feat1 → mean/std → StandardScaler → K-means++ → Style_0/1/2 txt
```

后续可以扩展：

1. 支持 `feat2`；
2. 支持 `feat1 + feat2` 拼接；
3. 支持 PCA 降维；
4. 支持 UMAP/t-SNE 可视化；
5. 支持类别分布统计；
6. 支持不同 K 值消融，例如 `K=4,5,6`；
7. 支持使用源域 baseline 权重代替 `yolov8_s.pth` 提取特征。

第一版先不要加入过多功能，优先保证流程跑通、输出文件正确、聚类结果可追踪。
