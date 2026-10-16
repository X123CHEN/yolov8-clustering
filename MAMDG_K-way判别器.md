# MAMDG 潜在风格域的单 K 类判别器：实验方案与 Codex 实施规范

> 版本：1.0 · 2026-10-14  
> 阶段定位：**固定离线硬聚类 K=2 → 单个 K 类域判别器（Hard CE + GRL） → 保留 MAMDG 检测与 FAB/Triplet → 对照实验和机制检查。**  
> 此文件是**实验设计/代码实施规范**，不是已经完成编码、跑通实验或验证效果的声明。所有仓库内路径、参数名称和模型张量维度应以 Codex 实际读取的本地源码为准。绝不能以猜测代替核实。

## 0. 核心结论：哪些不准改

1. 训练集只使用 MAMDG 原先三个国家：Japan、India、United States（JP/IN/US）；未见国家/地区 Norway、Czech、China-Drone、China-MotorBike 只能用于最终评测，不能参与聚类拟合、网络训练、调参或选 epoch。
2. **聚类已完成，不重选 K**：沿用第一阶段基于 YOLOv8 `feat1`（约 80×80，具体按原代码）通道 μ/σ 风格特征、StandardScaler、K-Means++ 得到的 **K=2** 离线簇及固定硬标签 `0,1`。沿用原图像 ID、预处理和聚类 artifacts；不可重新聚类，也不可用国家域标签替代它。
3. **域对抗结构替换为单一的 K-way 判别器**：仅一个共享判别器，输出 `logits.shape=[B,2]`；将聚类簇 ID 用作 CE 的目标，GRL 将梯度反向传给共享 Student/YOLOv8 特征提取器。
4. 保留 MAMDG 的检测任务结构和 loss；**保留原 FAB/Triplet** 的采样、类别映射、损失、超参数与作用位置。不能擅自换成 SupCon、重新设计三元组或增加类别对比模块。FAB 实际是否生效需要在本地用日志/梯度检查确认。
5. 此阶段只做**硬域监督**，不加离线 soft q、聚类温度 τ、EMA teacher、周期重聚类、在线中心更新或动态路由。这些属于后续实验，不能混入 K-way Hard CE 的对照。
6. 不改原始源/目标域数据划分、图像尺寸、YOLOv8 检测头、目标标签和评价口径。实验新代码尽量放独立模块，通过模式开关接入；保留可完全复现的原方法。
7. 不能因为替换判别器而无报告地删除 FAB 所依赖的任何模块。若旧专家网络仅参与域对抗则从新模式中停止使用；若 FAB 必须使用某个旧模块的特征，先审计依赖并报告，保持 FAB 计算语义一致，而不是悄悄破坏度量约束。

## 1. 研究问题与可检验假设

**核心问题**：MAMDG 基于国家域进行多判别器对抗，而实际造成分布偏移的风格属性可能跨越国家边界。将三国家训练样本重分为两个风格域后，一个 K 类域判别器是否能够学习更有用的域不变表示，并保持病害类别的可分辨性？

- H1（伪域有效性）：风格聚类不是简单国家分组；应通过簇×国家及簇×病害类别统计说明。
- H2（判别器确实起作用）：去掉 GRL 时域判别器能够识别固定伪域；加 GRL 后 Student 域信息减少，且病害检测指标不过度下降。
- H3（效果）：在相同训练条件、保留 FAB 的前提下，`Latent K-way (K=2)` 在四个未见目标域 mAP@0.5 平均值上不低于、最好优于 `Latent Multi`；比较 `Country Multi` 和 `Country K-way (K=3)` 以区别“换域标签”与“换域判别器”的影响。
- H4（非平凡对齐）：不是通过删除病害相关特征、域类别坍缩或类别混淆取得低域分类准确率。

> **注意**：本研究不假定 H1–H4 成立。域对抗带来的目标域提升尚待实验验证；多分类判别器并不保证逐对源域距离精确最小化。

## 2. 文献与代码的可核查支撑

### 2.1 原工作

- Chen et al., *Multi-Adversarial Metric-Constrained Domain Generalization for Road Damage Detection*, *Image and Vision Computing*, 2026, DOI: https://doi.org/10.1016/j.imavis.2026.106173 。论文公开的代码入口：https://github.com/envision30/MAMDG 。文献支持原多对抗+FAB；**本方案未能在线核验到该仓库的具体类名/函数名，Codex 必须在本地完成结构审计**。
- Matsuura & Harada, *Domain Generalization Using a Mixture of Multiple Latent Domains*, AAAI 2020. 论文：https://ojs.aaai.org/index.php/AAAI/article/view/6846 ；PDF：https://ojs.aaai.org/index.php/AAAI/article/download/6846/6700 ；作者代码：https://github.com/mil-tokyo/dg_mmld 。原方法利用风格聚类发现潜在域，训练多分类域判别器；原仓库存在 `clustering/`,`model/`,`loss/`,`train/`,`script/` 等目录。**其反复聚类设置不在本轮照搬**。
- Sicilia, Zhao & Hwang, *Domain adversarial neural networks for domain generalization: when it works and how to improve*, *Machine Learning* 112, 2685–2721 (2023). https://link.springer.com/article/10.1007/s10994-023-06324-x 。第 2.2 节（7）（8）明确使用 K 类源域 CE 和源-源对抗形式；附录 B 给出 `FC(1024)-ReLU-Dropout(0.5)-FC(1024)-ReLU-Dropout(0.5)-FC(K)` 这一适用于其图像分类任务的判别器设计，并采用 DANN 渐进 GRL 强度。**K-way 只是多源域对齐的实用代理，不等于严格最小化所有源域两两差异**。
- Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 2016. PDF：https://jmlr.org/papers/volume17/15-239/15-239.pdf 。GRL 的基本理论与实现。

### 2.2 代码借鉴边界

- `dg_mmld` 的 *思想* 可借鉴：固定/估计域标签→域分类 CE→GRL 对抗；但其 Python/PyTorch 和分类网络结构较老，**不能将旧依赖、PACS 数据接口或在线聚类步骤直接嫁接到 YOLOv8**。
- 尽可能复用 MAMDG 现有特征连接点、优化器、训练策略和现有判别器 MLP 参数；只有原结构无法复用时才实现轻量共享判别器。
- 仅从作者仓库源码核实后才写出具体文件名、函数名或精确层尺寸，不得在报告中声称未见过的代码细节来自原论文。

## 3. 具体数据与标签契约

### 3.1 离线 K=2 风格伪域

- 以现有**已完成**的聚类 artifact 为准：`cluster_id ∈ {0,1}`，图像级别、固定、覆盖所有用于域对抗的源训练样本。
- 原离线流程：固定的 YOLOv8 特征提取器在约 80×80 的 `feat1` 输出上按通道提取 `μ(F), σ(F)`，拼接，再使用已拟合 StandardScaler 与 K-Means++；**不重新拟合**。
- 如果 artifact 附带 `q`、centroid distances 或 scaler 的对象，硬基线只读取簇 ID，**不能用 q 代替 hard label**。
- 必须使用无歧义 `image_id` 或规范化的 `source_root_relative_path` 将聚类结果关联到训练图片，不能按 DataLoader 的当前顺序、batch index、临时排序、文件 basename 或国家名匹配。
- 重复采样或多个 worker 中相同图片仍应有相同的簇标签；`Unique/Repeat` 模式分别匹配既有协议，避免重复图像给簇比例造成无意的重新加权。
- 检查：样本覆盖率应为 100%；簇 ID 精确为 {0,1}；各簇有有效样本；每个训练 epoch 随机抽样核对 `image_id`、源国家、簇 ID、路径。

### 3.2 数据增强的硬标签有效性

域标签是**整张原始图像**的风格标签，若使用 Mosaic、MixUp 等跨图像混合增强，一张训练张量可能包含来自多个不同簇的图片，不能未经论证直接赋予单一硬簇 ID。

按以下优先级处理：

1. 首先审计原训练中的 Mosaic、MixUp、CopyPaste 等是否开启、返回什么图像来源元数据，以及单图增强（颜色扰动等）是否改变簇可识别性。
2. 如果批数据仍对应单一原图像，保持现有增强和 hard label，并正常计算域 CE。
3. 若存在跨来源混合图像，默认**不对这些混合样本计算 hard 域 CE**，但仍允许其参与原始检测损失及 FAB（前提是原代码如此）。对非混合样本计算域损失；必须记录有效域监督比率。
4. 如果几乎全部输入是混合样本、导致第 3 项无法产生足够域对抗训练信号，先停下做代码/实验方案审计。可以构造**成对且完全同样关闭跨图混合的两组新对照实验**，并明确它们不再是对原始 MAMDG 训练增强完全相同的比较；不得只给新方法关闭 Mosaic 而给旧方法保留。
5. 不在 hard-only 实验中把混合标签自动替换成 soft q；那会改变本轮研究问题。

### 3.3 未见目标域

Norway / Czech / China-Drone / China-MotorBike **不能**出现在：K-Means fit、StandardScaler fit、domain CE、检测训练、伪域权重计算、阈值选择、超参数/温度选择、早停与 best checkpoint 选择之中。目标域只执行最终推理及固定评估。

## 4. 新模型结构（硬约束）

```
JP/IN/US images + original detection labels + fixed pseudo-domain cluster_id
        │
        ▼
 shared MAMDG / YOLOv8 Student feature extractor Fθ
        ├──> 原检测头及原检测损失 L_det（不变）
        ├──> 原 FAB / Triplet 计算 L_FAB（不变）
        └──> 域判别特征 tap z（带梯度、训练态）
                    │
             [必要时 GAP/投影]
                    │
               GRL(r(p))
                    │
           一个 DomainDiscriminator
                    │
          logits [B, 2] (raw logits)
                    │
 CrossEntropy(logits, cluster_id)
```

### 4.1 特征连接位置

**第一优先：复用 MAMDG 原对抗支路所用的 Student 共享特征层**。必须在代码审计中明确该特征的张量 shape、梯度回传路线及与 FAB 的关系。

- 聚类所用 `feat1` μ/σ 是获得 **伪域标签** 的参考表示；域判别器输入应来自**当前可训练 Student 的特征**，而不是聚类缓存的静态 μ/σ。这保证 `L_domain` 能改变 Student。
- 若原域支路使用多个 feature map，首先确定原结构的设计；若确实无法保留完全一致的连接接口，使用固定的共享特征输入（例如只用原主域支路的 P3 特征并做 GAP），作为一次**明确记录的工程假设**；P3/P4 拼接只能成为单独标识的消融，不能冒充最小替换。
- 若对抗输入包含不同批次/国家的 expert 特征，必须明确 Student 梯度是否可达；新模块以 **共享 Student 特征** 为准，不依赖先按国家分流。
- 不把原国家标签、cluster ID、中心距离、风格向量直接拼进判别器输入；这些只能充当监督标签或用于分析。
- 如果使用 `AdaptiveAvgPool2d(1)`，对输入 `[B,C,H,W]` 转 `[B,C]`；若原域支路已输出 `[B,D]`，直接利用，不能重复池化。

### 4.2 判别器 MLP

优先：保留原 MAMDG 单个判别器的隐藏层设置/Dropout/初始化，将最后输出维度设为 K=2，其他并列判别器从新模式的梯度图及优化器中移除。

若原模型不可复用，可使用参数透明的轻量结构（**推荐工程默认，不是论文证明的最佳超参数**）：

```
Input feature z[B,D]
LayerNorm(D) [可选；原判别器有 BN/LN 时优先原样保持]
Linear(D, min(256,D))
ReLU
Dropout(p=0.5)
Linear(min(256,D), K=2)   # raw logits，不使用 sigmoid/softmax
```

- `K` 作为配置变量，禁止在代码逻辑中写死“只支持两类”，实际本次设 2。
- 不额外增加隐性专家、多个 binary head、one-vs-rest heads、pairwise discriminators。
- Softmax 只用于统计预测概率；训练把**原始 logits**交给 `torch.nn.functional.cross_entropy`。
- 共享 Student 和判别器参数都进入当前训练优化器与 checkpoint；权重恢复支持原模型不存在新 head 的情况（采用明确的新实验初始化策略并记录）。

### 4.3 具体可参考的 PyTorch 骨架（Codex 自行映射本地接口）

```python
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coeff * grad_output, None


def grl(x, coeff):
    return _GradientReverse.apply(x, coeff)


class KWayDomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, k: int = 2, hidden_dim: int = 256,
                 dropout: float = 0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, k),
        )

    def forward(self, student_feature, grl_coeff: float):
        z = student_feature
        if z.ndim == 4:
            z = self.pool(z).flatten(1)
        assert z.ndim == 2, f'Unexpected domain feature shape: {z.shape}'
        z = grl(z, grl_coeff)
        return self.net(z)  # B × K, logits


def kway_hard_ce(logits, cluster_ids, valid_mask=None, per_sample_weight=None):
    assert logits.shape[0] == cluster_ids.numel()
    assert cluster_ids.dtype == torch.long
    assert logits.shape[1] >= 2
    per_example = F.cross_entropy(logits, cluster_ids, reduction='none')
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        per_example = per_example[valid_mask]
        if per_sample_weight is not None:
            per_sample_weight = per_sample_weight[valid_mask]
    if per_example.numel() == 0:
        return logits.sum() * 0.0  # 此 batch 无合格单来源样本
    if per_sample_weight is None:
        return per_example.mean()
    w = per_sample_weight.float().to(logits.device)
    return (per_example * w).sum() / w.sum().clamp_min(1e-12)
```

**关键审计**：如果原 MAMDG 采用判别器交替更新而非 GRL，在新模式统一采用上述 GRL 同步对抗需明确记录；避免原代码已经反号又继续接 GRL，从而发生“双重反转”。

## 5. 优化目标与梯度定义

令第 i 张图像的固定伪域为 `c_i ∈ {0,1}`，学生特征为 `z_i = Fθ(x_i)`，单 K 类判别器为 `Dφ`：

\[
L_{dom}= -\frac{1}{B_{valid}}\sum_{i\in valid}\log \mathrm{softmax}(D_\phi(z_i))_{c_i}.
\]

保留现有的：

\[
L_{main}=L_{det}+\beta_{orig} L_{FAB}.
\]

GRL 路径下代码执行：

\[
L_{code}=L_{det}+\beta_{orig} L_{FAB}+\alpha_{dom}L_{dom}(D_\phi(\mathrm{GRL}_{r(p)}(z))).
\]

因此 `Dφ` 最小化域 CE，而 Student 的域梯度被反向；在域分支上 Student 实际接收 `-alpha_dom * r(p) * ∂L_dom/∂z`。不可再人为给 `L_dom` 加负号。

### 5.1 GRL 调度

用论文常见 DANN 进度调度作为**可复现起点**：

\[
r(p)=2/(1+\exp(-10p))-1,\qquad p=global\_step/max\_train\_steps.
\]

- `r(0)=0`，随训练增至约 1；使用 global optimizer step 而非仅每 epoch 更新。
- 如果本地原多对抗已存在特定 warm-up/scheduler，必须先审计：**优先保留原论文调度与训练轮数**，将 DANN schedule 作为明确单独的配置值或消融，不得暗中同时更换损失权重与调度。
- `alpha_dom` 和 `grl_coeff` 分开；`alpha_dom` 仅控制 CE 分支幅度，`grl_coeff` 只翻转返回 Student 的梯度；不能重复乘同一 schedule。

### 5.2 K=2 样本均衡（分为主试验与消融）

- 首选主实验：自然采样下**不加额外簇权重**，与原数据加载及检测损失的权重保持一致。计算 CE 时对有效样本取 mean。
- 若两个簇极不平衡，辅以单独的 `cluster_balanced_ce` 消融：根据**源训练集**簇计数 `n_k` 计算 `w_k ∝ 1/n_k`，并将 `(CE * w).sum()/w.sum()`，与原论文 MMLD 的类别/簇加权思路相近；不改检测采样分布。
- 不用按国家等权去篡改风格域标签，也不拿目标域数据估计 `w_k`。
- 记录随机分类参考准确率 `max(pi_0,pi_1)`、均衡准确率 50%、自然先验熵 `H(C)`；不平衡时不能将 `accuracy≈50%` 误认为“达到域不变”。

### 5.3 损失尺度选择与公平性

- 不凭空固定 `alpha_dom=1` 就认定与原“多个域损失求和/求均值”有可比性。**Codex 必须先审计原域损失聚合规则**、GRL/反号位置及原 loss 权重。
- 首个 pilot 采用原 MAMDG 中域对抗损失的有效权重定义作为起点；如定义无法直接迁移，通过相同源训练 mini-batch 上域分支对共享特征的梯度范数核对量级，记录 K-way / 原多判别器的域梯度比值。
- 若需探索强度，只用**源域验证集**选择少量预先指定候选，例如有效域梯度倍率 `{0.5, 1.0, 2.0}`，其余参数固定。不根据 Norway/Czech/China 目标域成绩回调。
- 不强行要求单头和多头参数数量完全相同，但需报告 head 参数量、额外 FLOPs/时延、梯度范数及损失形式；对性能增益的归因要谨慎。

## 6. 代码接入计划（先审计，再改动）

### 阶段 A｜只读代码审计（禁止改代码）

输出 `audit_before_change.md`，逐项列出 **真实路径+类名+方法名+张量形状**：

1. YOLOv8/MAMDG 的 Student backbone、neck、P3/P4、原域分支的具体 feature tap。
2. 国家域加载、expert/判别器路由、源域训练 batch 组织形式；是否依赖固定每国 batch 或遍历所有域。
3. 原多判别器层结构与参数量，域损失 BCE/CE、sum/mean、GRL/交替训练与系数、optimizer/scheduler。
4. FAB 三元组来源、类别信息、是否跨域构造、是否需要专家特征；实际训练 loss 对 FAB 是否非零、是否有 Student 梯度。
5. 现有离线聚类 artifact、标签字段、配置文件、簇数和两簇样本数；与数据加载图像 ID 的 join 策略。
6. Mosaic/MixUp/CopyPaste、单图颜色增强、`Unique/Repeat`、validation/test data leaks、AMP、DDP、checkpoint、epoch 选择策略。
7. 当前基线 run 的 config、seed、权重文件、训练集清单 hash。已存在 `hard_kway/soft_kway` 分支时先检查并记录，不能覆盖或以为不存在。

**审计完成后仍然只实施以下主线**，遇到关键矛盾先列问题与最小兼容方案，不得擅自重设计。

### 阶段 B｜新代码（独立目录/可配置开关）

建议逻辑组织，不指定原仓库绝对路径：

```
new_experiments/kway_domain/
  README.md
  discriminator.py       # GRL, KWayDomainDiscriminator
  cluster_index.py       # 已存在聚类 artifact 与 image ID 的严格对齐
  loss.py                # Hard CE + mask + optional balanced weights
  hooks.py               # 现有 Student 特征适配接口，不重写 YOLOv8
  config_example.yaml    # 一切超参可配置
  checks/                # 轻量梯度/标签/shape 自测
  reports/               # 审计、训练质量、结果汇总
```

代码只表达责任划分；Codex 可根据项目真实组织方式调整路径。保留模式切换：`country_multi`、`latent_multi`、`latent_kway_hard`，外加用于对照的 `country_kway` 与 `no_adversarial`。只在 `latent_kway_hard` 模式用已有离线 K=2 簇标签；`country_kway` 使用 JP/IN/US 三类标签并输出三维 logits，用于结构消融，不改变主研究方法。

### 阶段 C｜新实验配置（示例，实际键名由 Codex 映射）

```yaml
experiment:
  name: latent_kway_hard_k2
  base: mamdg
  mode: latent_kway_hard
  seed: 0            # 正式汇总至少3个配对seed
  restore_original_fab_triplet: true
  restore_original_detection: true

latent_domain:
  provenance: first_stage_offline_artifacts
  k: 2
  assignment: hard
  refresh_cluster: false
  refit_scaler: false
  refit_kmeans: false
  lookup: stable_image_id
  on_missing_label: error

adversarial:
  kind: single_kway
  classifier_input: original_student_domain_feature_tap
  input_pool: if_spatial_then_global_average
  classifier_output_dim: 2
  loss: cross_entropy_logits
  domain_loss_reduction: mean_valid_samples
  balanced_ce: false
  grl: true
  grl_schedule: dann_logistic_gamma_10  # 如原方案不同需可切换并单独记录
  alpha_domain: USE_AUDITED_ORIGINAL_EFFECTIVE_SCALE
  cluster_temperature: null

train:
  use_original_dataset_split: true
  use_original_imgsz: true
  use_original_optimizer_and_schedule: true
  use_original_augmentations: true
  mixed_source_image_domain_loss: mask_out
  use_target_for_selection: false

report:
  source_domains: [Japan, India, United_States]
  unseen_target_domains: [Norway, Czech, China_Drone, China_MotorBike]
  primary_metric: mean_target_mAP50
  secondary_metric: mean_target_mAP50_95
```

> `USE_AUDITED_ORIGINAL_EFFECTIVE_SCALE` 是一个**审计占位符**，不能直接作为数值传给训练代码。审计完成后填写实际数字、来源与换算说明。不能让 Codex 默认替换为任意值却不报告。

## 7. 单元测试与最小运行验收（必须先于正式训练）

1. **Label join**：`image_id -> cluster_id` 稳定，覆盖率 100%，硬标签全为整数 0/1；重复采样标记正确。
2. **Shape**：用一批实际 YOLOv8 图像检查 `[B,C,H,W] -> [B,D] -> [B,2]`；对 B=1 及常规 B 皆有效。
3. **Loss 与方向**：构造固定 `z`，验证关闭 GRL 时判别器 CE 的普通梯度，打开 GRL 后 `z.grad(GRL)≈-r*z.grad(noGRL)`，而判别器参数梯度方向不翻转。
4. **只接一次 GRL**：断言原多判别器的 domain loss 在新模式不再被加总；`L_domain` 只由一个 classifier 构成，且没有额外 `-L_domain`。
5. **Student 梯度贯通**：域损失单独反传时，Student 共享特征 tap 的上游可训练参数确实有非零梯度；误用 `.detach()` 直接验收失败。
6. **FAB 保真**：同输入、同标签、同初始权重、关闭新域分支时，与原 MAMDG 的 `L_det`, `L_FAB` 及 Student 梯度在允许浮点误差内一致；FAB 不是日志里打印但未参与 backward。
7. **Mix augment**：对混合不同簇的样本禁止直接 hard CE；确保有效域监督比例高于预先审计的可接受水平。
8. **梯度强度**：至少几个稳定训练步记录 `||grad_det||, ||grad_fab||, ||grad_domain||`（相同 tap、相同统计口径），留意 domain 梯度为 0、NaN/Inf、压倒检测梯度等。
9. **DDP/AMP**：若项目使用多卡或混合精度，检查每卡各类别的可见性、mask 样本数、未使用参数与 scaler；所有结果可汇总。
10. **推理独立**：推理时 Student+检测头即可正常产出检测结果；域判别器不得要求目标域标签、cluster scaler 或国家 ID。导出与原评估接口兼容。
11. **两种来源均存在**：为排除 CE 只是学到先验，取一个不经 GRL 的冻结 Student 样本，判别器单独训练时域平衡准确率应显著高于随机基准（若做不到，应先检查标签有效性/接线，而不是增大对抗强度）。
12. **完全重复性**：关掉 `domain_loss_weight` 且关闭新 head 对共享部分的更新后，结果应能在相同随机数/输入下退化为“原检测 + FAB”基线；从 checkpoint 恢复记录所有 optimizer、scheduler、new head 状态。

## 8. 实验组与实施顺序

### 8.1 主对照（不改变研究主线）

| ID | 训练域标签 | 域对抗结构 | FAB/Triplet | 用途 |
|---|---|---|---|---|
| B0 | 国家 JP/IN/US | MAMDG 原多判别器 | 保留 | 原始参照 |
| B1 | K=2 固定潜在风格域 | 原多判别器/专家 | 保留 | 单独验证“换域划分” |
| **E1** | **K=2 固定潜在风格域** | **一个 K 类判别器 + Hard CE + GRL** | **保留** | **本轮核心** |
| C1 | 国家 JP/IN/US | 一个 3 类判别器 | 保留 | 区分“单头结构”与“潜在域标签” |
| C2 | 不使用域标签 | 不使用对抗（α=0） | 保留 | 鉴别域对抗是否有实际贡献 |

保持 seed、采样协议、imgsz、训练轮数、学习率、optimizer、增广、FAB、检测 loss 等一致。B1 有多个专家/判别器而 E1 只有一个时，额外参数量不同，报告计算开销以辅助归因。若曾经已有 `Hard K-way` 实验结果，先审计其实现是否与本方案一致，并用同一套报告输出，不要无意义地覆盖已完成试验。

### 8.2 消融优先级

- 第一优先：B0/B1/E1/C1/C2、相同数据和 seed、FAB 实际生效检查、梯度方向检查；必要时对 E1 使用原权重与半倍/双倍的域有效梯度尺度候选（仅用源验证选值）。
- 第二优先：K=2 簇不均衡时比较 natural CE 与 inverse-frequency weighted CE（其余全相同）；如混合增强大比例丢弃域监督，单独配对关闭跨图混合的 B1/E1。
- 后续另立阶段：E2 Offline Soft K-way（固定聚类中心，只将 `Hard CE` 换成 `Soft CE`，保持 backbone/tap/训练/Triplet 不变）；温度仅属于 E2，必须区分温度调整概率锐度与 GRL 调度。再之后才讨论在线域中心/EMA/重聚类。

### 8.3 训练/评估次数

- 先做 1 seed 的**短跑**：检查数值与梯度，不用于宣传效果；再做 1 seed 的完整 pilot，确认可复现训练与推理。
- 正式主要对照建议至少 3 个**配对随机种子**，B0/B1/E1/C1/C2 用一致的 seed 列表。预算不足可先 B1/E1 的三 seed，然后补其余；报告中明确缺失组。
- 不能凭目标域结果挑选最好的 seed、alpha、epoch 或从多个尝试挑一个目标域最优的 checkpoint。
- 如果原项目分别使用 `Unique` 与 `Repeat` 数据采样协议，不将它们混为一组；首先选原实验指定的协议进行成对比较，再做另一协议的独立复核。

## 9. 评价与分析

### 9.1 主指标（必须分开报源/目标）

最终目标是四个**未见**目标域的检测泛化。

- 主指标：`Target Mean mAP@0.5 = (NO + CZ + China-Drone + China-MotorBike)/4`。
- 次指标：四目标的 `mAP@0.5:0.95` 平均，及四个域各自的完整结果；同时报告三源域的独立 mAP，但不能拿源域提升冒充目标泛化提升。
- 统计单位为**百分比点 pp**；使用三 seed 平均±标准差，以及相同 seed 的 `E1-B1` 成对差值。
- 固定评估脚本、类别映射、NMS/置信度/IoU 阈值、imgsz、集划分、VOC/COCO 算法及评估样本数。

### 9.2 判别器机制指标

训练过程至少记录：

- `L_det`、`L_FAB`、`L_domain`、整体 loss、`grl_coeff`、`alpha_domain`、域 loss 有效样本量和 mask 掉的混合样本比率。
- 两簇的原始样本数、训练实际出现次数与 batch 占比、国家×簇交叉表、病害类别×簇分布。
- `balanced accuracy`、per-cluster recall、真实/预测簇分布、confusion matrix；不能只用未经修正的 overall accuracy。
- 自然簇先验熵 `H(C)=-Σ pi_k log(pi_k)`、CE 与判别器置信度；在不平衡数据上用 `CE≈log(2)` 简单断言“混淆成功”是不严谨的。
- 域分支作用到 Student 上的梯度范数、`grad_domain/grad_det`，以及训练是否出现检测性能被域损失压制。
- 若可能：冻结训练前/后的 Student，在独立源验证集上训练相同容量的 post-hoc 域 probe（只训练 probe、不回传 Student），检验域信息是否真正减少；仅观察对抗判别器当下混淆，不能充分证明域信息不可恢复。
- 记录每个 run 的判别器参数量、单 step 时间、额外显存，保证简化结构的计算成本结论有据。

### 9.3 机制解释不能越界

- E1 > B1：支持“在固定风格划分与相同任务约束下，单 K-way 机制更合适”；但不能单凭该比较声称风格划分本身更有效。
- B1 > B0：支持“固定风格划分优于国家划分（在原多对抗结构下）”。
- E1 > C1：支持“在单 K-way 结构下，风格域标签相对国家标签有优势”；但因 K=2 vs K=3，不可完全消除类别数影响，必须在讨论中说明。
- E1 不优于 B1：不等于潜在风格域无效，可能是 K-way 梯度尺度、域伪标签噪声、域分类器容量、特征层选择或缺失了专家支路能力；使用日志诊断，不能以主观理由直接强制改为在线聚类。
- 高域混淆、低检测 mAP：很可能域对抗抹掉病害判别信息；检查 FAB 是否有效、类别分布混杂、grad 大小，而不是单纯加大对抗权重。

## 10. 交付清单（Codex 必须提供）

1. `audit_before_change.md`：原代码关键位置、原训练损失计算/梯度符号、FAB 是否实际生效、聚类 artifact ID 对齐、混合增强、可移除专家依赖、发现的问题。
2. `implementation_report.md`：改动的真实文件路径与函数、对原实验入口的影响、数据结构、CLI/yaml 参数、损失与 GRL 梯度如何连接。
3. 独立模块与模式开关，原 `country_multi` 可原样运行；`latent_kway_hard` 确实只有一个 `[B,2]` head。
4. 自动化测试与日志：shape、label join、GRL gradient sign、FAB loss/grad、domain update、augment mask、checkpoints、推理兼容。
5. `run_manifest.json`：git commit/diff、完整 config、seed、各数据清单和伪标签文件 SHA256、环境、训练模型初始化来源、脚本参数。
6. 1 seed 的短跑 smoke test 及其曲线/简报，正式训练前确保 code correctness；正式训练后给出每域 mAP、四目标均值、源域均值、均值±std 与成对 seed 变化。
7. `comparison.csv` 与 `report.md`：B0/B1/E1/C1/C2（若未跑齐则标为未完成），附每域结果和机制分析，不得编造尚未运行的指标。
8. README 一键复现实验命令，能够在**不删除/污染原有实验产物**的情况下从零重跑。

## 11. Codex 明确禁止事项

- 不重新训练/拟合 K-Means，不重新选 K，不改变 `feat1 μ/σ → scaler → Kmeans++ → fixed K=2` 主线。
- 不将 `K=2` 误解为保留两个二分类判别器；必须是**一个** `[B,2]` 多分类判别器。
- 不把原国家域标签交给 E1 的 CE，不把中心距离或簇 soft q 交给 Hard CE。
- 不把 GRL 与 `-L_domain` 叠加；不让域分支输入 `.detach()`。
- 不修改原 FAB/Triplet，不新增 SupCon，不开启在线聚类或 EMA，不擅自换损失项。
- 不针对未见域调参/选 best epoch，不因目标域指标而修改 cluster 定义。
- 不向未经验证的 Mosaic/MixUp 图像硬塞一个原图硬簇 ID。
- 不自行假设 MAMDG 原工程的确切代码类、路径、损失符号或参数。
- 不静默覆盖已有实验、模型权重、数据集列表和聚类结果。

## 12. 本轮完成判据

本轮完成不等于必须“提升 mAP”。**判据是**：单 K 类对抗机制被严格实现、取代原多判别器而非与其并存、离线 K=2 标签真正进入 CE、反向梯度作用于 Student、原 FAB 功能保留、对比协议严格且机制与性能数据可复验。只有在这些前提成立后，才能讨论是否进入下一阶段 Offline Soft K-way / 温度选择。

---

## 附：现实限制与证据状态

- 已在线核验 MMLD 作者 GitHub 仓库存在、目录与公开运行脚本存在，但本次没有逐行成功读取全部源码；不能冒称照搬其具体类的实现。
- 已核验 Sicilia 2023 公开论文包含多类 CE 的明确定义、MLP 示例及对 GRL 调度的描述；上述小型 PyTorch 代码是**根据已证实思想写出的适配骨架**，不是作者仓库原样复制。
- MAMDG 作者公开仓库地址能由发表论文确认，但本会话无法直接验证其中当前具体文件内容；必须以 Codex 本地审计为准。
- 来自项目先前讨论：已有 K=2 固定潜在风格划分与 Hard/Soft/Expert 实验记录。凡是已经有结果的分支，本轮按实际本地仓库先核对，不应重新命名为“原创新实验”或覆盖旧产物；任何未见源实验数值一律保持未知。
