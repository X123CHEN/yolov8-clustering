# 实施前代码审计

本轮四组：国家多专家、多判别器；潜在域多专家、多判别器；国家单 K-way；潜在域单 K-way。统一 Unique、seed=42、60 轮、batch=12、640×640、YOLOv8-s。GPU 分别为 0/1/2/3。

## 基线和数据

- 第一阶段实际实现：`utils/comparison_training.py`，`main()`；历史配置为 `results/comparison_20261005_100ep/{country_unique,latent_unique}/run_config.json`。
- `prepare_data()` / `read_records()` 用规范图片路径建立 records/owners；Unique 将去重后的 12,742 张图片按相同路径顺序组成单图记录，每轮完整遍历。源验证图片为 1,592 张，country/latent 为同一集合。
- `DomainDataset` 使用 `YoloDataset_eval`，Mosaic/MixUp 均关闭。保留该增强协议，单图 HSV/空间增强沿用原代码。不添加混合图标签，不生成软标签缓存。
- 硬聚类 artifact 为 `latent_domain_analysis/results/final_cluster_labels.csv`，标签字段 `latent_domain`；簇数固定为 2，样本数 5,913 / 6,829。图片路径按 resolve 关联，需与 existing combined_train.txt 的域归属逐一一致。
- 目标测试域只供训练完成后的评估，不参与聚类、训练或 checkpoint 选择。

## Student / 检测 / FAB

- `nets/yolo.py:YoloBody.forward()` 返回 `(dbox, cls, x, anchors, strides, feat3, feat2, feat1)`。
- `outputs[:5]` 送入 `nets/yolo_training.py:Loss`；`outputs[5]` 是 backbone feat3，640 输入、phi=s 时预期 `[B,512,20,20]`。feat2 为 `[B,256,40,40]`，feat1 为 `[B,128,80,80]`；运行时验证。
- 第一阶段域对抗使用 feat3。新单头继续接这个 Student feature，不切换到聚类提取层 feat1 或 neck P3/P5。
- FAB 保留第一阶段 `utils/comparison_training.py:domain_triplet()`，调用 `pooled_objects()` / `mined_triplet()`；feat3 与 feat2 的目标框平均池化；基于病害类别 batch-hard；排除 self/无效 anchor；跨域 margin=0.5、域内 margin=0.1、域内权重=0.1；有效域池和两个尺度均值；K=2 不重复全局池；总权重 0.3。
- 此 FAB 使用 Student 特征，不依赖冻结专家。国家两组共享国家分组，潜在两组共享簇分组，不因替换判别器改变 FAB。原旧入口 utils_fit.py 的三元组细节不同，本轮以用户指定的第一阶段实际实现为参照。
- FAB 参与 total.backward()，运行检查 loss 与上游梯度非零；检查域权重为零时检测/FAB 数值及梯度与原函数一致。

## 原多专家、多判别器

- `Backbone_pre` 从 `../yolov8-pretrain/results/expert_pretrain_20261004_200ep/{Japan,India,United_States,Cluster_0,Cluster_1}/best_epoch_weights.pth` 提取 backbone 权重，eval + requires_grad=False。
- `nets/maddg.py:Discriminator1` 使用卷积投影及三段 Conv/BN/LeakyReLU，最终单通道；feat3 输入通道设 512。原输出是 Patch-LSGAN 分数，非 BCE/域分类 CE。
- Student 域损失：各硬域子 batch `(D(feat)-1)^2`，按全 batch 图像数归一化；总有效系数 0.7。
- D 域损失：每域 `0.5*(mean((D(expert)-1)^2)+mean(D(student.detach())^2))`。Student 先更新，再 D；没有 GRL，不存在旧负号。
- Student optimizer=SGD，momentum=0.937、weight_decay=5e-4；batch12 的拟合 lr=0.001875，最低 0.00001875；cos 调度。本轮四组均采用 60 轮调度周期。D optimizer=Adam，lr=1e-5、betas=(0.9,0.999)。Student 梯度裁剪 2.0。

## 新单头接入决定

- 复用 `utils/latent_adversarial.py:KWayDiscriminator` 的卷积结构，只变最终 K 通道并空间平均，输出 `[B,K]`。国家 K=3，潜在 K=2；不加载专家，旧分支停止构建。
- 普通 Hard CE，对全部单来源有效样本取均值，不使用额外簇均衡权重。
- 单次 Student/D 前向与反向；GRL 反转仅作用于 Student，CE 为正。保留 Student SGD 与 D Adam 类型/学习率，各自 step 同一个 backward 得到的梯度。两个 optimizer 及 head 状态共同保存。
- 原 Student 域分支有效系数为 0.7。按相同源 batch 上的原 LSGAN/新 CE 特征梯度范数比标定固定 alpha_domain，公式为 median(0.7*原梯度范数/新梯度范数)。国家组 3 个 batch 得到 0.011994974424338084，潜在组 2 个 batch 得到 0.012946990562582444；训练时固定，不依据目标测试调参。GRL 使用 global optimizer step 的 DANN gamma=10 调度，与 alpha 分离。这是相对原交替 LSGAN 的明确机制变更。
- 现有 hard_kway/soft_kway 属于先验混淆分支；hard_kway_grl 的 V2 包含均衡 CE、反馈强度、附加前景目标，不直接用于本轮。既有配置和产物保持独立。

## 验证与评估

- 历史运行单卡、fp32，无 DDP/AMP、无冻结；本轮相同。
- 初始化统一 `model_data/yolov8_s.pth`；checkpoint 按最低源验证检测 loss 选择，Student EMA 用于验证和最终推理。原 detector EMA 与文档禁止的在线聚类 EMA teacher 用途不同。
- `utils/comparison_evaluation.py` 提供 strict Student 权重加载和七域固定测试；置信度 0.001、NMS IoU 0.5、标准类别置信度，无 TTA，640 输入；同时输出 VOC AP50 与 COCO AP50/AP50:95，不混用算法。
- 原训练源验证 callback 默认 confidence=0.05，每轮记录 AP；checkpoint 仍由 val_loss 决定。测试七域 list 合计样本数在启动前记录。
- 现有训练保存 Student、EMA、优化器、D 与随机状态；新模式增加真实恢复入口。
- 四个 screen 各使用独立目录/日志。报告分别记录训练壁钟耗时（含初始化、验证、保存）、训练 batch 时间之和、七域测试壁钟耗时、分域耗时、峰值显存和模型参数量。

运行验收结果写入同目录 `verification.json`；正式 run manifest 记录配置、数据清单与固定簇标签 SHA256、git commit/diff、初始化来源、环境及启动命令。
