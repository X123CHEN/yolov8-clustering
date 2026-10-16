# 四组60轮实验实现

## 代码和基线

新增代码全部位于 `new_experiments/kway_domain/`。第一阶段原文件、历史 results 和专家权重保留。

- `cluster_index.py:audit_source_labels()`：规范路径 join，确认国家/潜在 Unique 使用相同图片、目标框和验证集；固定 CSV 标签与既有数据列表一一一致。
- `discriminator.py:GradientReverse / hard_domain_loss()`：复用原卷积 KWayDiscriminator，原始 logits 对固定整数标签做自然均值 CE，仅翻转 Student 梯度。
- `training.py:main()`：原检测 Loss、DomainDataset、domain_triplet 直接导入第一阶段实现。多头沿用原 LSGAN 训练顺序；单头一次 backward，Student SGD 与 D Adam 各更新一次。alpha_domain固定；GRL随global step调度。
- `verify.py`：实际源 batch 标签/shape/梯度方向/FAB/零域权重退化检查、源样本冻结域probe、同batch原/新域梯度标定。
- `check_smoke.py`：四模式短跑、优化器和checkpoint状态、成对初始检测/FAB梯度一致、纯Student推理检查。
- `evaluation.py`：复用 comparison_evaluation 的 StrictYOLO、VOC/COCO指标实现，seed=42；七域固定列表全部评估。
- `worker.py / launch.py`：四个GPU、四个screen；各组训练后自动测试，状态与时间记录。
- `report.py:summarize()`：每组结果分析、四组结果与耗时对照、成对差值。

## 固定配置

seed=42，Unique，60轮，batch12，8 workers，输入640，phi=s；fp32，单卡，无Mosaic/MixUp。三源训练12742张、源验证1592张。源验证最低检测loss选epoch，EMA detector用于推理。Triplet总权重0.3，跨域/域内margin0.5/0.1，域内系数0.1，feat3/feat2两尺度原语义保留。

Student从yolov8_s.pth初始化，原SGD(momentum0.937,weight_decay5e-4)，初始lr0.001875，最低lr0.00001875，四组调度周期60轮。判别器Adam(lr1e-5,betas0.9/0.999)。多头Student对抗有效权重0.7。单头基于相同源训练batch的feature gradient一次标定，国家alpha=0.011994974424338084，潜在alpha=0.012946990562582444。公式和逐batch范数保存在verification.json。

## 运行结果

真实源batch GRL方向、head梯度方向、B=1/B=12和K=2/K=3均通过。FAB与Student域梯度非零，domain权重为0时任务loss与梯度一致。固定源样本上的无GRL域probe可学习标签，检查范围为接线与可学习性。

四组各完成3步短跑和源验证、权重/optimizer状态保存；检查结果见verification.json。训练loss和FAB初始梯度在同域的多头/单头组一致，单头CE只接一次GRL。推理在源验证图片上严格加载Student权重，不需要域标签或专家。

正式训练和测试结果由各组目录报告与根目录四组实验对比报告记录。训练与测试壁钟分别计时，计时包括各自初始化和指标/保存开销，另列训练batch和分域测试时间。
