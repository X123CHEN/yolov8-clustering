# 第一阶段设置下的四组 K-way 对照

四组统一 seed=11、100轮训练与100轮余弦学习率日程、Unique、batch=12、workers=8、640×640、YOLOv8-s、FP32，OMP/MKL/OpenBLAS线程数均为2。复用第一阶段训练/验证清单及硬域标签；每轮12742张训练图、1592张源验证图。所有组使用同一预训练初始化文件，各多专家组沿用原有专家权重。

| 实验目录 | GPU | 结构 |
|---|---:|---|
| 01_country_multi_unique | 0 | 国家域、三个冻结专家、三个LSGAN判别器 |
| 02_latent_multi_unique | 1 | 潜在K=2域、两个冻结专家、两个LSGAN判别器 |
| 03_country_kway_unique | 2 | 国家域、单K=3判别器、Hard CE+GRL |
| 04_latent_kway_unique | 3 | 潜在K=2域、单K=2判别器、Hard CE+GRL |

四组使用相同检测与Triplet实现，权重为1和0.3。01/02的LSGAN权重0.7；03/04的域权重在seed=11初始化下，按原源训练batch特征梯度标定方法重新计算，固定用于训练，数值与记录见verification.json。

最佳权重按最低源验证检测loss选择，使用Student EMA推理。标准测试置信度为max(sigmoid(class_logits))，解码器直接使用第一阶段标准置信度复测的代码副本；640直接缩放、confidence=0.001、NMS IoU=0.5、无TTA、COCO maxDets=100，P/R/F1阈值为0.5。每组训练结束后自动测试七域9433张图片。

主比较指标是七个测试集等权平均VOC mAP50；同时报告七域COCO mAP50、未见四域指标及训练成本。目标测试不参与标定或权重选择。

source_snapshot/保存并实际执行本次代码；run_experiment.py将数据与专家路径指向项目原目录。source_snapshot/utils/utils_bbox.py保留训练回调原解码口径；最终测试显式使用source_snapshot/standard_utils_bbox.py。训练源验证AP只用于记录，最佳权重选择仍使用检测loss。

run_manifest.json记录命令、screen和GPU映射；status.json记录任务状态；各组training.log和testing.log记录进度；checks/仅为3步启动验证，不属于正式训练。四组实验对比报告.md和comparison.csv随任务状态更新。
