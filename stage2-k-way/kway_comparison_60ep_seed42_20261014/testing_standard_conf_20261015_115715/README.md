# 四组标准类别置信度七域复测

使用四组原有 best_epoch_weights.pth（最低源验证检测 loss），seed=42，不重新训练。

置信度为 max(sigmoid(class_logits))，解码器直接复用第一阶段标准置信度复测的代码副本 code/standard_utils_bbox.py。

640×640直接缩放，confidence=0.001，NMS IoU=0.5，无TTA；P/R/F1采用score≥0.5；VOC连续AP50与COCO101点AP50/AP50:95分开报告，COCO maxDets=100。每组七域9433张图；source/target/all分别按3/4/7域等权平均。

原始测试保留在各训练组 testing/。本次所有结果在本目录，smoke/仅为每域1张的启动检查，不属于正式结果。code/保存本次测试入口和标准解码器；launch_logs/保存日志与退出码；四组完成后自动生成测试结果报告.md、domain_metrics.csv、class_metrics.csv和macro_metrics.csv。

| 实验 | GPU | screen |
|---|---:|---|
| 01_country_multi_unique | 0 | kway_std_1_20261015_115715 |
| 02_latent_multi_unique | 1 | kway_std_2_20261015_115715 |
| 03_country_kway_unique | 2 | kway_std_3_20261015_115715 |
| 04_latent_kway_unique | 3 | kway_std_4_20261015_115715 |
