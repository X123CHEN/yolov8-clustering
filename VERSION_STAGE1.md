# 第一阶段-潜在域&K的确定-1.0

归档日期：2026-10-06（Asia/Shanghai）。本版本归档 yolov8-hard-clustering 第一阶段的潜在域发现、K 确定、硬聚类五组对照训练及七域测试。

## 当前阶段

- 最终 K=2；正式训练簇为 5,913 / 6,829 张，验证簇为 738 / 854 张。
- K 选择仅使用源域训练数据；验证集使用训练 scaler 和聚类中心分配。
- 五组实验：baseline、country_repeat、latent_repeat、country_unique、latent_unique；100 epochs、seed 11。
- 模型权重按源域验证集最低 val loss 选择；每模型测试 9,433 张图片。
- 正式采用标准置信度 max(sigmoid(class_logits))，候选阈值 0.001、NMS IoU 0.5；P/R/F1 使用 score≥0.5。
- 修正后的四目标域 VOC mAP@50：baseline 24.69%、country_repeat 26.29%、latent_repeat 27.21%、country_unique 26.83%、latent_unique 27.10%。
- 去重组潜在域相对国家域：目标 VOC50 +0.27 个百分点、COCO50:95 +0.22 个百分点。单随机种子结果只提供初步证据。

## 实验资料

- [执行方案](潜在域K选择与硬聚类执行方案.md)
- [K 选择模块说明](latent_domain_analysis/README.md)
- [K 选择报告](latent_domain_analysis/results/K选择报告.md)
- [最终 K 确认](latent_domain_analysis/results/final_k_decision.json)
- [稳健性补充报告](latent_domain_analysis/results/robustness/稳健性补充报告.md)
- [域划分说明](latent_domain_analysis/domain_splits/README.md)
- [五组训练说明](results/comparison_20261005_100ep/README.md)
- [原置信度测试报告](results/test_comparison_20261006/测试结果报告.md)
- [标准置信度测试报告](results/test_comparison_standard_conf_20261006_201514/测试结果报告.md)

## 上传范围与复现

本版本包含当前项目源代码、文档、风格特征缓存、K 选择与稳健性产物、域划分清单、训练配置/审计/指标、两次测试配置/指标/代码快照，以及类别定义 model_data/rdd_classes.txt。

数据集图片及原始标注、.venv、模型权重（.pth/.pt）、完整推理预测文件和大型运行日志保留于本地，没有纳入此 Git 版本。复现训练或推理需另行提供原始 RDD2022、model_data/yolov8_s.pth、专家及五组 best 权重；原配置和 TXT 保留服务器绝对路径，迁移机器时需相应调整。

原测试结果保持独立；标准置信度重测的完整性记录位于 results/test_comparison_standard_conf_20261006_201514/original_results_verification.json。

原始仓库的 README.md 和 README_style.md 保留历史说明；本阶段以本文件及上述实验记录为准。
