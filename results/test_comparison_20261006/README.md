# 五模型七域测试

GPU 1–5，screen T1–T5，顺序为baseline、country_repeat、latent_repeat、country_unique、latent_unique。五个模型独立加载本次comparison_20261005_100ep中最低源域val loss的best_epoch_weights.pth。

每模型七域、9433张图片，共35组评估。输出目录按模型/国家隔离。原始预测为全精度坐标和置信度，保存在detection-results；GT保存在ground-truth；COCO导出和metrics.json位于国家目录。

640输入、直接缩放、confidence=.001、NMS IoU=.5、无TTA。VOC连续AP50和COCO101点AP50/AP50:95分开报告。固定score>=.5下计算VOC匹配的P/R/F1，各类别宏平均。COCO maxDets=100；area使用正确的width*height，未沿用旧导出器的area-10。

四个未见目标域Norway/new_Czech/new_China_Drone/new_China_MotorBike等权宏平均为主要对比，三个源域测试与全部七域宏平均另列。不用测试结果挑选checkpoint或改变阈值。

启动脚本/日志/退出码见launch_logs；smoke_tests仅用于工程检查，不属于正式结果；source_snapshot记录运行代码。全部完成后执行 python -m utils.summarize_comparison_tests --root 本目录 生成汇总和报告。

自动汇总已在screen TestSummary中运行。全量完成后生成domain_metrics.csv、class_metrics.csv、macro_metrics.csv、comparison_deltas.csv和测试结果报告.md；evaluation_status.json记录汇总成功或失败。
