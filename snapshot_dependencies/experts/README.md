# 五个单域专家训练

200 epochs；batch size 16；seed 11；YOLOv8-s；仅检测损失；best 按完整源域验证集的最低 val loss 保存（各批损失按图片数加权）。

原始图片和 TXT 只读取；国家和簇专家分别独立从 model_data/yolov8_s.pth 初始化。

| screen | GPU | 专家 | train | val |
|---|---|---|---|---|
| R1 | 1 | India | 2578 | 322 |
| R2 | 2 | Japan | 6320 | 790 |
| R3 | 3 | United_States | 3844 | 480 |
| R4 | 4 | Cluster_0 | 5913 | 738 |
| R5 | 5 | Cluster_1 | 6829 | 854 |

正式权重在各专家子目录；终端日志在 launch_logs；短程检查结果在 smoke_tests，不用于后续方法。
每个专家的 run_config.json 记录实际参数，best_checkpoint.json 记录最优轮次和 val loss。mAP 仅记录，不用于选权重。
