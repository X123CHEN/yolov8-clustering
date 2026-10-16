# 四组 Unique 域对抗对照

| 目录 | 域标签 | 对抗结构 | GPU |
|---|---|---|---:|
| 01_country_multi_unique | JP/IN/US | 三个冻结专家、三个 LSGAN 判别器 | 0 |
| 02_latent_multi_unique | 固定 K=2 簇 | 两个冻结专家、两个 LSGAN 判别器 | 1 |
| 03_country_kway_unique | JP/IN/US | 一个 K=3 分类器、Hard CE、GRL | 2 |
| 04_latent_kway_unique | 固定 K=2 簇 | 一个 K=2 分类器、Hard CE、GRL | 3 |

四组使用 seed=42、60 轮、Unique、batch=12、640×640、YOLOv8-s、同一 Student 初始化和60轮学习率调度。数据加载、检测 loss 和 FAB 使用第一阶段 `utils/comparison_training.py` 的函数。域特征使用 backbone feat3；FAB 使用 feat3/feat2。源验证最小检测 loss 的 EMA 权重用于最终七域测试。

单头复用原判别器卷积主干，更换最终输出通道并做空间平均。Student SGD 与判别器 Adam 对一次 joint backward 的梯度同步更新。`alpha_domain` 在源训练 batch 上按原 LSGAN 的有效梯度量级一次标定并固定，GRL 单独使用 global optimizer step 的 DANN gamma=10 调度。单头不加载专家、不使用 soft q、类别条件输入、额外前景 loss 或在线聚类。

## 从零运行

在项目根目录执行。`RUN_DIR` 指向新建的独立实验目录，放在项目根目录下，与历史 results 分开。

```bash
PYTHON=/data4/czh/anaconda3/envs/yolo_dg/bin/python
RUN_DIR="$PWD/kway_comparison_60ep_seed42_$(date +%Y%m%d_%H%M%S)"
mkdir "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" -m new_experiments.kway_domain.verify --output "$RUN_DIR/verification.json" > "$RUN_DIR/verification.log" 2>&1
```

按 GPU 0/1/2/3 分别执行四组短跑。单头的 ALPHA 从 verification.json 读取，多判别器沿用0.7。

```bash
for ITEM in "0 01_country_multi_unique country" "1 02_latent_multi_unique latent" "2 03_country_kway_unique country" "3 04_latent_kway_unique latent"; do
  read -r GPU EXPERIMENT KIND <<< "$ITEM"
  ALPHA=0.7
  if [[ "$EXPERIMENT" == *kway* ]]; then
    ALPHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["calibrations"][sys.argv[2]]["alpha_domain"])' "$RUN_DIR/verification.json" "$KIND")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" -m new_experiments.kway_domain.training --experiment "$EXPERIMENT" --results_path "$RUN_DIR/checks" --epochs 60 --seed 42 --smoke_steps 3 --alpha_domain "$ALPHA" > "$RUN_DIR/smoke_${GPU}.log" 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m new_experiments.kway_domain.check_smoke --suite "$RUN_DIR"
"$PYTHON" -m new_experiments.kway_domain.launch --suite "$RUN_DIR"
```

启动器建立四个 screen。每个 worker 顺序执行本组60轮训练、七域测试和报告生成；四个 worker 并行。screen名称及命令保存在 run_manifest.json。使用 `screen -ls` 查看会话，使用 `screen -r <名称>` 查看 worker，训练进度在 `<实验目录>/training.log`。

## 产物与耗时

每组保存 run_config.json、训练日志、metrics、gradient_checks.jsonl、epoch_audit.jsonl、best/last权重、training_state.pt、testing/summary.csv、七域指标、timing.json 和结果分析报告.md。

训练结束时生成训练简报，测试结束后更新检测分析；套件根目录同步更新 comparison.csv 和四组实验对比报告.md。训练/测试各自的壁钟秒数、训练 batch 累计耗时、源验证 loss 时间、每步均时、分域测试时间均有记录。完整训练壁钟包含初始化、验证、每轮AP和保存；完整测试壁钟包含模型加载、推理、指标计算和保存。

## 恢复训练

训练在完整 epoch 保存 Student、EMA、两个类型的 optimizer、判别器、global step、scheduler定义与随机状态。已有 training_state.pt 的中断运行可重新执行本组 launch_logs/*.sh，worker 自动添加 --resume。不能重复启动仍在运行的同一组。已开始但未完成的测试需另设测试目录后重跑。

## 比较口径

主比较为实验3减实验1、实验4减实验2，分别报告未见域平均 COCO AP50 差值（pp）和训练时间加速比。COCO AP50:95 与 VOC AP50 分开列出。国家与潜在域两组的FAB域分组各自沿用第一阶段，结构成对比较保持一致。

本轮单个配对seed；训练时间受并行共享CPU与存储影响。源训练样本上的冻结probe用于验证标签可学习和梯度接线，不代表独立源验证上的域不变性结论。
