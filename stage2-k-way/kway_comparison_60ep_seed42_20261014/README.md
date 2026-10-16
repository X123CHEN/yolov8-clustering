# K类判别器四组60轮对照

seed=42；GPU 0/1/2/3；Unique；每组60轮；训练完成后自动评估七域。

| 实验目录 | GPU | 内容 |
|---|---:|---|
| 01_country_multi_unique | 0 | 三国家、三专家、三判别器、Triplet |
| 02_latent_multi_unique | 1 | 两潜在域、两专家、两判别器、Triplet |
| 03_country_kway_unique | 2 | 三国家、单K=3判别器、Hard CE+GRL、Triplet |
| 04_latent_kway_unique | 3 | 两潜在域、单K=2判别器、Hard CE+GRL、Triplet |

运行状态见status.json；screen和启动命令见run_manifest.json；训练进度见每组training.log，测试见testing.log；报告见每组结果分析报告.md及根目录四组实验对比报告.md。所有结果在本独立目录内。

代码、验证和复现命令见 ../new_experiments/kway_domain/README.md。审计见audit_before_change.md，实现见implementation_report.md，正式运行保留source_snapshot和git_diff.patch。
