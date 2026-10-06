# 五组国家域/潜在域对比实验

入口：train_style_adv.py --experiment；实现：utils/comparison_training.py。

| screen | GPU | 实验 | 训练图片出现次数/轮 |
|---|---|---|---|
| R1 | 1 | baseline | 18960 |
| R2 | 2 | country_repeat | 18960 |
| R3 | 3 | latent_repeat | 13658 |
| R4 | 4 | country_unique | 12742 |
| R5 | 5 | latent_unique | 12742 |

100 epochs，实际总 batch size 12（不是每域12），seed 11，输入640，YOLOv8-s，SGD momentum .937、weight_decay .0005、nominal LR .01、按总batch缩放LR、cos schedule。未冻结Student，FP32。Mosaic/MixUp/FFT/MixStyle关闭，保留原单图片颜色/几何增强。训练和验证保留尾批。各组相同Student初始参数；R1/R2相同采样；R4/R5相同去重图片顺序。

国家域顺序：Japan、India、United_States；潜在域顺序：Cluster_0、Cluster_1。专家使用既有best且冻结，域标签来自合并TXT槽位，按单域TXT校验，不使用旧风格距离缓存/软路由。Cluster_1专家在第166轮退出，按用户确认使用第52轮best。

总损失：Ldet + 0.7*Ladv + 0.3*Ltriplet。对抗使用feat3(512x20x20)、相同结构的独立Patch判别器、LSGAN；Student按图片平均；每个判别器独立优化其real/fake平均损失，Adam lr=1e-5，betas=(.9,.999)。Student更新时固定判别器参数及BN统计，判别器更新时Student/专家特征detach。

三元组保留原方法的类别挖掘：两两域合并池，加上K>2时的全域池，及域内池；它不强制正样本跨域。统一inter margin=.5、intra margin=.1，intra权重=.1；按有效池取平均，再对20/40两个尺度取平均。消除旧两域函数的重复全域池和margin=0差异。排除自身作为正样本，没有有效正负样本的anchor跳过，缺域池跳过。框特征按正确的[y,x]坐标裁剪、边界截断及空间均值池化，修正旧实现的xy轴顺序与边界问题。所有完整方法使用同一实现。

验证使用同一1592张图片、相同排序，val loss按每批图片数加权。best仅按最低val loss，mAP每轮记录不用于选权重。保存best/last detector state_dict，以及含Student/EMA/优化器/判别器/RNG的training_state.pt。epoch_audit.jsonl记录实际图片次数、域计数、判别器更新及有效三元组anchor次数（各池/尺度累加）。

对比限制：重复与去重组的每轮更新数不同，重复组也重新平衡了域分布；R2/R3反映整个方案差异，R4/R5更直接比较统一图片预算下的域划分。仅一个随机种子，最终域泛化结论还需固定未见目标域评估。最优val loss不包含对抗/三元组，训练总loss包含这些项，不应直接比较两者绝对数值。

原始图片/TXT不修改。source_before保留旧代码，source_snapshot保留正式运行代码；smoke_tests不属于正式结果。launch_logs含各screen脚本、日志和退出码。
