# 表格字段与来源

| 文件 | 内容 | 数值单位 |
|---|---|---|
| `experiment_macro_metrics.csv` | 五套训练/测试协议的source、target、all域宏平均，共66行 | 三列AP为百分数0～100 |
| `experiment_domain_metrics.csv` | 对应协议的逐域结果，共154行 | 三列AP为百分数0～100 |
| `stage2_100_cost_and_accuracy.csv` | 原100轮四组comparison.csv的完整字段 | AP保留原0～1；时间秒；参数为个数 |
| `stage2_100_paired_differences.csv` | 03−01、04−02、02−01、04−03 | 精度差pp；时间/参数减少为百分比，可为负 |
| `k_selection_metrics.csv` | 原K选择汇总，K=2～8 | 保留原指标单位 |
| `file_inventory.csv` | 80个主线、快照或历史关联文件及角色 | 项目根相对路径 |
| `code_index.csv` | 上述Python文件的245个顶层函数/类及行号 | 当前文件行号 |

`experiment_*`中的`source`字段是实际CSV的位置；`protocol`区分第一阶段原分数、第一阶段标准复测、60轮原分数、60轮标准复测、100轮标准测试。`confidence=uncertainty`按实际执行解码器标识，不能用历史配置里写错的`confidence_mode`覆盖这一判定。

`source/target/all`分别表示源三域、未见四域、全部七域等权平均，不表示把原图合并后重算AP。原始权重、检测结果和逐类指标仍在各归档实验目录，本资料只读并汇总它们。

构建脚本会从七域summary重新算平均，并与100轮套件的已记录宏平均核对；遇到不完整七域或指标混用会停止，而不会补零生成结果。
