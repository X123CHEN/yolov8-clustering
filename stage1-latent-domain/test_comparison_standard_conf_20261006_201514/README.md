# 标准类别置信度重新测试

置信度为 max(sigmoid(class_logits))。使用原五组 best 权重、七域 test_all.txt，GPU 1–5，screen T1–T5。其余测试参数与原测试一致。原测试结果保留于 ../test_comparison_20261006，不覆盖。

source_before 保存修改前代码；source_snapshot 保存本次代码；original_results_sha256.json 记录原结果全部文件校验值。完成后自动生成测试结果报告.md 和四类汇总 CSV。
