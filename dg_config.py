"""风格硬/软聚类对抗实验配置与消融实验操作说明。

最常修改的四项：
    STYLE_EXPERIMENT_NAME      E2/E3/E4/E5 路由模式
    STYLE_SOFT_TEMPERATURE     软概率温度 tau
    STYLE_CONFIDENCE_FLOOR    E5 置信度下限
    STYLE_ADV_LOSS_WEIGHT     Student 对抗损失系数 lambda_adv

推荐实验顺序：
    A. 固定 tau=1、floor=0.2、adv=0.7，依次运行 E2/E3/E4/E5；
    B. 在 A 中最佳软路由上扫描 tau；
    C. 如果 E5 有优势，再扫描 floor 和 confidence power；
    D. 最后固定最佳路由参数，扫描 lambda_adv。

重要原则：一次只改变一个因素。输入尺寸、随机种子、训练轮数、检测损失、专家权重、
聚类缓存及三个 Style 编号映射应保持一致，否则不能作为严格消融。
"""

from pathlib import Path


# 项目根目录，缓存、专家权重和日志均以此为基准。
PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# 1. 主消融实验：E2/E3/E4/E5 只修改 STYLE_EXPERIMENT_NAME
# =============================================================================
# E2 style_hard：
#   使用原硬标签的 one-hot 权重，每张图片只参与一个专家/判别器分支。
#   作用：建立改用 feat1_80x80 后的新硬路由基线。
#
# E3 exp3_soft_dense：
#   使用完整 q=[q0,q1,q2]，每张图片参与三个分支，三个权重之和为 1。
#   作用：验证保留全部风格混合信息是否优于硬路由。
#
# E4 exp4_soft_top2：
#   只保留 q 中最大的两个分量并重新归一化，每张图片参与两个分支。
#   作用：检验第三个弱相关分支是否引入噪声。
#
# E5 exp5_soft_top2_conf：
#   先使用完整 q 计算熵置信度，再 Top-2 并乘置信度。
#   作用：在 E4 基础上降低风格边界/模糊样本的对抗贡献。
STYLE_EXPERIMENT_PRESETS = {
    "style_hard": {"mode": "hard", "topk": 1, "use_confidence": False},
    "exp3_soft_dense": {"mode": "soft_dense", "topk": 3, "use_confidence": False},
    "exp4_soft_top2": {"mode": "soft_top2", "topk": 2, "use_confidence": False},
    "exp5_soft_top2_conf": {
        "mode": "soft_top2_conf", "topk": 2, "use_confidence": True
    },
}

# 当前运行硬路由基线；完成后依次改为：
#   "exp3_soft_dense"
#   "exp4_soft_top2"
#   "exp5_soft_top2_conf"
STYLE_EXPERIMENT_NAME = "exp5_soft_top2_conf"

# 风格对抗总开关。正式 E2-E5 均保持 True；关闭后不属于这组路由消融。
STYLE_ADV_ENABLED = True

# 原聚类 K=3，且已经训练三个单域专家，不能在本阶段改变。
NUM_STYLE_DOMAINS = 3

# 聚类中心 k、Style_k 专家和第 k 个判别器必须严格同序。
STYLE_EXPERT_PATHS = [
    str(PROJECT_ROOT / "results" / "Pre_train" / f"Style_{k}" / "best_epoch_weights.pth")
    for k in range(NUM_STYLE_DOMAINS)
]

# 聚类与对抗统一使用 feat1：输入 640 时为 128×80×80。
# Student 使用 outputs[7]，专家使用返回特征的下标 0；判别器输入通道为 128。
# 若改层，必须同步修改 Student/专家取层位置和三个判别器输入通道，不能只改字符串。
STYLE_FEATURE_LAYER = "feat1_80x80"

# 专家只作为固定参考，不参与训练。
FREEZE_STYLE_EXPERTS = True
STYLE_EXPERT_EVAL_MODE = True
STYLE_EXPERT_NO_GRAD = True

# =============================================================================
# 2. 已验证的离线聚类缓存，通常不修改
# =============================================================================
STYLE_TRAIN_CACHE = str(PROJECT_ROOT / "style_cache" / "train_style_distances.npz")
STYLE_VAL_CACHE = str(PROJECT_ROOT / "style_cache" / "val_style_distances.npz")
STYLE_CLUSTER_PARAMS = str(PROJECT_ROOT / "style_cache" / "cluster_params.npz")
STYLE_CLUSTER_METADATA = str(PROJECT_ROOT / "style_cache" / "cluster_metadata.json")

# 强制 argmax(q) 与原硬标签完全一致：训练集和验证集当前都已验证为 100%。
# 如果出现任何不一致，训练应立即停止；禁止通过放宽 tolerance 掩盖缓存错配。
ASSERT_HARD_LABEL_MATCH = True
HARD_LABEL_MATCH_TOLERANCE = 0.0

# =============================================================================
# 3. 软路由参数
# =============================================================================
# q_k = softmax(-distance_sq_k / (tau * distance_scale))。
#
# tau 越小：概率越尖锐，主风格权重更大，更接近硬路由；
#             优点是减少弱相关专家干扰，风险是软路由退化为硬路由。
# tau 越大：概率越平滑，多个风格分支权重更接近；
#             优点是保留混合风格，风险是风格区分度下降、引入无关专家。
#
# 第一轮固定 1.0；只有完成 E2-E5 主对比后再扫描 0.5 / 1.0 / 2.0。
STYLE_SOFT_TEMPERATURE = 1.0
STYLE_SOFT_EPS = 1e-8  # 数值稳定项，不作为消融变量。

# =============================================================================
# 4. E5 聚类置信度参数
# =============================================================================
# entropy：使用完整三个概率，能反映整体分布是否模糊，是当前正式实现和推荐方案。
# max_prob：只看最大概率，没有完整利用第二、第三分量的信息。
# margin：第一大概率减第二大概率，强调主风格与次风格的差距。
# 注意：当前训练代码正式支持 entropy；不要仅修改字符串尝试未实现的类型。
STYLE_CONFIDENCE_TYPE = "entropy"

# 最终 confidence = floor + (1-floor) * raw_confidence，再进行 power 次幂。
# floor=0.0：允许极模糊样本接近完全不参与风格对抗，降权最强；
# floor=0.2：所有样本至少保留 20% 基础影响，是推荐初始值；
# floor=0.5：降权较温和，E5 会更接近 E4。
# 只在 E5 生效；E2/E3/E4 内部会把 confidence 置为 1。
STYLE_CONFIDENCE_FLOOR = 0.2

# gamma > 1：拉大高低置信样本差异，更强调典型风格，可能过度忽略难样本；
# gamma = 1：不额外改变置信度曲线，第一轮使用该值；
# 0 < gamma < 1：减弱样本间差异，保留更多边界样本，E5 更接近 E4。
# 建议先固定 1.0；若 E5 有效，再补充 0.5 / 1.0 / 2.0 消融。
STYLE_CONFIDENCE_POWER = 1.0

# 必须保持 True。顺序应为：完整 q -> 熵置信度 -> Top-2 -> 重新归一化 -> 乘置信度。
# 若先 Top-2 再计算熵，会人为提高模糊样本的置信度。
CONFIDENCE_FROM_FULL_PROB = True

# =============================================================================
# 5. 路由和对抗损失
# =============================================================================
# 第一版三个专家均执行完整 batch。E4/E5 的 Top-2 当前只稀疏“损失路由”，
# 不会减少三个专家的前向次数，因此 E4/E5 未必比 E3 明显更快。
# 真正子 batch 稀疏加速还要处理 DDP 空分支和 BatchNorm，当前保持 False。
STYLE_SPARSE_EXPERT_FORWARD = False

# 额外按阈值截断小权重的功能当前未接入；0.0 表示不截断。
STYLE_MIN_ROUTING_WEIGHT = 0.0

# 总损失：L_total = L_det + lambda_adv * L_adv + 原有 triplet 项。
# 这里不会降低 L_det；聚类置信度只作用于风格对抗损失。
# 0.7 来自原硬聚类实验 args.W_gen，并替代训练入口中的旧 args.W_gen。
# 主路由消融固定 0.7，确定最佳路由后扫描 0.3 / 0.5 / 0.7 / 1.0：
#   较小：训练更稳定，但风格对齐可能不足；
#   较大：对齐更强，但可能压制检测特征的类别/定位判别能力。
STYLE_ADV_LOSS_WEIGHT = 0.7

# warmup 开启后，前 N 个 epoch 将 lambda_adv 从较小值逐渐升至设定值。
# 首轮关闭，以便四个模式使用同一固定强度；若初期 loss 明显震荡，再单独比较开/关。
STYLE_ADV_WARMUP_ENABLED = False
STYLE_ADV_WARMUP_EPOCHS = 10

# 离线路由不得参与梯度计算。关闭 detach 会改变算法定义，正式实验保持 True。
DETACH_ROUTING_WEIGHT = True
DETACH_CONFIDENCE_WEIGHT = True

# =============================================================================
# 6. 日志，建议保持开启
# =============================================================================
# 日志将记录分支损失、分支权重质量、最大概率、熵、置信度及 Top-2 组合计数。
# 启动时还会保存 dg_config 快照、聚类 metadata 和实际解析后的 JSON 配置。
LOG_STYLE_BRANCH_LOSS = True
LOG_STYLE_BRANCH_MASS = True
LOG_STYLE_PROBABILITY_STATS = True
LOG_STYLE_CONFIDENCE_STATS = True
SAVE_STYLE_ROUTING_SNAPSHOT = True

# =============================================================================
# 7. 初步消融清单（仅供参考，不会自动覆盖上面的当前配置）
# =============================================================================
INITIAL_ABLATION_PLAN = (
    # A：四种路由主对比，其余参数固定。
    {"run": "E2_hard", "stage": "A_main", "experiment": "style_hard",
     "temperature": 1.0, "confidence_floor": 0.2, "adv_weight": 0.7},
    {"run": "E3_dense_tau1", "stage": "A_main", "experiment": "exp3_soft_dense",
     "temperature": 1.0, "confidence_floor": 0.2, "adv_weight": 0.7},
    {"run": "E4_top2_tau1", "stage": "A_main", "experiment": "exp4_soft_top2",
     "temperature": 1.0, "confidence_floor": 0.2, "adv_weight": 0.7},
    {"run": "E5_top2_conf_tau1_floor02", "stage": "A_main",
     "experiment": "exp5_soft_top2_conf", "temperature": 1.0,
     "confidence_floor": 0.2, "adv_weight": 0.7},

    # B：在最佳软路由模式上扫描温度。
    *({"run": f"tau_{tau}", "stage": "B_temperature", "experiment": "BEST_SOFT_MODE",
       "temperature": tau, "confidence_floor": 0.2, "adv_weight": 0.7}
      for tau in (0.5, 1.0, 2.0)),

    # C：仅 E5 在最佳温度下扫描置信度下限。
    *({"run": f"floor_{floor}", "stage": "C_confidence",
       "experiment": "exp5_soft_top2_conf", "temperature": "BEST_TAU",
       "confidence_floor": floor, "adv_weight": 0.7}
      for floor in (0.0, 0.1, 0.2, 0.5)),

    # D：固定最佳路由后扫描对抗强度。
    *({"run": f"adv_{weight}", "stage": "D_adv_weight", "experiment": "BEST_MODE",
       "temperature": "BEST_TAU", "confidence_floor": "BEST_FLOOR",
       "adv_weight": weight}
      for weight in (0.3, 0.5, 0.7, 1.0)),
)

# 如何执行上面的清单：
#
# Stage A（必须先做）：
#   只修改 STYLE_EXPERIMENT_NAME，其他参数保持 tau=1.0、floor=0.2、adv=0.7。
#   建议顺序 E2 -> E3 -> E4 -> E5。E2 必须重新训练，因为对抗层已由 feat3 改为 feat1。
#   比较主要目标域 mAP，同时观察训练是否稳定、各分支质量是否极端失衡。
#
# Stage B（选出最佳软模式后）：
#   把 BEST_SOFT_MODE 替换为 Stage A 中 E3/E4/E5 的最佳者，分别运行 tau=0.5/1.0/2.0。
#   如果 tau 降低后结果接近 E2，说明路由可能过尖；如果 tau 增大后下降，说明弱分支噪声较大。
#
# Stage C（仅当 E5 值得继续时）：
#   固定最佳 tau，扫描 floor。floor 越小，模糊样本被压制得越强。
#   若 floor=0 最好，说明边界样本可能确实有害；若 floor=0.5 更好，说明不应过度忽略难样本。
#
# Stage D（最后进行）：
#   固定最佳 mode/tau/floor，只改 STYLE_ADV_LOSS_WEIGHT。
#   若 adv 越大检测性能越差，说明对抗任务压制了检测表征；若 adv 太小时下降，则说明对齐不足。
#
# 每次正式运行前建议人工确认以下摘要：
#   1. STYLE_EXPERIMENT_NAME 与预期一致；
#   2. STYLE_FEATURE_LAYER 仍为 feat1_80x80；
#   3. 三个专家路径存在且顺序为 Style_0/1/2；
#   4. 启动日志 hard_label_match_rate 为 1.0；
#   5. 不同实验输出到不同日志目录；
#   6. 除当前扫描变量外，随机种子、epoch、batch size 和检测设置完全相同。


def resolve_style_experiment_config():
    """解析训练循环实际使用的当前实验配置。"""
    if STYLE_EXPERIMENT_NAME not in STYLE_EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown STYLE_EXPERIMENT_NAME: {STYLE_EXPERIMENT_NAME}")
    preset = STYLE_EXPERIMENT_PRESETS[STYLE_EXPERIMENT_NAME]
    cfg = {
        "experiment_name": STYLE_EXPERIMENT_NAME,
        "mode": preset["mode"],
        "topk": preset["topk"],
        "use_confidence": preset["use_confidence"],
        "temperature": STYLE_SOFT_TEMPERATURE,
        "confidence_type": STYLE_CONFIDENCE_TYPE,
        "confidence_floor": STYLE_CONFIDENCE_FLOOR,
        "confidence_power": STYLE_CONFIDENCE_POWER,
        "adv_weight": STYLE_ADV_LOSS_WEIGHT,
        "sparse_expert_forward": STYLE_SPARSE_EXPERT_FORWARD,
        "eps": STYLE_SOFT_EPS,
    }
    validate_style_config(cfg)
    return cfg


def validate_style_config(cfg):
    """启动前检查非法组合，防止消融设置被意外混用。"""
    valid_modes = {"hard", "soft_dense", "soft_top2", "soft_top2_conf"}
    if cfg["mode"] not in valid_modes:
        raise ValueError(f"Invalid style mode: {cfg['mode']}")
    if NUM_STYLE_DOMAINS != 3:
        raise ValueError("This implementation requires NUM_STYLE_DOMAINS == 3")
    if cfg["temperature"] <= 0:
        raise ValueError("STYLE_SOFT_TEMPERATURE must be > 0")
    if not 0.0 <= cfg["confidence_floor"] <= 1.0:
        raise ValueError("STYLE_CONFIDENCE_FLOOR must be in [0, 1]")
    if cfg["confidence_power"] <= 0:
        raise ValueError("STYLE_CONFIDENCE_POWER must be > 0")
    expected = {
        "hard": (1, False), "soft_dense": (3, False),
        "soft_top2": (2, False), "soft_top2_conf": (2, True),
    }[cfg["mode"]]
    if (cfg["topk"], cfg["use_confidence"]) != expected:
        raise ValueError(f"Illegal preset for mode {cfg['mode']}")
    if cfg["sparse_expert_forward"]:
        raise ValueError("Sparse expert forward is not implemented")
