"""四组潜在域消融的唯一日常配置入口。

直接运行 python train_latent_ablation.py；主要修改 EXPERIMENT。
COMMON 为四组共用参数，EXPERIMENTS 中同名参数覆盖 COMMON。
PROFILE 选择历史方案或审查后的 60 轮方案；所有日常参数仍只改本文件。
修改 alpha 会生成独立缓存，禁止覆盖旧缓存。
不读取 DG_* 环境变量；CUDA_VISIBLE_DEVICES 如已设置则优先于 gpu。
"""
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "hard_kway"
PROFILE = "sequential60"

# 四组共用。首要训练日志按历史 hard_expert 的完整口径生成；
# 最终七域测试仍统一采用标准类别置信度，不受这个训练日志设置影响。
# 不能把 legacy_uncertainty 与 max_class_sigmoid 两种曲线直接比较。
VALIDATION = {
    "metric": "source_val_coco_ap50",
    "confidence_mode": "legacy_uncertainty",
    "confidence": 0.05,
    "nms_iou": 0.5,
    "max_boxes": 100,
    "voc_fallback": False,
}

COMMON = {
    "num_domains": 2,
    "data_protocol": "unique",
    "epochs": 100,
    "scheduler_epochs": 100,
    "batch_size": 12,
    "seed": 11,
    "num_workers": 8,
    "gpu": "5",
    "input_shape": [640, 640],
    "phi": "s",
    "initial_weights": str(ROOT / "model_data/yolov8_s.pth"),
    "classes_path": str(ROOT / "model_data/rdd_classes.txt"),
    "results_root": str(ROOT / "results/latent_ablation"),
    "nominal_lr": 0.01,
    "min_lr_ratio": 0.01,
    "momentum": 0.937,
    "weight_decay": 5e-4,
    "critic_lr": 1e-5,
    "critic_betas": [0.9, 0.999],
    "gradient_clip": 2.0,
    "detection_weight": 1.0,
    "triplet_weight": 0.3,
    "adv_weight": 0.7,
    "adv_warmup_epochs": 0.0,
    "generator_bn_mode": "running",
    "expert_fake_assignment": "follow",
    "kway_confusion_target": "assignment_prior",
    "kway_head": "legacy_spatial",
    "kway_hidden": 128,
    "conditional_prior_smoothing": 20.0,
    "expert_normalization": "batch",
    "expert_group_norm_groups": 32,
    "expert_reference_routing": "hard",
    "coco_period": 1,
    # 首轮固定结构：P3 用于聚类，feat3 用于对抗，feat3/feat2 用于 Triplet。
    "adversarial_feature": "feat3_20x20",
    "triplet_domain": "hard",
    "triplet_inter_margin": 0.5,
    "triplet_intra_margin": 0.1,
    "triplet_intra_weight": 0.1,
    "adversarial_head": "legacy",
    "adversarial_objective": "uniform_confusion",
    "foreground_weight": 0.0,
    "source_ap_enabled": True,
    "feedback_period": 100,
    "feedback_smoothing": 0.9,
    "feedback_epsilon": 1e-8,
    "adv_gradient_ratio": 0.05,
    "adv_lambda_cap": 10.0,
    "center_mode": "fixed",
    "center_freeze_epochs": 5,
    "center_update_window": 8,
    "center_update_power": 0.6,
    "center_update_offset": 100,
}

MEMBERSHIP = {
    "method": "student_t",
    "alpha": 1.0,
    "eps": 1e-8,
    "analysis_root": str(ROOT / "latent_domain_analysis"),
    "cache_root": str(ROOT / "results/latent_membership"),
}

EXPERT_PATHS = [
    str(ROOT.parent / "yolov8-pretrain/results/expert_pretrain_20261004_200ep"
        / f"Cluster_{k}/best_epoch_weights.pth")
    for k in range(2)
]

# 修改各组专属参数就在这里；标签形式、网络数量由预设定义，不另设重复开关。
EXPERIMENTS = {
    "hard_expert": {"assignment": "hard", "architecture": "expert_binary", "adv_weight": 0.7},
    "soft_expert": {"assignment": "soft", "architecture": "expert_binary", "adv_weight": 0.7},
    "hard_kway": {"assignment": "hard", "architecture": "single_kway", "adv_weight": 0.7},
    "soft_kway": {"assignment": "soft", "architecture": "single_kway", "adv_weight": 0.7},
    "hard_kway_grl": {"assignment": "hard", "architecture": "single_kway", "adv_weight": 0.7},
}

# offline100 保留首轮实现；两个 refine60 方案都由专用 train-only 工具启动。
# 两种 K-way 的权重、G 先验、BN 和 warmup 相同，唯一目标差别是 D 的 hard/soft label。
PROFILES = {
    "sequential60": {
        "common": {"epochs": 60, "scheduler_epochs": 100, "adversarial_head": "pooled_mlp",
                   "adversarial_feature": "feat3_20x20", "foreground_weight": 0.1,
                   "critic_lr": 1e-4, "source_ap_enabled": True, "coco_period": 5, "adv_weight": 1.0},
        "membership": {"alpha": 2.0},
        "experiments": {
            "hard_expert": {"adversarial_objective": "expert_lsgan"},
            "hard_kway": {"adversarial_objective": "uniform_confusion"},
            "hard_kway_grl": {"adversarial_objective": "grl_domain_ce"},
            "soft_kway": {"adversarial_objective": "uniform_confusion"},
        },
    },
    "offline100": {"common": {}, "membership": {}, "experiments": {}},
    # 机制复盘后的整套结构方案。强度须在正式启动前一次校准并冻结。
    "structural60": {
        "common": {"epochs": 60, "scheduler_epochs": 100, "adv_warmup_epochs": 5.0,
                   "coco_period": 5},
        "membership": {"alpha": 4.0},
        "experiments": {
            "hard_expert": {"epochs": 100, "adv_warmup_epochs": 0.0, "coco_period": 1},
            "soft_expert": {"expert_normalization": "group", "adv_weight": 0.010424976306894625,
                            "expert_reference_routing": "sample_soft"},
            "hard_kway": {"kway_head": "conditional_style", "adversarial_feature": "feat1_style",
                          "kway_confusion_target": "conditional_hard_prior", "adv_weight": 0.785962379802654},
            "soft_kway": {"kway_head": "conditional_style", "adversarial_feature": "feat1_style",
                          "kway_confusion_target": "conditional_hard_prior", "adv_weight": 0.785962379802654},
        },
    },
    "refine60": {
        "common": {"epochs": 60, "scheduler_epochs": 100, "adv_warmup_epochs": 5.0},
        "membership": {"alpha": 2.0},
        "experiments": {
            "hard_expert": {"epochs": 100, "adv_warmup_epochs": 0.0},
            "soft_expert": {"expert_fake_assignment": "hard"},
            "hard_kway": {"adv_weight": 0.03, "generator_bn_mode": "batch",
                          "kway_confusion_target": "hard_prior"},
            "soft_kway": {"adv_weight": 0.03, "generator_bn_mode": "batch",
                          "kway_confusion_target": "hard_prior"},
        },
    },
    # 第二轮 60 轮：保留检测训练配方，减少软标签混合，调整 K-way 对抗强度。
    # alpha 对 hard 标签不产生变化；hard_kway 的有效变更是 adv_weight。
    "refine60_alpha4_adv01": {
        "common": {"epochs": 60, "scheduler_epochs": 100, "adv_warmup_epochs": 5.0},
        "membership": {"alpha": 4.0},
        "experiments": {
            "hard_expert": {"epochs": 100, "adv_warmup_epochs": 0.0},
            "soft_expert": {"expert_fake_assignment": "hard"},
            "hard_kway": {"adv_weight": 0.01, "generator_bn_mode": "batch",
                          "kway_confusion_target": "hard_prior"},
            "soft_kway": {"adv_weight": 0.01, "generator_bn_mode": "batch",
                          "kway_confusion_target": "hard_prior"},
        },
    },
}


def validate(config):
    import math
    name = config["experiment"]
    expected = {
        "hard_expert": ("hard", "expert_binary"), "soft_expert": ("soft", "expert_binary"),
        "hard_kway": ("hard", "single_kway"), "soft_kway": ("soft", "single_kway"),
        "hard_kway_grl": ("hard", "single_kway"),
    }
    if name not in expected or (config["assignment"], config["architecture"]) != expected[name]:
        raise ValueError("实验名称与 assignment/architecture 不一致")
    if config["num_domains"] != 2 or config["data_protocol"] != "unique":
        raise ValueError("本阶段固定 K=2、unique 协议")
    if config["input_shape"] != [640, 640] or config["phi"] != "s":
        raise ValueError("本阶段固定 YOLOv8-s、640x640")
    for key in ("epochs", "batch_size", "num_workers", "seed"):
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f"{key} 必须为整数")
    if config["epochs"] < 1 or config["batch_size"] < 2 or config["num_workers"] < 0 or config["seed"] < 0:
        raise ValueError("无效的训练轮数、batch、workers 或 seed")
    if (not isinstance(config.get("scheduler_epochs", config["epochs"]), int)
        or config.get("scheduler_epochs", config["epochs"]) < config["epochs"]):
        raise ValueError("scheduler_epochs 必须为不小于实际训练轮数的整数")
    warmup = config.get("adv_warmup_epochs", 0.0)
    if not math.isfinite(warmup) or not 0 <= warmup < config["epochs"]:
        raise ValueError("adv_warmup_epochs 必须位于 [0, epochs)")
    for key, allowed, default in [
        ("generator_bn_mode", ("running", "batch"), "running"),
        ("expert_fake_assignment", ("follow", "hard"), "follow"),
        ("kway_confusion_target", ("assignment_prior", "hard_prior", "uniform", "conditional_hard_prior"), "assignment_prior"),
        ("kway_head", ("legacy_spatial", "conditional_style"), "legacy_spatial"),
        ("expert_normalization", ("batch", "group"), "batch"),
        ("expert_reference_routing", ("hard", "sample_soft"), "hard"),
    ]:
        if config.get(key, default) not in allowed:
            raise ValueError(f"无效的 {key}")
    for key in ("nominal_lr", "min_lr_ratio", "critic_lr", "gradient_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} 必须是有限正数")
    for key in ("detection_weight", "triplet_weight", "adv_weight", "weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} 必须是有限非负数")
    if not 0 < config["momentum"] < 1 or len(config["critic_betas"]) != 2 or not all(0 <= x < 1 for x in config["critic_betas"]):
        raise ValueError("无效的优化器参数")
    style_head = config.get("kway_head", "legacy_spatial") == "conditional_style"
    if style_head and (config['architecture'] != 'single_kway' or config['kway_confusion_target'] != 'conditional_hard_prior'):
        raise ValueError('Style K-way requires the common source conditional hard prior')
    if not style_head and config.get('kway_confusion_target') == 'conditional_hard_prior':
        raise ValueError('Conditional prior requires the conditional style head')
    if config.get('expert_reference_routing', 'hard') == 'sample_soft' and (
        config['architecture'] != 'expert_binary' or config['assignment'] != 'soft'
        or config.get('expert_normalization') != 'group' or config['expert_fake_assignment'] != 'follow'):
        raise ValueError('Soft sampling requires soft real/fake matching with sample-independent GroupNorm')
    for key, default in [('coco_period', 1), ('kway_hidden', 128), ('expert_group_norm_groups', 32)]:
        if not isinstance(config.get(key, default), int) or isinstance(config.get(key, default), bool) or config.get(key, default) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not math.isfinite(config.get('conditional_prior_smoothing', 20.)) or config.get('conditional_prior_smoothing', 20.) <= 0:
        raise ValueError('Conditional smoothing must be positive')
    if (config["adversarial_feature"] != ('feat1_style' if style_head else 'feat3_20x20') or config["triplet_domain"] != "hard"
        or [config[k] for k in ("triplet_inter_margin", "triplet_intra_margin", "triplet_intra_weight")] != [0.5, 0.1, 0.1]):
        raise ValueError("本轮沿用原对抗特征层与 Hard-LD Triplet，不支持仅改配置切换实现")
    m = config["membership"]
    if m["method"] != "student_t" or not math.isfinite(m["alpha"]) or m["alpha"] <= 0 or not 0 < m["eps"] < 1e-3:
        raise ValueError("无效的 Student-t 参数")
    if config["architecture"] == "expert_binary" and len(config["expert_paths"]) != config["num_domains"]:
        raise ValueError("专家路径必须与 Cluster 编号一一对应")
    if config.get('adversarial_head') == 'pooled_mlp':
        objective = config['adversarial_objective']
        if config['epochs'] != 60 or config['foreground_weight'] != 0.1:
            raise ValueError('v2 uses 60 epochs and common foreground CE')
        if objective not in ('expert_lsgan', 'uniform_confusion', 'grl_domain_ce'):
            raise ValueError('Invalid v2 objective')
        if (objective == 'expert_lsgan') != (config['architecture'] == 'expert_binary'):
            raise ValueError('Expert/objective mismatch')
        if config['feedback_period'] < 1 or not 0 <= config['feedback_smoothing'] < 1:
            raise ValueError('Invalid gradient feedback')
        if config.get('center_mode', 'fixed') not in ('fixed', 'online'):
            raise ValueError('Invalid center mode')
    if "validation" in config:
        v = config["validation"]
        if v["metric"] != "source_val_coco_ap50" or v["voc_fallback"]:
            raise ValueError("训练日志必须使用源域 COCO AP50，禁止静默回退 VOC")
        if v["confidence_mode"] not in ("legacy_uncertainty", "max_class_sigmoid"):
            raise ValueError("未知验证置信度口径")
        if v["confidence"] != 0.05 or v["nms_iou"] != 0.5 or v["max_boxes"] != 100:
            raise ValueError("本阶段保持历史验证阈值和最大框数")
    return config


def resolve(experiment=None, profile=None):
    name = experiment or EXPERIMENT
    profile_name = profile or PROFILE
    if name not in EXPERIMENTS:
        raise ValueError(f"未知实验: {name}")
    if profile_name not in PROFILES:
        raise ValueError(f"未知方案: {profile_name}")
    selected = PROFILES[profile_name]
    config = deepcopy(COMMON)
    unknown = set(EXPERIMENTS[name]) - set(COMMON) - {"assignment", "architecture"}
    if unknown:
        raise ValueError(f"未知参数: {sorted(unknown)}")
    config.update(deepcopy(EXPERIMENTS[name]))
    config.update(deepcopy(selected["common"]))
    config.update(deepcopy(selected["experiments"].get(name, {})))
    membership = deepcopy(MEMBERSHIP)
    membership.update(deepcopy(selected["membership"]))
    if name == "hard_expert" and profile_name != 'sequential60':
        membership = deepcopy(MEMBERSHIP)
    config.update(experiment=name, profile=profile_name, membership=membership, expert_paths=list(EXPERT_PATHS),
                  validation=deepcopy(VALIDATION))
    return validate(config)
