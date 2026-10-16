# -*- coding: utf-8 -*-
# =============================================================================
#  域泛化(DG)实验总开关。两种用法，任选其一：
#
#  【用法A：命令行环境变量（推荐跑多组合）】
#     不改本文件，运行前用环境变量覆盖。环境变量会被 DataLoader 子进程继承，
#     Windows/Linux 都安全。例：
#       Linux :  DG_USE_ADV=false DG_ALIGN=supcon DG_FFT=true \
#                CUDA_VISIBLE_DEVICES=0 python train_5datasets.py --results_path ./logs/R6
#       Win   :  $env:DG_USE_ADV='false'; $env:DG_ALIGN='supcon'; $env:DG_FFT='true';
#                python train_5datasets.py --results_path ./logs/R6
#
#  【用法B：直接改本文件的默认值】
#     把下面每个开关的“默认值”改掉即可（第二个参数）。
#
#  不设任何环境变量、也不改默认 = 你现在的原始行为（对抗 + 三元组）= 实验R1。
#
#  ---- 环境变量名对照 ----
#     DG_USE_ADV        对抗分支开关 (true/false)
#     DG_ALIGN          内容对齐槽 (triplet/supcon/none)
#     DG_SUPCON_TEMP    SupCon 温度
#     DG_FFT            FFT 开关 (true/false)
#     DG_FFT_P / DG_FFT_BETA / DG_FFT_LAM
#     DG_MIXSTYLE       MixStyle 开关 (true/false)
#     DG_MIXSTYLE_P / DG_MIXSTYLE_ALPHA / DG_MIXSTYLE_MODE (random/crossdomain)
#
#  ---- 消融阶梯预设（环境变量写法）----
#     R0 纯检测器 : DG_USE_ADV=false DG_ALIGN=none
#     R1 现有方法 : (什么都不设, 默认)
#     R2 只FFT    : DG_USE_ADV=false DG_ALIGN=none  DG_FFT=true
#     R3 只MixStyle: DG_USE_ADV=false DG_ALIGN=none DG_MIXSTYLE=true
#     R4 只SupCon : DG_USE_ADV=false DG_ALIGN=supcon
#     R5 只对抗   : DG_ALIGN=none
#     R6 主方法★  : DG_USE_ADV=false DG_ALIGN=supcon DG_FFT=true
#     R7 主+对抗  : DG_ALIGN=supcon DG_FFT=true
#     R8 MixStyle版主方法: DG_USE_ADV=false DG_ALIGN=supcon DG_MIXSTYLE=true
# =============================================================================
import os


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None or v == '':
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on', 't', 'y')


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else default


def _env_str(name, default):
    v = os.environ.get(name)
    return v.strip().lower() if v not in (None, '') else default


class CFG:
    # ---------- 1) 对抗分支 (true=保留对抗=原行为; false=砍掉对抗, 且无需专家权重) ----------
    use_adv = _env_bool('DG_USE_ADV', False)

    # ---------- 2) 内容对齐槽 (三选一, 互斥): triplet | supcon | none ----------
    align_type = _env_str('DG_ALIGN', 'none')
    supcon_temp = _env_float('DG_SUPCON_TEMP', 0.07)

    # ---------- 3) 风格增强: FFT 振幅交换 (数据端, 不改 loss) ----------
    fft_enabled = _env_bool('DG_FFT', False)
    fft_p = _env_float('DG_FFT_P', 0.5)        # 施加概率
    fft_beta = _env_float('DG_FFT_BETA', 0.08)  # 低频窗占比 (越大风格替换越强)
    fft_lam = _env_float('DG_FFT_LAM', 1.0)     # 振幅混合比 (1.0=整段低频换掉)

    # ---------- 4) 风格增强: MixStyle (模型端, 作用于 feat1, 不改 loss) ----------
    mixstyle_enabled = _env_bool('DG_MIXSTYLE', False)
    mixstyle_p = _env_float('DG_MIXSTYLE_P', 0.5)
    mixstyle_alpha = _env_float('DG_MIXSTYLE_ALPHA', 0.1)
    mixstyle_mode = _env_str('DG_MIXSTYLE_MODE', 'random')  # random | crossdomain
