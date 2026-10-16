# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
import models
import argparse
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn, init_model, init_random_seed)
from nets.maddg import FeatEmbedder
from core import loss
import itertools
import torch.optim as optim
from nets.backbone import Backbone

from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import json
import shutil
from utils.saver import Saver
from utils.dataloader import YoloDataset_eval,yolo_dataset_collate_eval
from utils.utils_fit import fit_one_epoch
from utils.style_cache import StyleDistanceCache
import dg_config
# from utils.utils_fit_woUS import fit_one_epoch
from core.loss import TripletLossCal
# from core.loss import TripletLossCal_double

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser(description="MADDoG")
# wrq修改自己的datasets--------------------------------------------------------------------------------------------------
# JIU
# parser.add_argument('--dataset1', type=str, default='Japan')
# parser.add_argument('--dataset2', type=str, default='India')
# parser.add_argument('--dataset3', type=str, default='United_States')
parser.add_argument('--dataset_target', type=str, default='Norway')
# -------------------------------------------------------------------------------------------------------------------

# czh修改自己的datasets，风格聚类后--------------------------------------------------------------------------------------------------
parser.add_argument('--dataset1', type=str, default='Style_0')
parser.add_argument('--dataset2', type=str, default='Style_1')
parser.add_argument('--dataset3', type=str, default='Style_2')

# datasets
# # NIJ
# parser.add_argument('--dataset1', type=str, default='Norway')
# parser.add_argument('--dataset2', type=str, default='India')
# parser.add_argument('--dataset3', type=str, default='Japan')
# parser.add_argument('--dataset_target', type=str, default='United_States')
# -------------------------------------------------------------------------------------------------------------------
# # UNI
# parser.add_argument('--dataset1', type=str, default='United_States')
# parser.add_argument('--dataset2', type=str, default='Norway')
# parser.add_argument('--dataset3', type=str, default='India')
# parser.add_argument('--dataset_target', type=str, default='Japan')
# -------------------------------------------------------------------------------------------------------------------
# # UNI
# parser.add_argument('--dataset1', type=str, default='Japan')
# parser.add_argument('--dataset2', type=str, default='United_States')
# parser.add_argument('--dataset3', type=str, default='Norway')
# parser.add_argument('--dataset_target', type=str, default='India')
# -------------------------------------------------------------------------------------------------------------------

# model
parser.add_argument('--arch_FeatExt', type=str, default='FeatExtractor')
parser.add_argument('--arch_FeatEmbd', type=str, default='FeatEmbedder')
parser.add_argument('--arch_Dis1', type=str, default='Discriminator1')
parser.add_argument('--arch_Dis2', type=str, default='Discriminator2')
parser.add_argument('--arch_Dis3', type=str, default='Discriminator3')
parser.add_argument('--init_type', type=str, default='xavier')
parser.add_argument('--embed_size', type=int, default=128)

# optimizer
parser.add_argument('--lr_DG_conf', type=float, default=0.0001)#
parser.add_argument('--lr_critic', type=float, default=0.00001)# czh修改 之前是0.0001
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)

# training configs
parser.add_argument('--training_type', type=str, default='Train')
parser.add_argument('--results_path', type=str, default='./logs/Style_soft/260712/soft_top2_conf')
# parser.add_argument('--results_path', type=str, default='./logs/5datasets_doamin/20240424')
# parser.add_argument('--results_path', type=str, default='./logs/5datasets_doamin/tttest_20240419_')
parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--batchsize', type=int, default=10)

# parser.add_argument('--training_type', type=str, default='Test')
# parser.add_argument('--results_path', type=str, default='./results/Test_20191008/')
# parser.add_argument('--batchsize', type=int, default=1)
# parser.add_argument('--tstfile', type=str, default='Train_20191008')
# parser.add_argument('--tstdataset', type=str, default='OULUCASIAMSU')
# parser.add_argument('--snapshotnum', type=str, default='2')

# parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dongjie_epochs', type=int, default=0)
parser.add_argument('--epochs', type=int, default=2)
# parser.add_argument('--pre_epochs', type=int, default=10)
parser.add_argument('--pre_epochs', type=int, default=5)
parser.add_argument('--log_step', type=int, default=1)
parser.add_argument('--tst_step', type=int, default=100)
parser.add_argument('--model_save_step', type=int, default=500)
parser.add_argument('--model_save_epoch', type=int, default=1)
parser.add_argument('--manual_seed', type=int, default=None)
parser.add_argument('--W_trip', type=float, default=0.3)
parser.add_argument('--W_gen', type=float, default=0.7)
parser.add_argument('--W_intra', type=float, default=0.1)
parser.add_argument('--W_genave', type=float, default=1 / 3)
parser.add_argument('--W_yolo', type=int, default=1.0)  # yolo 损失 的权重系数

# print(parser.parse_args())
args = parser.parse_args()

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
    style_cfg = dg_config.resolve_style_experiment_config()
    style_cache = StyleDistanceCache(
        dg_config.STYLE_TRAIN_CACHE, dg_config.STYLE_CLUSTER_METADATA
    )
    val_style_cache = StyleDistanceCache(
        dg_config.STYLE_VAL_CACHE, dg_config.STYLE_CLUSTER_METADATA
    )
    experiment_dirname = (
        f"{style_cfg['experiment_name']}_tau{style_cfg['temperature']}_"
        f"floor{style_cfg['confidence_floor']}_adv{style_cfg['adv_weight']}"
    )
    args.results_path = str(dg_config.PROJECT_ROOT / "logs" / "Style_soft" / "260712" / experiment_dirname)
    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=3 python train_beifen.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    classes_path = '/data4/czh/yolov8-cluster/yolov8-hard-clustering/model_data/rdd_classes.txt'
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''，下面的Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   从0开始训练有两个方案：
    #   1、得益于Mosaic数据增强方法强大的数据增强能力，将UnFreeze_Epoch设置的较大（300及以上）、batch较大（16及以上）、数据较多（万以上）的情况下，
    #      可以设置mosaic=True，直接随机初始化参数开始训练，但得到的效果仍然不如有预训练的情况。（像COCO这样的大数据集可以这样做）
    #   2、了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = './model_data/yolov8_s.pth'
    # model_path = '/cvgroup/home/wurx/yolov8_beifen/logs/3datasets_domain/together_3domain/best_epoch_weights.pth'
    # model_path = '/cvgroup/home/wurx/yolov8-pytorch-master/logs/5datasets_doamin/20240429_w_baseline/best_epoch_weights.pth'
    # ------------------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    # ------------------------------------------------------#
    input_shape = [640, 640]
    # ------------------------------------------------------#
    #   phi             所使用到的yolov8的版本
    #                   n : 对应yolov8_n
    #                   s : 对应yolov8_s
    #                   m : 对应yolov8_m
    #                   l : 对应yolov8_l
    #                   x : 对应yolov8_x
    # ------------------------------------------------------#
    phi = 's'
    # phi = 's'
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ------------------------------------------------------------------#
    #   mosaic              马赛克数据增强。
    #   mosaic_prob         每个step有多少概率使用mosaic数据增强，默认50%。
    #
    #   mixup               是否使用mixup数据增强，仅在mosaic=True时有效。
    #                       只会对mosaic增强后的图片进行mixup的处理。
    #   mixup_prob          有多少概率在mosaic后使用mixup数据增强，默认50%。
    #                       总的mixup概率为mosaic_prob * mixup_prob。
    #
    #   special_aug_ratio   参考YoloX，由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。
    #                       当mosaic=True时，本代码会在special_aug_ratio范围内开启mosaic。
    #                       默认为前70%个epoch，100个世代会开启70个世代。
    # ------------------------------------------------------------------#
    # mosaic = False
    # Offline style distances are defined for original images, not mosaics.
    mosaic = False
    mosaic_prob = 0.5
    mixup = False
    # mixup = True
    mixup_prob = 0.2
    special_aug_ratio = 0.5
    # ------------------------------------------------------------------#
    #   label_smoothing     标签平滑。一般0.01以下。如0.01、0.005。
    # ------------------------------------------------------------------#
    label_smoothing = 0
    # label_smoothing = 0.005   

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，Freeze_Train = True，此时仅仅进行冻结训练。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从0开始训练：
    #       Init_Epoch = 0，UnFreeze_Epoch >= 300，Unfreeze_batch_size >= 16，Freeze_Train = False（不冻结训练）
    #       其中：UnFreeze_Epoch尽量不小于300。optimizer_type = 'sgd'，Init_lr = 1e-2，mosaic = True。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    # ------------------------------------------------------------------#
    
    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 4
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
    #                           Adam可以使用相对较小的UnFreeze_Epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    Freeze_Train = False

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    # optimizer_type = "adam"
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    # weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = args.results_path
    # save_dir = 'logs_20231120_EMA_MADDG_DWR'
    # save_dir = 'test'
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #                   安装pycocotools库后，评估体验更佳。
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    # eval_period = 10
    eval_period = 1
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = 4

    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    # ------------------------------------------------------#
    # 测试
    # train_annotation_path = '/cvgroup/home/wurx/Datasets/test_BDD100K_cityscapes_KITTI_train.txt'
    # val_annotation_path   = '/cvgroup/home/wurx/Datasets/test_BDD100K_cityscapes_KITTI_val.txt'   

    train_annotation_path = '/data4/czh/yolov8-cluster/yolov8-hard-clustering/260708_style_train.txt'
    val_annotation_path   = '/data4/czh/yolov8-cluster/yolov8-hard-clustering/260708_style_val.txt'   
    seed_everything(seed)
    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ------------------------------------------------------#
    #   获取classes和anchor
    # ------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ----------------------------------------------------#
    #   下载预训练权重
    # ----------------------------------------------------#
    # if pretrained:
    #     if distributed:
    #         if local_rank == 0:
    #             download_weights(phi)
    #         dist.barrier()
    #     else:
    #         download_weights(phi)

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#

    ####wrq将maddg代码融合进 yolov8代码
    # savefilename = ""
    # savefilename = osp.join(args.dataset1 + args.dataset2 + args.dataset3 + '1')
    
    # czh 修改----------------------------------------------
    savefilename = osp.join(args.dataset1 + '_' + args.dataset2 + '_' + args.dataset3 + '_adv')
    summary_writer = SummaryWriter(osp.join(args.results_path, 'log', savefilename))
    saver = Saver(args, savefilename)
    arch_FeatExt = 'FeatExtractor'
    arch_FeatEmbd = 'FeatEmbedder'
    embed_size = 128
    arch_Dis1 = 'Discriminator1'
    arch_Dis2 = 'Discriminator2'
    arch_Dis3 = 'Discriminator3'

    FeatExtmodel = models.create(arch_FeatExt)
    # FeatExtmodel = feature123

    FeatExtmodel_pre1 = models.create(arch_FeatExt)
    FeatExtmodel_pre2 = models.create(arch_FeatExt)
    FeatExtmodel_pre3 = models.create(arch_FeatExt)

    # FeatEmbdmodel = models.create(arch_FeatEmbd, embed_size=embed_size)

    Dismodel1 = models.create(arch_Dis1)
    Dismodel2 = models.create(arch_Dis2)
    Dismodel3 = models.create(arch_Dis3)

    # FeatExtS1_restore = osp.join('/disk/home/wurx/yolov8-pytorch-master/results', 'Pre_train', 'snapshots', args.dataset1, 'Japan.pth')
    # FeatExtS2_restore = osp.join('/disk/home/wurx/yolov8-pytorch-master/results', 'Pre_train', 'snapshots', args.dataset2, 'India.pth')  
    # FeatExtS3_restore = osp.join('/disk/home/wurx/yolov8-pytorch-master/results', 'Pre_train', 'snapshots', args.dataset3, 'United_States.pth')

    PreFeatExtorS1 = init_model(net=FeatExtmodel_pre1, init_type=args.init_type, restore=dg_config.STYLE_EXPERT_PATHS[0]).to(device)
    PreFeatExtorS2 = init_model(net=FeatExtmodel_pre2, init_type=args.init_type, restore=dg_config.STYLE_EXPERT_PATHS[1]).to(device)
    PreFeatExtorS3 = init_model(net=FeatExtmodel_pre3, init_type=args.init_type, restore=dg_config.STYLE_EXPERT_PATHS[2]).to(device)
    for expert in (PreFeatExtorS1, PreFeatExtorS2, PreFeatExtorS3):
        expert.eval()
        for parameter in expert.parameters():
            parameter.requires_grad = False

    Dis_restore1 = None
    Dis_restore2 = None
    Dis_restore3 = None
    FeatExt_restore = None
    Detector_restore = None
    FeatEmbd_restore = None

    # FeatExtor = init_model(net=FeatExtmodel, init_type=args.init_type, restore=FeatExt_restore).to("cuda")
    FeatEmbdmodel = FeatEmbedder(in_channels=128, nettype=None).to("cuda")
    Discriminator1 = init_model(net=Dismodel1, init_type=args.init_type, restore=Dis_restore1)
    Discriminator2 = init_model(net=Dismodel2, init_type=args.init_type, restore=Dis_restore2)
    Discriminator3 = init_model(net=Dismodel3, init_type=args.init_type, restore=Dis_restore3)

    model = YoloBody(input_shape=input_shape, num_classes=num_classes, phi=phi, pretrained=pretrained).to('cuda')

    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict, temp_dict_backbone = [], [], {}, {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   获得损失函数
    # ----------------------#
    yolo_loss = Loss(model)
    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ----------------------------#
    #   权值平滑
    # ----------------------------#
    # ema_backbone = ModelEMA(backbone)
    ema = ModelEMA(model_train)
    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        style_snapshot = dict(style_cfg)
        style_snapshot.update({
            "distance_scale": style_cache.distance_scale,
            "train_cache": dg_config.STYLE_TRAIN_CACHE,
            "val_cache": dg_config.STYLE_VAL_CACHE,
            "style_expert_paths": dg_config.STYLE_EXPERT_PATHS,
            "hard_label_match_rate": 1.0,
            "val_hard_label_match_rate": val_style_cache.match_rate,
        })
        with open(os.path.join(save_dir, "style_experiment_config.json"), "w", encoding="utf-8") as f:
            json.dump(style_snapshot, f, ensure_ascii=False, indent=2)
        shutil.copy2(dg_config.STYLE_CLUSTER_METADATA,
                     os.path.join(save_dir, "cluster_metadata.json"))
        shutil.copy2(dg_config.__file__, os.path.join(save_dir, "config_snapshot.py"))
        print("\n[Style Experiment]")
        for key, value in style_snapshot.items():
            print(f"{key}: {value}")

    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (
                optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                    num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            # for param in backbone.parameters():
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # wrq修改代码---------------------------------------#
        # FeatExtor.train()
        FeatEmbdmodel.train()
        Discriminator1.train()
        Discriminator2.train()
        Discriminator3.train()
        PreFeatExtorS1.eval()
        PreFeatExtorS2.eval()
        PreFeatExtorS3.eval()
        from torch.nn import DataParallel

        # FeatExtor = DataParallel(FeatExtor)  # DataParallel：数据并行处理
        FeatEmbder     = DataParallel(FeatEmbdmodel)
        Discriminator1 = DataParallel(Discriminator1)
        Discriminator2 = DataParallel(Discriminator2)
        Discriminator3 = DataParallel(Discriminator3)
        PreFeatExtorS1 = DataParallel(PreFeatExtorS1)
        PreFeatExtorS2 = DataParallel(PreFeatExtorS2)
        PreFeatExtorS3 = DataParallel(PreFeatExtorS3)
        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})
        # optimizer.add_param_group({"params": itertools.chain(backbone.parameters())})
        optimizer.add_param_group({"params": itertools.chain(FeatEmbder.parameters())})

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                    mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
                                    train=True, special_aug_ratio=special_aug_ratio,
                                    style_cache=style_cache)
        val_dataset = YoloDataset_eval(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                       mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                       special_aug_ratio=0)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate_eval, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # FeatExtor.train()
        FeatEmbdmodel.train()
        Discriminator1.train()
        Discriminator2.train()
        Discriminator3.train()
        PreFeatExtorS1.eval()
        PreFeatExtorS2.eval()
        PreFeatExtorS3.eval()
        from torch.nn import DataParallel

        # FeatExtor = DataParallel(FeatExtor)  # DataParallel：数据并行处理
        FeatEmbder = DataParallel(FeatEmbdmodel)  # wrq该FeatEmbder  为FeatEmbdmodel
        Discriminator1 = DataParallel(Discriminator1)
        Discriminator2 = DataParallel(Discriminator2)
        Discriminator3 = DataParallel(Discriminator3)
        PreFeatExtorS1 = DataParallel(PreFeatExtorS1)
        PreFeatExtorS2 = DataParallel(PreFeatExtorS2)
        PreFeatExtorS3 = DataParallel(PreFeatExtorS3)

        criterionAdv = loss.GANLoss()  # 对抗性损失（Adversarial Loss）
        # criterionCls = torch.nn.BCEWithLogitsLoss()  # 分类任务的损失函数，常见的分类损失函数包括交叉熵损失（Cross-Entropy Loss）。

        # optimizer_DG_conf = optim.Adam(itertools.chain(model.backbone.parameters()),
        #                                lr=args.lr_DG_conf, betas=(args.beta1, args.beta2))
        # optimizer_DG_conf = optim.Adam(itertools.chain(backbone.parameters(), FeatEmbder.parameters()),
        #                                lr=args.lr_DG_conf, betas=(args.beta1, args.beta2))

        optimizer_critic1 = optim.Adam(Discriminator1.parameters(),
                                       lr=args.lr_critic,
                                       betas=(args.beta1, args.beta2))

        optimizer_critic2 = optim.Adam(Discriminator2.parameters(),
                                       lr=args.lr_critic,
                                       betas=(args.beta1, args.beta2))

        optimizer_critic3 = optim.Adam(Discriminator3.parameters(),
                                       lr=args.lr_critic,
                                       betas=(args.beta1, args.beta2))

        # iternum = max(len(data_loader1_real), len(data_loader1_fake),
        #               len(data_loader2_real), len(data_loader2_fake),
        #               len(data_loader3_real), len(data_loader3_fake))
        #
        # print('iternum={}'.format(iternum))
        #
        ####wrq将maddg代码融合进 yolov8代码

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            # eval_callback = EvalCallback(backbone, model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
            #                              eval_flag=eval_flag, period=eval_period)
            eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                         map_out_path=os.path.join(log_dir, 'temp_map_out'),
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            current_epoch = epoch
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                # for param in backbone.parameters():
                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if ema:
                    ema.updates = epoch_step * epoch

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate_eval, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # fit_one_epoch(model_train, model, ema, ema_backbone, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                          epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir,
                          Unfreeze_batch_size,

                          FeatEmbder, Discriminator1, Discriminator2, Discriminator3, PreFeatExtorS1, PreFeatExtorS2,
                          PreFeatExtorS3,
                          # backbone,FeatEmbder, Discriminator1, Discriminator2, Discriminator3, PreFeatExtorS1,PreFeatExtorS2, PreFeatExtorS3,
                          current_epoch, criterionAdv, args, TripletLossCal, optimizer_critic1,
                          optimizer_critic2, optimizer_critic3, summary_writer, saver,
                          style_cfg=style_cfg, distance_scale=style_cache.distance_scale,
                          local_rank=local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()

            try:
                from types import SimpleNamespace
                from utils.metrics_summary import generate_summary

                summary_args = SimpleNamespace(
                    results_path=args.results_path,
                    results_path_rel=True,
                    weights_path=os.path.join(save_dir, 'best_epoch_weights.pth'),
                    classes_path=classes_path,
                    fps_interval=100,
                    phi=phi,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    batch_size=Unfreeze_batch_size,
                    num_train=num_train,
                    num_val=num_val,
                    planned_epochs=UnFreeze_Epoch,
                    style_cfg=style_cfg,
                    num_gpus=ngpus_per_node if ngpus_per_node > 0 else 1,
                    map_threshold=0.45,
                    cpu=False,
                )
                generate_summary(log_dir, summary_args)
            except Exception as e:
                print(f'[metrics] metrics_summary generation failed: {e}')
