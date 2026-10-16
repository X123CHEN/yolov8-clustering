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
# from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn, init_model, init_random_seed)
from utils.utils_fit import fit_one_epoch
from nets.maddg import FeatEmbedder
from nets.backbone import Backbone
from utils.dataloader import get_dataset_loader, get_tgtdataset_loader
import os.path as osp
import models
from tensorboardX import SummaryWriter
from utils.saver import *
from utils.saver import (Saver)
import argparse
from core.utils import get_inf_iterator, mkdir
from core import evaluate
from torch.nn import DataParallel
from core import loss
import itertools
import os
from collections import OrderedDict
import torchvision.utils as vutils
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from core.Train import TripletLossCal


parser = argparse.ArgumentParser(description="MADDoG")
# wrq修改自己的datasets--------------------------------------------------------------------------------------------------
# IJU
parser.add_argument('--dataset1', type=str, default='India')
parser.add_argument('--dataset2', type=str, default='Japan')
parser.add_argument('--dataset3', type=str, default='United_States')
parser.add_argument('--dataset_target', type=str, default='Norway')
# -------------------------------------------------------------------------------------------------------------------

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
# parser.add_argument('--lr_DG_depth', type=float, default=0.0001)
parser.add_argument('--lr_DG_conf', type=float, default=0.00001)
parser.add_argument('--lr_critic', type=float, default=0.00001)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)

# training configs
parser.add_argument('--training_type', type=str, default='Train')
parser.add_argument('--results_path', type=str, default='./results/Train_20231027XiuGaiAdvCUOWU')
parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--batchsize', type=int, default=10)

# parser.add_argument('--training_type', type=str, default='Pre_train')
# parser.add_argument('--results_path', type=str, default='./results/Pre_train/')
# parser.add_argument('--batchsize', type=int, default=10)
# parser.add_argument('--dataset_target', type=str, default='MSU')

# wrq修改datasets--------------------------------------------------------------------------------------------------
# parser.add_argument('--training_type', type=str, default='Pre_train')
# parser.add_argument('--results_path', type=str, default='/disk/home/wurx/MADDoG-master/results/Pre_train/')
# parser.add_argument('--batchsize', type=int, default=4)
# parser.add_argument('--dataset_target', type=str, default='Norway')
# -------------------------------------------------------------------------------------------------------------------

# parser.add_argument('--training_type', type=str, default='Test')
# parser.add_argument('--results_path', type=str, default='./results/Test_20191008/')
# parser.add_argument('--batchsize', type=int, default=1)
# parser.add_argument('--tstfile', type=str, default='Train_20191008')
# parser.add_argument('--tstdataset', type=str, default='OULUCASIAMSU')
# parser.add_argument('--snapshotnum', type=str, default='2')

parser.add_argument('--dongjie_epochs', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--pre_epochs', type=int, default=10)
parser.add_argument('--pre_epochs', type=int, default=5)
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--tst_step', type=int, default=100)
parser.add_argument('--model_save_step', type=int, default=100)
parser.add_argument('--model_save_epoch', type=int, default=1)
parser.add_argument('--manual_seed', type=int, default=None)
parser.add_argument('--W_trip', type=int, default=1)
# parser.add_argument('--W_depth', type=int, default=1)
parser.add_argument('--W_gen', type=int, default=1)
parser.add_argument('--W_intra', type=int, default=0.1)
parser.add_argument('--W_cls', type=int, default=1)
parser.add_argument('--W_genave', type=int, default=1 / 3)
parser.add_argument('--W_yolo', type=int, default=0.5)#yolo 的bbox的损失值 的权重系数

print(parser.parse_args())
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。pip

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
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
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
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
    classes_path = 'model_data/rdd_classes.txt'
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
    model_path = '/disk/home/wurx/yolov8-pytorch-master/model_data/best_epoch_weights.pth'
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
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    # ------------------------------------------------------------------#
    #   label_smoothing     标签平滑。一般0.01以下。如0.01、0.005。
    # ------------------------------------------------------------------#
    label_smoothing = 0

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，Freeze_Train = True，此时仅仅进行冻结训练。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练12千瓦开始训练：
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
    Freeze_Epoch = 50
    Freeze_batch_size = 1
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #                           SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
    #                           Adam可以使用相对较小的UnFreeze_Epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    # UnFreeze_Epoch = 10
    UnFreeze_Epoch = 50
    Unfreeze_batch_size = 1
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    Freeze_Train = True

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
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 10
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs/20231016'
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
    eval_period = 10
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = 0

    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   val_annotation_path     验证图片路径和标签
    # ------------------------------------------------------#
    train_annotation_path = '2022_train.txt'
    val_annotation_path = '2022_val.txt'

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
    num_classes = num_classes + 1 #1 ： 背景噪声
    # print("class_names是这个：",class_names)
    # print("num_classes是这个：", num_classes)

    # ----------------------------------------------------#
    #   下载预训练权重
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    from nets.backbone import Backbone
    model = YoloBody(Backbone(), input_shape, num_classes, phi, pretrained=True)

    #wrq修改代码####################################################################################
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
        load_key, no_load_key, temp_dict = [], [], {}
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

    # ---------------------------#
    #   wrq添加三元组损失wrq修改的代码
    # --------------------------#
    FeatEmbdmodel = FeatEmbedder(in_channels=512, nettype=None).to("cuda")
    # ----------------------#
    # ----------------------#
    #   获得损失函数
    # ----------------------#
    yolo_loss = Loss(model)      #把  对抗损失的值 也加上去 一起去评估模型
    # ----------------------#
    #   记录Loss
    # ----------------------#
    # if local_rank == 0:
    #     time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    #     log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    #     loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    # else:
    #     loss_history = None

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
    ema = ModelEMA(model_train)

    # ----------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()

    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

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
        # if Freeze_Train:
        #     for param in FeatExtmodel.parameters():
        #         param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
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
        # train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
        #                             mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob,
        #                             train=True, special_aug_ratio=special_aug_ratio)
        # # print(train_dataset[0],train_dataset[1])
        # val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
        #                           mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
        #                           special_aug_ratio=0)

        # if distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
        #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
        #     batch_size = batch_size // ngpus_per_node
        #     shuffle = False
        # else:
        #     train_sampler = None
        #     val_sampler = None
        #     shuffle = True
        #
        # gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
        #                  pin_memory=True,
        #                  drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
        #                  worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        #
        # gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
        #                      pin_memory=True,
        #                      drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
        #                      worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   wrq添加的多的新的数据集加载器。
        # ----------------------#
        manual_seed = None

        savefilename = ""
        savefilename = osp.join(args.dataset1 + args.dataset2 + args.dataset3 + '1')
        seed = init_random_seed(manual_seed)
        summary_writer = SummaryWriter(osp.join(args.results_path, 'log', savefilename))

        Saver = Saver(args, savefilename)
        print("以下是参数设置：---------------------------------------------------------")
        Saver.print_config()

        data_loader1_real = get_dataset_loader(name=args.dataset1, getreal=True, batch_size=args.batch_size)
        data_loader1_fake = get_dataset_loader(name=args.dataset1, getreal=False, batch_size=args.batch_size)

        data_loader2_real = get_dataset_loader(name=args.dataset2, getreal=True, batch_size=args.batch_size)
        data_loader2_fake = get_dataset_loader(name=args.dataset2, getreal=False, batch_size=args.batch_size)

        data_loader3_real = get_dataset_loader(name=args.dataset3, getreal=True, batch_size=args.batch_size)
        data_loader3_fake = get_dataset_loader(name=args.dataset3, getreal=False, batch_size=args.batch_size)

        data_loader_target = get_tgtdataset_loader(name=args.dataset_target, batch_size=args.batch_size)

        # parser = argparse.ArgumentParser(description="MADDoG")
        # parser.add_argument('--dataset1', type=str, default='India')
        # parser.add_argument('--dataset2', type=str, default='Japan')
        # parser.add_argument('--dataset3', type=str, default='United_States')
        # parser.add_argument('--dataset_target', type=str, default='Norway')
        # parser.add_argument('--batchsize', type=int, default=2)

        #################### load models#####################
        arch_FeatExt = 'FeatExtractor'
        arch_FeatEmbd = 'FeatEmbedder'
        embed_size = 128
        arch_Dis1 = 'Discriminator1'
        arch_Dis2 = 'Discriminator2'
        arch_Dis3 = 'Discriminator3'
        init_type = 'xavier'

        FeatExtmodel = models.create(arch_FeatExt)

        FeatExtmodel_pre1 = models.create(arch_FeatExt)#wrq  改过了，改成backbone
        FeatExtmodel_pre2 = models.create(arch_FeatExt)
        FeatExtmodel_pre3 = models.create(arch_FeatExt)

        # FeatEmbdmodel = models.create(arch_FeatEmbd, embed_size=embed_size)


        Dismodel1 = models.create(arch_Dis1)
        Dismodel2 = models.create(arch_Dis2)
        Dismodel3 = models.create(arch_Dis3)

        FeatExtS1_restore = osp.join('results', 'Pre_train', 'snapshots', args.dataset1, 'DGFA-Ext-final.pt')  # /disk/home/wurx/MADDoG-master/results/Pre_train/snapshots
        FeatExtS2_restore = osp.join('results', 'Pre_train', 'snapshots', args.dataset2, 'DGFA-Ext-final.pt')  # 这个路径下相应的，对应的预训练权重。
        FeatExtS3_restore = osp.join('results', 'Pre_train', 'snapshots', args.dataset3, 'DGFA-Ext-final.pt')

        PreFeatExtorS1 = init_model(net=FeatExtmodel_pre1, init_type=init_type, restore=FeatExtS1_restore)   ###名字被wrq改成 FeatExtorS1 改成 PreFeatExtorS1
        PreFeatExtorS2 = init_model(net=FeatExtmodel_pre2, init_type=init_type, restore=FeatExtS2_restore)   #也是一样的 改2
        PreFeatExtorS3 = init_model(net=FeatExtmodel_pre3, init_type=init_type, restore=FeatExtS3_restore)

        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None

        FeatExt_restore = None
        Detector_restore = None
        FeatEmbd_restore = None

        # FeatEmbder = init_model(net=FeatEmbdmodel, init_type=init_type, restore=FeatEmbd_restore)

        FeatExtor = init_model(net=FeatExtmodel, init_type=init_type, restore=FeatExt_restore)

        Discriminator1 = init_model(net=Dismodel1, init_type=init_type, restore=Dis_restore1)
        Discriminator2 = init_model(net=Dismodel2, init_type=init_type, restore=Dis_restore2)
        Discriminator3 = init_model(net=Dismodel3, init_type=init_type, restore=Dis_restore3)

        print(">>> FeatExtor <<<")
        print(FeatExtor)

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#

        # if local_rank == 0:
        #     eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
        #                                  eval_flag=eval_flag, period=eval_period)
        # else:
        #     eval_callback = None

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#


        ###########################################################wrq将Train.py的代码移动到这个train.py中！！！！！！！！！！！
        # 1. setup network #
        ####################
        # set train state for Dropout and BN layers
        FeatExtor.train()
        FeatEmbdmodel.train()

        Discriminator1.train()
        Discriminator2.train()
        Discriminator3.train()

        PreFeatExtorS1.eval()
        PreFeatExtorS2.eval()
        PreFeatExtorS3.eval()

        FeatExtor = DataParallel(FeatExtor)  # DataParallel：数据并行处理

        FeatEmbder = DataParallel(FeatEmbdmodel)  # wrq该FeatEmbder  为FeatEmbdmodel

        Discriminator1 = DataParallel(Discriminator1)
        Discriminator2 = DataParallel(Discriminator2)
        Discriminator3 = DataParallel(Discriminator3)

        PreFeatExtorS1 = DataParallel(PreFeatExtorS1)
        PreFeatExtorS2 = DataParallel(PreFeatExtorS2)
        PreFeatExtorS3 = DataParallel(PreFeatExtorS3)

        ####################
        # setup criterion and optimizer
        criterionAdv = loss.GANLoss()  # 对抗性损失（Adversarial Loss）
        # criterionCls = torch.nn.BCEWithLogitsLoss()  # 分类任务的损失函数，常见的分类损失函数包括交叉熵损失（Cross-Entropy Loss）。

        optimizer_DG_conf = optim.Adam(itertools.chain(FeatExtor.parameters(), model.parameters(), FeatEmbder.parameters()),
                                           lr=args.lr_DG_conf,
                                           betas=(args.beta1, args.beta2))


        optimizer_critic1 = optim.Adam(Discriminator1.parameters(),
                                           lr=args.lr_critic,
                                           betas=(args.beta1, args.beta2))

        optimizer_critic2 = optim.Adam(Discriminator2.parameters(),
                                           lr=args.lr_critic,
                                           betas=(args.beta1, args.beta2))

        optimizer_critic3 = optim.Adam(Discriminator3.parameters(),
                                           lr=args.lr_critic,
                                           betas=(args.beta1, args.beta2))

        # scheduler_DG_conf = torch.optim.lr_scheduler.ExponentialLR(optimizer_DG_conf, gamma=0.70)
        # scheduler_critic1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_critic1, gamma=0.70)
        # scheduler_critic2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_critic2, gamma=0.70)
        # scheduler_critic3 = torch.optim.lr_scheduler.ExponentialLR(optimizer_critic3, gamma=0.70)

        iternum = max(len(data_loader1_real), len(data_loader1_fake),
                          len(data_loader2_real), len(data_loader2_fake),
                          len(data_loader3_real), len(data_loader3_fake))

        print('iternum={}'.format(iternum))
        ###########################################################wrq将Train.py的代码移动到这个train.py中！！！！！！！！！！！

        global_step = 0  #wrq添加代码

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
####################################################################wrq将Train.py的代码移动到这个train.py中！！！！！！！！！！！
            current_epoch = epoch
            # scheduler_DG_conf.step()  #wrq添加权重衰减器
            # if current_epoch >= args.dongjie_epochs:
            #     scheduler_critic1.step()
            #     scheduler_critic2.step()
            #     scheduler_critic3.step()



            #wrq添加以下代码
            data1_real = get_inf_iterator(data_loader1_real)
            data1_fake = get_inf_iterator(data_loader1_fake)

            data2_real = get_inf_iterator(data_loader2_real)
            data2_fake = get_inf_iterator(data_loader2_fake)

            data3_real = get_inf_iterator(data_loader3_real)
            data3_fake = get_inf_iterator(data_loader3_fake)

            for step in range(iternum):
                FeatExtor.train()
                FeatEmbder.train()

                Discriminator1.train()
                Discriminator2.train()
                Discriminator3.train()

                # ============ one batch extraction ============#
                # ----------------------------------------------------------wrq修改
                cat_img1_real, lab1_real, real1_x_y_w_h, qiyu1_bbox = next(data1_real)
                cat_img1_fake, lab1_fake, _, _ = next(data1_fake)

                cat_img2_real, lab2_real, real2_x_y_w_h, qiyu2_bbox = next(data2_real)
                cat_img2_fake, lab2_fake, _, _ = next(data2_fake)

                cat_img3_real, lab3_real, real3_x_y_w_h, qiyu3_bbox = next(data3_real)
                cat_img3_fake, lab3_fake, _, _ = next(data3_fake)

                # bboxes = torch.randn(10, 5).float().to(cat_img3_real.device)  ###TODO!!!!!!!!!! 输入正确的bboxes

                # ============ one batch collection ============#
                # fake_x_y_w_h_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0])

                ori_img1 = torch.cat([cat_img1_real, cat_img1_fake], 0).cuda()
                lab1 = torch.cat([lab1_real, lab1_fake], 0)
                # real1_x_y_w_h_tensor = torch.tensor([[float(val) for val in item] for item in real1_x_y_w_h]).view(args.batch_size, 4)
                # real1_x_y_w_h_tensor = torch.tensor([[float(val) for val in item] for item in real1_x_y_w_h])
                # bboxes1 = torch.cat([real1_x_y_w_h_tensor, fake_x_y_w_h_tensor], 0)

                ori_img2 = torch.cat([cat_img2_real, cat_img2_fake], 0).cuda()
                lab2 = torch.cat([lab2_real, lab2_fake], 0)
                # real2_x_y_w_h_tensor = torch.tensor([[float(val) for val in item] for item in real2_x_y_w_h]).view(
                #     args.batch_size, 4)
                # real2_x_y_w_h_tensor = torch.tensor([[float(val) for val in item] for item in real2_x_y_w_h])
                # bboxes2 = torch.cat([real2_x_y_w_h_tensor, fake_x_y_w_h_tensor], 0)

                ori_img3 = torch.cat([cat_img3_real, cat_img3_fake], 0).cuda()
                lab3 = torch.cat([lab3_real, lab3_fake], 0)
                # real3_x_y_w_h_tensor = torch.tensor([[float(val) for val in item] for item in real3_x_y_w_h]).view(
                #     args.batch_size, 4)
                # real3_x_y_w_h_tensor = torch.tensor([[float(val) for val in item] for item in real3_x_y_w_h])
                # bboxes3 = torch.cat([real3_x_y_w_h_tensor, fake_x_y_w_h_tensor], 0)

                # wrq修，修改尺寸， 让 img1、2、3的尺寸相同可cat，，使用resize
                # 假设 ori_img1 的大小是 (height, width)
                height, width = 640, 640
                ori_img1 = F.interpolate(ori_img1, size=(height, width), mode='bilinear', align_corners=False)
                # 使用 interpolate 函数将 ori_img2 调整为相同的大小
                ori_img2 = F.interpolate(ori_img2, size=(height, width), mode='bilinear', align_corners=False)
                # 使用 interpolate 函数将 ori_img3 调整为相同的大小
                ori_img3 = F.interpolate(ori_img3, size=(height, width), mode='bilinear', align_corners=False)

                ori_img = torch.cat([ori_img1, ori_img2, ori_img3], 0)
                ori_img = ori_img.cuda()

                label = torch.cat([lab1, lab2, lab3], 0)
                # bbox1 = torch.cat((lab1_real.view(-1, 1), real1_x_y_w_h_tensor), dim=1)
                # bbox2 = torch.cat((lab2_real.view(-1, 1), real2_x_y_w_h_tensor), dim=1)
                # bbox3 = torch.cat((lab3_real.view(-1, 1), real3_x_y_w_h_tensor), dim=1)
                # #复制batch个   fake 的  9，0，0，0，0
                # fake_x_y_w_h_tensor = fake_x_y_w_h_tensor.repeat(args.batch_size, 1)
                # fake_x_y_w_h_tensor = torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0]])
                fake_x_y_w_h_tensor2_1 = torch.tensor([(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
                fake_x_y_w_h_tensor4_3 = torch.tensor([(3.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
                fake_x_y_w_h_tensor6_5 = torch.tensor([(5.0, 0.0, 0.0, 0.0, 0.0, 0.0)])  # 序号，label，  x，y，w，h

                # 创建一个PyTorch tensor
                real1_x_y_w_h_tensor = torch.tensor([float(value) for value in real1_x_y_w_h[0].split(', ')])
                real2_x_y_w_h_tensor = torch.tensor([float(value) for value in real2_x_y_w_h[0].split(', ')])
                real3_x_y_w_h_tensor = torch.tensor([float(value) for value in real3_x_y_w_h[0].split(', ')])
                bbox1_0 = torch.tensor([0])
                bbox2_2 = torch.tensor([2])
                bbox3_4 = torch.tensor([4])
                # bbox1 = torch.cat((bbox1_0,lab1_real,real1_x_y_w_h_tensor),dim=0)
                # bbox2 = torch.cat((bbox2_2,lab2_real,real2_x_y_w_h_tensor), dim=0)
                # bbox3 = torch.cat((bbox3_4,lab3_real,real3_x_y_w_h_tensor), dim=0)

                bbox1 = torch.cat((bbox1_0, lab1_real, real1_x_y_w_h_tensor), dim=0).unsqueeze(0)
                bbox2 = torch.cat((bbox2_2, lab2_real, real2_x_y_w_h_tensor), dim=0).unsqueeze(0)
                bbox3 = torch.cat((bbox3_4, lab3_real, real3_x_y_w_h_tensor), dim=0).unsqueeze(0)

                if len(qiyu1_bbox) != 0:
                    qiyu1_bbox = [torch.tensor([float(value) for value in bbox_str[0].split(',')]) for bbox_str in
                                  qiyu1_bbox]
                    qiyu1_bbox = torch.stack(qiyu1_bbox)
                    new1_column = 0 * torch.ones((qiyu1_bbox.size(0), 1))
                    qiyu1_bbox = torch.cat((new1_column, qiyu1_bbox), dim=1)
                    bbox1 = torch.cat((bbox1, qiyu1_bbox), dim=0)

                if len(qiyu2_bbox) != 0:
                    qiyu2_bbox = [torch.tensor([float(value) for value in bbox_str[0].split(',')]) for bbox_str in
                                  qiyu2_bbox]
                    qiyu2_bbox = torch.stack(qiyu2_bbox)
                    new2_column = 2 * torch.ones((qiyu2_bbox.size(0), 1))
                    qiyu2_bbox = torch.cat((new2_column, qiyu2_bbox), dim=1)
                    bbox2 = torch.cat((bbox2, qiyu2_bbox), dim=0)

                if len(qiyu3_bbox) != 0:
                    qiyu3_bbox = [torch.tensor([float(value) for value in bbox_str[0].split(',')]) for bbox_str in
                                  qiyu3_bbox]
                    qiyu3_bbox = torch.stack(qiyu3_bbox)
                    new3_column = 4 * torch.ones((qiyu3_bbox.size(0), 1))
                    qiyu3_bbox = torch.cat((new3_column, qiyu3_bbox), dim=1)
                    bbox3 = torch.cat((bbox3, qiyu3_bbox), dim=0)

                bboxes = torch.cat((bbox1, fake_x_y_w_h_tensor2_1,
                                    bbox2, fake_x_y_w_h_tensor4_3,
                                    bbox3, fake_x_y_w_h_tensor6_5), dim=0)
                # bboxes = torch.cat((bbox1.unsqueeze(0), bbox2.unsqueeze(0), bbox3.unsqueeze(0)), dim=0)
                # label = label.long().squeeze().cuda()

                with torch.no_grad():
                    pre_feat_ext1 = PreFeatExtorS1(ori_img1)[2] #torch.Size([2, 256, 40, 40])
                    pre_feat_ext2 = PreFeatExtorS2(ori_img2)[2] #torch.Size([2, 256, 40, 40])
                    pre_feat_ext3 = PreFeatExtorS3(ori_img3)[2] #torch.Size([2, 256, 40, 40])

                # ============ domain generalization supervision ============#
                optimizer_DG_conf.zero_grad()
                # 下列特征分别大小分别是：256, 80, 80。      512, 40, 40。     1024 * deep_mul(1.0)=1024 , 20, 20
                feat_ext, feat_ext2, feat_ext3 = FeatExtor(ori_img)
                # torch.Size([6, 128, 80, 80]) torch.Size([6, 256, 40, 40]) torch.Size([6, 512, 20, 20])


                outputs = model(feat_ext, feat_ext2, feat_ext3)
                loss_boxes = yolo_loss(outputs, bboxes)
                # print("loss_boxes: ", loss_boxes)

                feat_tgt = feat_ext3 #feat_tgt的size：torch.Size([6, 512, 20, 20])

                # ************************* confusion all **********************************#
                # target_real_label = 2 or 1 or 3 or 4
                # target_real_label = label.float()
                # target_fake_label = 0
                # target_fake_label = torch.tensor([0., 0., 0., 0., 0., 0.]).cuda()

                # predict on generator                       # yolov8 骨干提出3个不同尺度的特征，然后对三个尺度的特征都进行生成损失的计算
                if current_epoch >= args.dongjie_epochs:
                    loss_generator1 = criterionAdv(Discriminator1(feat_tgt), 0.9)
                    loss_generator2 = criterionAdv(Discriminator2(feat_tgt), 0.9)
                    loss_generator3 = criterionAdv(Discriminator3(feat_tgt), 0.9)
                else:
                    loss_generator1 = 0
                    loss_generator2 = 0
                    loss_generator3 = 0

                # loss_generator4 = criterionAdv(Discriminator1(feat_ext2), True)
                # loss_generator5 = criterionAdv(Discriminator2(feat_ext2), True)
                # loss_generator6 = criterionAdv(Discriminator3(feat_ext2), True)
                #
                # loss_generator7 = criterionAdv(Discriminator1(feat_ext3), True)
                # loss_generator8 = criterionAdv(Discriminator2(feat_ext3), True)
                # loss_generator9 = criterionAdv(Discriminator3(feat_ext3), True)

                feat_embd = FeatEmbder(feat_tgt)  ## Size([6, 512]),(6*batch_size=6,,6是三个鉴别器都有正负样本，所以2*3)

                ########## cross-domain triplet loss #########
                # print(feat_embd.size(), lab1.size(), lab2.size(), lab3.size())
                Loss_triplet = TripletLossCal(args, feat_embd, lab1, lab2, lab3)

                # Loss_cls = criterionCls(label_pred.squeeze(), label.float())
                # yolov8 骨干提出3个不同尺度的特征，然后对三个尺度的特征都进行生成损失的计算

                Loss_gen_feat_tgt = args.W_genave * (loss_generator1 + loss_generator2 + loss_generator3)
                # Loss_gen_feat_ext2 = args.W_genave * (loss_generator4 + loss_generator5 + loss_generator6)
                # Loss_gen_feat_ext3 = args.W_genave * (loss_generator7 + loss_generator8 + loss_generator9)

                # Loss_G = args.W_trip * Loss_triplet + args.W_cls * Loss_cls + args.W_gen * Loss_gen
                # Loss_G = args.W_trip * Loss_triplet + args.W_gen * (Loss_gen_feat_tgt + Loss_gen_feat_ext2 + Loss_gen_feat_ext3) + 1.0 * loss_boxes
                Loss_G = args.W_gen * Loss_gen_feat_tgt + args.W_yolo * loss_boxes
                # Loss_G = args.W_trip * Loss_triplet + args.W_gen * Loss_gen_feat_tgt

                if current_epoch >= args.dongjie_epochs:
                    Loss_G.backward()
                    optimizer_DG_conf.step()
                else:
                    # for param in Loss_G.parameters():
                    #     param.requires_grad = False
                    Loss_gen_feat_tgt = 0.0

                # ************************* confusion domain 1 with 2,3 **********************************#

                feat_src1 = torch.cat([pre_feat_ext1, pre_feat_ext1, pre_feat_ext1], 0) # feat_src: torch.Size([6, 256, 40, 40])

                # #wrq硬改将  预训练得到的特征([6, 256, 40, 40]) 强转([6, 512, 20, 20])  不知道什么原因 预训练得到的特征本就该是512, 20, 20
                # model_to_512 = nn.Sequential(
                #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                #     nn.BatchNorm2d(512),
                #     nn.ReLU()
                # )
                # model_to_512.to(feat_src.device)
                # feat_src = model_to_512(feat_src)

                # predict on discriminator
                optimizer_critic1.zero_grad()
                real_loss = criterionAdv(Discriminator1(feat_src1), 0.9)#CSDN 说正 用0.9  负用0.1 效果更佳
                fake_loss = criterionAdv(Discriminator1(feat_tgt.detach()), 0.1)
                loss_critic1 = 0.5 * (real_loss + fake_loss)

                if current_epoch >= args.dongjie_epochs:
                    loss_critic1.backward()
                    optimizer_critic1.step()
                else:
                    for param in criterionAdv.parameters():
                        param.requires_grad = False
                    loss_critic1 = 0

                # ************************* confusion domain 2 with 1,3 **********************************#

                feat_src2 = torch.cat([pre_feat_ext2, pre_feat_ext2, pre_feat_ext2], 0)

                # predict on discriminator
                optimizer_critic2.zero_grad()
                real_loss = criterionAdv(Discriminator2(feat_src2), 0.9)
                fake_loss = criterionAdv(Discriminator2(feat_tgt.detach()), 0.1)
                loss_critic2 = 0.5 * (real_loss + fake_loss)

                if current_epoch >= args.dongjie_epochs:
                    loss_critic2.backward()
                    optimizer_critic2.step()
                else:
                    # for param in criterionAdv.parameters():
                    #     param.requires_grad = False
                    loss_critic2 = 0

                # ************************* confusion domain 3 with 1,2 **********************************#

                feat_src3 = torch.cat([pre_feat_ext3, pre_feat_ext3, pre_feat_ext3], 0)

                # predict on discriminator
                optimizer_critic3.zero_grad()
                real_loss = criterionAdv(Discriminator3(feat_src3), 0.9)
                fake_loss = criterionAdv(Discriminator3(feat_tgt.detach()), 0.1)
                loss_critic3 = 0.5 * (real_loss + fake_loss)

                if current_epoch >= args.dongjie_epochs:
                    loss_critic3.backward()
                    optimizer_critic3.step()
                else:
                    # for param in criterionAdv.parameters():
                    #     param.requires_grad = False
                    loss_critic3 = 0

                # ============ tensorboard the log info ============#
                if current_epoch >= args.dongjie_epochs:
                    info = {
                        'Loss_triplet': Loss_triplet.item(),
                        'loss_boxes': loss_boxes.item(),
                        'Loss_G': Loss_G.item(),

                        'loss_critic1': loss_critic1.item(),
                        'loss_generator1': loss_generator1.item(),
                        'loss_critic2': loss_critic2.item(),
                        'loss_generator2': loss_generator2.item(),
                        'loss_critic3': loss_critic3.item(),
                        'loss_generator3': loss_generator3.item(),
                    }
                else:
                    info = {
                        'Loss_triplet': Loss_triplet.item(),
                        'loss_boxes': loss_boxes.item(),
                        'Loss_G': Loss_G.item(),
                    }
                for tag, value in info.items():
                    summary_writer.add_scalar(tag, value, global_step)

                    # if (step+1) % args.tst_step == 0:
                    # depth_Pre_real = torch.cat([depth_Pre[0:args.batchsize],depth_Pre[2*args.batchsize:3*args.batchsize], depth_Pre[4*args.batchsize:5*args.batchsize]],0)
                    # depth_Pre_fake = torch.cat([depth_Pre[args.batchsize:2*args.batchsize],depth_Pre[3*args.batchsize:4*args.batchsize], depth_Pre[5*args.batchsize:6*args.batchsize]],0)
                    #
                    # depth_Pre_all = vutils.make_grid(depth_Pre, normalize=True, scale_each=True)
                    # depth_Pre_real = vutils.make_grid(depth_Pre_real, normalize=True, scale_each=True)
                    # depth_Pre_fake = vutils.make_grid(depth_Pre_fake, normalize=True, scale_each=True)

                    # summary_writer.add_image('Depth_Image_all', depth_Pre_all, global_step)
                    # summary_writer.add_image('Depth_Image_real', depth_Pre_real, global_step)
                    # summary_writer.add_image('Depth_Image_fake', depth_Pre_fake, global_step)

                # ============ print the log info ============#
                if current_epoch >= args.dongjie_epochs:
                    if (step + 1) % args.log_step == 0:
                        errors = OrderedDict([
                            ('Loss_triplet', Loss_triplet.item()),
                            ('Loss_G', Loss_G.item()),
                            ('Loss_yolo', loss_boxes.item()),

                            ('loss_critic1', loss_critic1.item()),
                            ('loss_generator1', loss_generator1.item()),
                            ('loss_critic2', loss_critic2.item()),
                            ('loss_generator2', loss_generator2.item()),
                            ('loss_critic3', loss_critic3.item()),
                            ('loss_generator3', loss_generator3.item())])

                        Saver.print_current_errors(epoch=(epoch + 1), i=(step + 1), errors=errors)
                else:
                    if (step + 1) % args.log_step == 0:
                        errors = OrderedDict([
                            ('Loss_triplet', Loss_triplet.item()),
                            ('Loss_G', Loss_G.item()),
                            ('Loss_yolo', loss_boxes.item()),
                        ]
                        )

                        Saver.print_current_errors(epoch=(epoch + 1), i=(step + 1), errors=errors)

                # if (step + 1) % args.tst_step == 0:
                #     evaluate.evaluate_img(args, FeatExtor, data_loader_target, (epoch + 1), (step + 1), Saver)

                global_step += 1

                #############################
                # 2.4 save model parameters #
                #############################
                # 创建一个新的OrderedDict，将model和FeatExtor的state_dict合并
                combined_state_dict = {}
                combined_state_dict.update(FeatExtor.state_dict())
                combined_state_dict.update(model.state_dict())

                if ((step + 1) % args.model_save_step == 0):
                    model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)
                    mkdir(model_save_path)



                    # 保存整个模型的state_dict
                    torch.save(combined_state_dict, os.path.join(model_save_path,
                                                                "DGFA-toal_yolo-{}-{}.pt".format(epoch + 1,
                                                                                             step + 1)))
                    torch.save(model.state_dict(), os.path.join(model_save_path,
                                                                    "DGFA-model-{}-{}.pt".format(epoch + 1,
                                                                                               step + 1)))
                    torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                                                                    "DGFA-Ext-{}-{}.pt".format(epoch + 1,
                                                                                               step + 1)))

                    torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                                                                     "DGFA-Embd-{}-{}.pt".format(epoch + 1,
                                                                                                 step + 1)))

                    torch.save(Discriminator1.state_dict(), os.path.join(model_save_path,
                                                                         "DGFA-D1-{}-{}.pt".format(epoch + 1,
                                                                                                   step + 1)))

                    torch.save(Discriminator2.state_dict(), os.path.join(model_save_path,
                                                                         "DGFA-D2-{}-{}.pt".format(epoch + 1,
                                                                                                   step + 1)))

                    torch.save(Discriminator3.state_dict(), os.path.join(model_save_path,
                                                                         "DGFA-D3-{}-{}.pt".format(epoch + 1,
                                                                                                   step + 1)))

            if ((epoch + 1) % args.model_save_epoch == 0):
                model_save_path = os.path.join(args.results_path, 'snapshots', savefilename)
                mkdir(model_save_path)
                torch.save(model.state_dict(), os.path.join(model_save_path,
                                                                "DGFA-model-{}.pt".format(epoch + 1)))

                torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                                                                "DGFA-Ext-{}.pt".format(epoch + 1)))

                torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                                                                 "DGFA-Embd-{}.pt".format(epoch + 1)))

                torch.save(Discriminator1.state_dict(), os.path.join(model_save_path,
                                                                     "DGFA-D1-{}.pt".format(epoch + 1)))

                torch.save(Discriminator2.state_dict(), os.path.join(model_save_path,
                                                                     "DGFA-D2-{}.pt".format(epoch + 1)))

                torch.save(Discriminator3.state_dict(), os.path.join(model_save_path,
                                                                     "DGFA-D3-{}.pt".format(epoch + 1)))

    ###########################################################wrq将Train.py的代码移动到这个train.py中！！！！！！！！！！！
####################################################################wrq将Train.py的代码移动到这个train.py中！！！！！！！！！！！

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

                # gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                #                  pin_memory=True,
                #                  drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                #                  worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                # # print(gen)
                #
                # gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                #                      pin_memory=True,
                #                      drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                #                      worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            # gen.dataset.epoch_now = epoch
            # gen_val.dataset.epoch_now = epoch

            # if distributed:
            #     train_sampler.set_epoch(epoch)

            # set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            # set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # fit_one_epoch(args, model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
            #               epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, maddg_loss,
            #               local_rank,)

            # if distributed:
            #     dist.barrier()

        ###########################################################wrq将Train.py的代码移动到这个train.py中！！！！！！！！！！！
        torch.save(model.state_dict(), os.path.join(model_save_path,
                                                        "DGFA-model-final.pt"))
        torch.save(FeatExtor.state_dict(), os.path.join(model_save_path,
                                                        "DGFA-Ext-final.pt"))
        torch.save(FeatEmbder.state_dict(), os.path.join(model_save_path,
                                                         "DGFA-Embd-final.pt"))
        torch.save(Discriminator1.state_dict(), os.path.join(model_save_path,
                                                             "DGFA-D1-final.pt"))
        torch.save(Discriminator2.state_dict(), os.path.join(model_save_path,
                                                             "DGFA-D2-final.pt"))
        torch.save(Discriminator3.state_dict(), os.path.join(model_save_path,
                                                             "DGFA-D3-final.pt"))
        ###########################################################wrq将Train.py的代码移动到这个train.py中！！！！！！！！！！！
        print("训练已完成")
