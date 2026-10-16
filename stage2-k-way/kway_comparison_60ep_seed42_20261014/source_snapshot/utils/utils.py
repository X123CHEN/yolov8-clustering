import random
import os
import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
from torch.nn import init
import torch.nn as nn

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
        
def download_weights(phi, model_dir="./model_data"):
    import os

    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
        "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
        "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
        "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
        "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)


#wrq添加代码----------------------------------------------
def init_model(net, restore, init_type, init=True):
    """Init models with cuda and weights."""
    # init weights of model
    if init:
        init_weights(net, init_type)
    # restore model weights
    if restore is not None and os.path.exists(restore):
        print(restore)
        checkpoint = torch.load(restore, map_location='cpu')
        # for key in checkpoint:
        #     print(key)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if 'backbone.' in k:
                new_state_dict[k[9:]] = v

        ### create new OrderedDict that does not contain `module.`
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # ks = []
        # for k, v in state_dict.items():
        #     name = k[7:]  #    remove `module.`  解决预训练模型中， 因为多gpu并行运行，参数前面会多出 module的问题
            # new_state_dict[name] = v
        #     ks.append(k)
        # ks2 = []
        # for k, v in net.state_dict().items():
        #     ks2.append(k)
        # print(ks2)
        # load params
        # new_state_dict = {}
        # for k, v in checkpoint.items():
        #     new_state_dict[k[9:]] = v
        net.load_state_dict(new_state_dict)

        # net.load_state_dict(torch.load(restore))
        net.restored = True
        print("*************Restore model from: {}".format(os.path.abspath(restore)))

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'orthogonal_rnn':
        net.apply(weights_init_orthogonal_rnn)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):  # 检查是否是卷积层
        init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):  # 检查是否是全连接层
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):  # 检查是否是批量归一化层
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal_rnn(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal_(m.all_weights[0][0], gain=1)
        init.orthogonal_(m.all_weights[0][1], gain=1)
        init.constant_(m.all_weights[0][2], 1)
        init.constant_(m.all_weights[0][3], 1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().detach().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)

