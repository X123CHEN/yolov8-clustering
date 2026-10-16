import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

from pdb import set_trace as st


class FeatEmbedder(nn.Module):#wrq为了适配三元组损失的输入尺寸（batchsize,128）修改编码器  使用了conv1x1放弃了conv3x3
    def __init__(self, in_channels, nettype=None):
        super(FeatEmbedder, self).__init__()

        self.nettype = nettype

        self.conv = nn.Sequential(
            conv1x1(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv1x1(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            conv1x1(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.conv(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        feat = x
        return F.normalize(feat, p=2, dim=1)

# class FeatEmbedder(nn.Module):
#     # wrq修改in_channels=128 为 512
#     # def __init__(self, embed_size, in_channels=128, nettype=None):
#     def __init__(self, in_channels=256, nettype=None):
#         super(FeatEmbedder, self).__init__()
#
#         self.nettype = nettype
#
#         self.conv = nn.Sequential(
#             conv3x3(in_channels, 128),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#
#             conv3x3(128, 256),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#
#             conv3x3(256, 512),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )
#
#         self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
#
#
#     def forward(self, x):
#         # print(x.size())
#         x = self.conv(x)
#         x = self.avgpooling(x)
#         x = x.view(x.size(0), -1)
#         feat = x # feat的size： torch.Size([6, 512])
#         # print("feat的size：", feat.size())
#
#         return F.normalize(feat, p=2, dim=1)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def deconv3x3(in_channels, out_channels, stride=2, padding=1, output_padding=1, bias=False):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        bias=bias)


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        # self.conv = conv3x3(in_channels, out_channels)
        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Downconv(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 196),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x


class DOWN(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(DOWN, self).__init__()
        self.mpconv = nn.Sequential(
            Downconv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x




def TripletLossCal(args, feat_ext, lab1, lab2, lab3):
    criterionTri_inter = TripletLoss(margin=0.5)
    # criterionTri_inter = TripletLoss(margin=0.1)
    criterionTri_intra = TripletLoss(margin=0.1)

    # avgpool = nn.AvgPool2d(kernel_size=32, stride=1)
    # feat_ext_pl = avgpool(feat_ext).squeeze()

    feat_embd1 = feat_ext[0:args.batchsize * 2]
    feat_embd2 = feat_ext[args.batchsize * 2:args.batchsize * 4]
    feat_embd3 = feat_ext[args.batchsize * 4:args.batchsize * 6]

    #跨域损失，  1 2 3  三个源域
    ########## 1.1 cross-domain triplet loss #########
    loss_Tri_12 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2], 0), torch.cat([lab1, lab2], 0))[0]
    loss_Tri_23 = criterionTri_inter(torch.cat([feat_embd2, feat_embd3], 0), torch.cat([lab2, lab3], 0))[0]
    loss_Tri_13 = criterionTri_inter(torch.cat([feat_embd1, feat_embd3], 0), torch.cat([lab1, lab3], 0))[0]
    loss_tri_inter = loss_Tri_12 + loss_Tri_23 + loss_Tri_13

    #域内损失 1 2 3 自己跟自己的损失
    ########### 1.2 intra-domain triplet loss #########
    loss_Tri_1 = criterionTri_intra(feat_embd1, lab1)[0]
    loss_Tri_2 = criterionTri_intra(feat_embd2, lab2)[0]
    loss_Tri_3 = criterionTri_intra(feat_embd3, lab3)[0]
    loss_tri_intra = loss_Tri_1 + loss_Tri_2 + loss_Tri_3

    #parser.add_argument('--W_intra', type=int, default=0.1)
    Loss_triplet = loss_tri_inter + args.W_intra * loss_tri_intra  #args.W_intra  这是什么？


    return Loss_triplet




class TripletLoss(object):

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
#
# def hard_example_mining(dist_mat, labels, return_inds=False):
#     """For each anchor, find the hardest positive and negative sample.
#     Args:
#       dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#       labels: pytorch LongTensor, with shape [N]
#       return_inds: whether to return the indices. Save time if `False`
#     Returns:
#       dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#       dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#       p_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#       n_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#     NOTE: Only consider the case in which all labels have same num of samples,
#       thus we can cope with all anchors in parallel.
#     """
#
#     assert len(dist_mat.size()) == 2
#     assert dist_mat.size(0) == dist_mat.size(1)
#     N = dist_mat.size(0)
#
#     # shape [N, N]
#     is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#     is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
#
#     # `dist_ap` means distance(anchor, positive)
#     # both `dist_ap` and `relative_p_inds` with shape [N, 1]
#     dist_ap, relative_p_inds = torch.max(
#         dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
#     # `dist_an` means distance(anchor, negative)
#     # both `dist_an` and `relative_n_inds` with shape [N, 1]
#     dist_an, relative_n_inds = torch.min(
#         dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#     # shape [N]
#     dist_ap = dist_ap.squeeze(1)
#     dist_an = dist_an.squeeze(1)
#
#     if return_inds:
#         # shape [N, N]
#         ind = (labels.new().resize_as_(labels)
#                .copy_(torch.arange(0, N).long())
#                .unsqueeze(0).expand(N, N))
#         # shape [N, 1]
#         p_inds = torch.gather(
#             ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
#         n_inds = torch.gather(
#             ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
#         # shape [N]
#         p_inds = p_inds.squeeze(1)
#         n_inds = n_inds.squeeze(1)
#         return dist_ap, dist_an, p_inds, n_inds
#
#     return dist_ap, dist_an


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist




class Discriminator1(nn.Module):
    def __init__(self, nc=128, ndf=128, input_channels=128):
        super(Discriminator1, self).__init__()

        self.fea_project = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # feat1: 128x80x80 => 128x80x80

        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 128 10
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # 256, 5
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 512, 3
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4   （4*4的卷积核）第三个位置是卷积核大小
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False), # 1, 2

        )

    def forward(self, x):  # x: [B, 128, 80, 80]
        # output = self.model(x)
        x = self.fea_project(x) # torch.Size([6, 128, 42, 42])
        output = self.model(x)   #TODO
        return output




class Discriminator2(nn.Module):
    def __init__(self, nc=128, ndf=128, input_channels=128):
        # def __init__(self, nc=128, ndf=128):
        super(Discriminator2, self).__init__()

        self.fea_project = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # feat1: 128x80x80 => 128x80x80

        # self.model = nn.Sequential(
        #     # input is (nc) x 32 x 32
        #     nn.BatchNorm2d(nc),  ####wrq根据  csdn建议 在卷积之前家BN
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 16 x 16
        #
        #     nn.BatchNorm2d(ndf),
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # state size. (ndf*2) x 8 x 8
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 4 x 4
        #     nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
        #
        # )
        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4   （4*4的卷积核）第三个位置是卷积核大小
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),

        )

    def forward(self, x):
        # print(x.size())
        x = self.fea_project(x)
        output = self.model(x)
        return output

class Discriminator3(nn.Module):
    def __init__(self, nc=128, ndf=128, input_channels=128):
        # def __init__(self, nc=128, ndf=128):
        super(Discriminator3, self).__init__()

        self.fea_project = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # feat1: 128x80x80 => 128x80x80
        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4   （4*4的卷积核）第三个位置是卷积核大小
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),

        )
    def forward(self, x):
        output = self.model(self.fea_project(x))
        return output
#
# class Discriminator1(nn.Module):
#     def __init__(self, nc=128, ndf=128):
#         super(Discriminator1, self).__init__()
#
#         self.fea_project = nn.Sequential(
#             #out_channels=128, 由128改为 512
#             nn.ConvTranspose2d(in_channels=512, out_channels=nc, kernel_size=4, stride=2,
#                                padding=0),
#             nn.BatchNorm2d(nc),
#             nn.ReLU()
#         ) # 512x20x20 => 128 x 32 x 32
#
#         self.model = nn.Sequential(
#             # input is (nc) x 32 x 32
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 16 x 16
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 8 x 8
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 4 x 4
#             nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
#         )
#         # self.model = nn.Sequential(
#         #     # input is (nc) x 32 x 32
#         #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(ndf),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (ndf) x 16 x 16
#         #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(ndf * 2),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (ndf*2) x 8 x 8
#         #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(ndf * 4),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (ndf*4) x 4 x 4   （4*4的卷积核）第三个位置是卷积核大小
#         #     # wrq修改卷积核大小4*4 改为 3*3
#         #     # nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
#         #     nn.Conv2d(ndf * 4, 1, 3, 1, 1, bias=False),
#         #
#         # )
#
#     def forward(self, x):#x 的size 是 6 * 512 * 20 * 20
#         # output = self.model(x)
#         x = self.fea_project(x) # torch.Size([6, 128, 42, 42])
#         output = self.model(x)   #TODO
#         return output
#
#
# class Discriminator2(nn.Module):
#     def __init__(self, nc=128, ndf=128):
#         # def __init__(self, nc=128, ndf=128):
#         super(Discriminator2, self).__init__()
#
#         self.fea_project = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2,
#                                padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )  # 512x20x20 => 128 x 32 x 32
#         self.model = nn.Sequential(
#             # input is (nc) x 32 x 32
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 16 x 16
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 8 x 8
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 4 x 4   （4*4的卷积核）第三个位置是卷积核大小
#             # wrq修改卷积核大小4*4 改为 3*3
#             nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
#             # nn.Conv2d(ndf * 4, 1, 3, 1, 1, bias=False),
#
#         )
#
#     def forward(self, x):
#         # print(x.size())
#         output = self.model(self.fea_project(x))
#         return output
#
#
# class Discriminator3(nn.Module):
#     def __init__(self, nc=128, ndf=128):
#         # def __init__(self, nc=128, ndf=128):
#         super(Discriminator3, self).__init__()
#
#         self.fea_project = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2,
#                                padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )  # 512x20x20 => 128 x 32 x 32
#
#         self.model = nn.Sequential(
#             # input is (nc) x 32 x 32
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 16 x 16
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 8 x 8
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 4 x 4   （4*4的卷积核）第三个位置是卷积核大小
#             # wrq修改卷积核大小4*4 改为 3*3
#             nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
#             # nn.Conv2d(ndf * 4, 1, 3, 1, 1, bias=False),
#
#         )
#
#     def forward(self, x):
#         output = self.model(self.fea_project(x))
#         return output

def TripletLossCal1(args, feat_ext, lab1, lab2, lab3):
    criterionTri_inter = TripletLoss(margin=0.5)
    criterionTri_intra = TripletLoss(margin=0.1)

    # avgpool = nn.AvgPool2d(kernel_size=32, stride=1)
    # feat_ext_pl = avgpool(feat_ext).squeeze()

    feat_embd1 = feat_ext[0:args.batchsize * 2]
    feat_embd2 = feat_ext[args.batchsize * 2:args.batchsize * 4]
    feat_embd3 = feat_ext[args.batchsize * 4:args.batchsize * 6]

    ########## 1.1 cross-domain triplet loss #########
    loss_Tri_12 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2], 0), torch.cat([lab1, lab2], 0))[0]
    loss_Tri_23 = criterionTri_inter(torch.cat([feat_embd2, feat_embd3], 0), torch.cat([lab2, lab3], 0))[0]
    loss_Tri_13 = criterionTri_inter(torch.cat([feat_embd1, feat_embd3], 0), torch.cat([lab1, lab3], 0))[0]
    loss_tri_inter = loss_Tri_12 + loss_Tri_23 + loss_Tri_13

    ########### 1.2 intra-domain triplet loss #########
    loss_Tri_1 = criterionTri_intra(feat_embd1, lab1)[0]
    loss_Tri_2 = criterionTri_intra(feat_embd2, lab2)[0]
    loss_Tri_3 = criterionTri_intra(feat_embd3, lab3)[0]
    loss_tri_intra = loss_Tri_1 + loss_Tri_2 + loss_Tri_3

    Loss_triplet = loss_tri_inter + args.W_intra * loss_tri_intra

    return Loss_triplet


def hard_example_mining(dist_mat, labels,return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    # # 先判断距离矩阵是不是二维，若不是二维则报错
    # # 判断距离矩阵是否是方阵，若不是方阵则报错
    # # shape [N, N]
    # is_pos = labels.view(N, 1).expand(N, N).eq(labels.view(N, 1).expand(N, N).t()).float()
    # is_neg = labels.view(N, 1).expand(N, N).ne(labels.view(N, 1).expand(N, N).t()).float()
    # # `dist_ap` means distance(anchor, positive)
    # # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # # x =dist_mat[is_pos]
    # dist_ap, _ = torch.max(dist_mat * is_pos, dim=1, keepdim=True)
    # # `dist_an` means distance(anchor, negative)
    # # both `dist_an` and `relative_n_inds` with shape [N]
    # dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1,keepdim=True)
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.view(N, 1).expand(N, N).eq(labels.view(N, 1).expand(N, N).t()).float().to(dist_mat.device)
    is_neg = labels.view(N, 1).expand(N, N).ne(labels.view(N, 1).expand(N, N).t()).float().to(dist_mat.device)
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # dist_ap, relative_p_inds = torch.max(dist_mat * is_pos, dim=1, keepdim=True)
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    # dist_an, relative_n_inds = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1, keepdim=True)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds
    return dist_ap, dist_an

class TripletLoss(object):


    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat)

        # dist_mat.size(),labels.size())
        # torch.Size([4, 4]) torch.Size([4])
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
