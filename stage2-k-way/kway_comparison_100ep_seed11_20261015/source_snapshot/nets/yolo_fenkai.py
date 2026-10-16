import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv, SimAM
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors


def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
                                                                    ###block_num=2,,hidc=256,是dy_head加上去的
    def __init__(self,backbone, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        # ---------------------------------------------------#
        # self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        # self.backbone = Backbone(phi, pretrained=pretrained)
        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
                                       base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth,
                                       shortcut=False)

        # 256, 80, 80 => 256, 40, 40
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
                                         shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                         int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        # ------------------------加强特征提取网络------------------------#

        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape = None
        self.nl = len(ch)
        # self.stride     = torch.zeros(self.nl)
        # self.stride = torch.tensor(
        #     [256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        # 用上面注释的代码去测试backbone得出的结果是3 16 8 # torch.Size([1, 128, 32, 32])，torch.Size([1, 256, 16, 16])，torch.Size([1, 512, 8, 8])，他们的-2的位置就是了
        self.stride = torch.tensor([256 / x for x in [32,16,8]])

        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        # #wrq 添加dy_head
        # self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        # self.dyhead = nn.Sequential(*[DyHeadBlock(hidc) for i in range(block_num)])


        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        # self.simam = SimAM(e_lambda=1e-4)
        # self.ema1 = EMA(channels=128,factor=8)
        # self.ema2 = EMA(channels=256,factor=8)
        # self.ema3 = EMA(channels=512,factor=8)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def forward(self, feat1, feat2, feat3):
    # def forward(self, x):
        # feat1, feat2, feat3 = self.backbone(x)
        # tri_feature = feat3  # tri_feature的形状是这样子的： torch.Size([8, 512, 20, 20])
        # ------------------------加强特征提取网络------------------------#
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3 = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        # ------------------------加强特征提取网络------------------------#
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = P3.shape  # BCHW

        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400;
        #                                           box self.reg_max * 4, 8400
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)


#
# class EMA(nn.Module):
#     def __init__(self, channels, factor=8):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channels // self.groups > 0
#         self.softmax = nn.Softmax(-1)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
#         self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)
#
#
# class DyHeadBlock(nn.Module):
#     """DyHead Block with three types of attention.
#     HSigmoid arguments in default act_cfg follow official code, not paper.
#     https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
#     """
#
#     def __init__(self,
#                  in_channels,
#                  norm_type='GN',
#                  zero_init_offset=True,
#                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
#         super().__init__()
#         self.zero_init_offset = zero_init_offset
#         # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
#         self.offset_and_mask_dim = 3 * 3 * 3
#         self.offset_dim = 2 * 3 * 3
#
#         if norm_type == 'GN':
#             norm_dict = dict(type='GN', num_groups=16, requires_grad=True)
#         elif norm_type == 'BN':
#             norm_dict = dict(type='BN', requires_grad=True)
#
#         from mmcv.cnn import build_activation_layer, build_norm_layer
#         self.spatial_conv_high = DyDCNv2(in_channels, in_channels, norm_cfg=norm_dict)
#         self.spatial_conv_mid = DyDCNv2(in_channels, in_channels)
#         self.spatial_conv_low = DyDCNv2(in_channels, in_channels, stride=2)
#         self.spatial_conv_offset = nn.Conv2d(
#             in_channels, self.offset_and_mask_dim, 3, padding=1)
#         self.scale_attn_module = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
#             nn.ReLU(inplace=True), build_activation_layer(act_cfg))
#         self.task_attn_module = DyReLU(in_channels)
#         self._init_weights()
#
#     from mmengine.model import normal_init
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, 0, 0.01)
#         if self.zero_init_offset:
#             constant_init(self.spatial_conv_offset, 0)
#
#     def forward(self, x):
#         """Forward function."""
#         outs = []
#         for level in range(len(x)):
#             # calculate offset and mask of DCNv2 from middle-level feature
#             offset_and_mask = self.spatial_conv_offset(x[level])
#             offset = offset_and_mask[:, :self.offset_dim, :, :]
#             mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
#
#             mid_feat = self.spatial_conv_mid(x[level], offset, mask)
#             sum_feat = mid_feat * self.scale_attn_module(mid_feat)
#             summed_levels = 1
#             if level > 0:
#                 low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
#                 sum_feat += low_feat * self.scale_attn_module(low_feat)
#                 summed_levels += 1
#             if level < len(x) - 1:
#                 # this upsample order is weird, but faster than natural order
#                 # https://github.com/microsoft/DynamicHead/issues/25
#                 high_feat = F.interpolate(
#                     self.spatial_conv_high(x[level + 1], offset, mask),
#                     size=x[level].shape[-2:],
#                     mode='bilinear',
#                     align_corners=True)
#                 sum_feat += high_feat * self.scale_attn_module(high_feat)
#                 summed_levels += 1
#             outs.append(self.task_attn_module(sum_feat / summed_levels))
#
#         return outs
#
#
# class DyDCNv2(nn.Module):
#     """ModulatedDeformConv2d with normalization layer used in DyHead.
#     This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
#     because DyHead calculates offset and mask from middle-level feature.
#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         stride (int | tuple[int], optional): Stride of the convolution.
#             Default: 1.
#         norm_cfg (dict, optional): Config dict for normalization layer.
#             Default: dict(type='GN', num_groups=16, requires_grad=True).
#     """
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  stride=1,
#                  norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
#         super().__init__()
#         self.with_norm = norm_cfg is not None
#         bias = not self.with_norm
#         self.conv = ModulatedDeformConv2d(
#             in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
#         if self.with_norm:
#             self.norm = build_norm_layer(norm_cfg, out_channels)[1]
#
#     def forward(self, x, offset, mask):
#         """Forward function."""
#         x = self.conv(x.contiguous(), offset, mask)
#         if self.with_norm:
#             x = self.norm(x)
#         return x
#
# class DyReLU(nn.Module):
#     def __init__(self, inp, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
#                  init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
#         super(DyReLU, self).__init__()
#         self.oup = inp
#         self.lambda_a = lambda_a * 2
#         self.K2 = K2
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         self.use_bias = use_bias
#         if K2:
#             self.exp = 4 if use_bias else 2
#         else:
#             self.exp = 2 if use_bias else 1
#         self.init_a = init_a
#         self.init_b = init_b
#
#         # determine squeeze
#         if reduction == 4:
#             squeeze = inp // reduction
#         else:
#             squeeze = _make_divisible(inp // reduction, 4)
#         # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
#         # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))
#
#         self.fc = nn.Sequential(
#             nn.Linear(inp, squeeze),
#             nn.ReLU(inplace=True),
#             nn.Linear(squeeze, self.oup * self.exp),
#             h_sigmoid()
#         )
#         if use_spatial:
#             self.spa = nn.Sequential(
#                 nn.Conv2d(inp, 1, kernel_size=1),
#                 nn.BatchNorm2d(1),
#             )
#         else:
#             self.spa = None
#
#     def forward(self, x):
#         if isinstance(x, list):
#             x_in = x[0]
#             x_out = x[1]
#         else:
#             x_in = x
#             x_out = x
#         b, c, h, w = x_in.size()
#         y = self.avg_pool(x_in).view(b, c)
#         y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
#         if self.exp == 4:
#             a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
#             a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#             a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
#
#             b1 = b1 - 0.5 + self.init_b[0]
#             b2 = b2 - 0.5 + self.init_b[1]
#             out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
#         elif self.exp == 2:
#             if self.use_bias:  # bias but not PL
#                 a1, b1 = torch.split(y, self.oup, dim=1)
#                 a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#                 b1 = b1 - 0.5 + self.init_b[0]
#                 out = x_out * a1 + b1
#
#             else:
#                 a1, a2 = torch.split(y, self.oup, dim=1)
#                 a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#                 a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
#                 out = torch.max(x_out * a1, x_out * a2)
#
#         elif self.exp == 1:
#             a1 = y
#             a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#             out = x_out * a1
#         import torch.nn.functional as F
#         if self.spa:
#             ys = self.spa(x_in).view(b, -1)
#             ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
#             ys = F.hardtanh(ys, 0, 3, inplace=True)/3
#             out = out * ys
#
#         return out