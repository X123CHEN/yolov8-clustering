# import torch
# import torch.nn as nn
# import math
#
#
# def autopad(k, p=None, d=1):
#     # kernel, padding, dilation
#     # 对输入的特征层进行自动padding，按照Same原则
#     if d > 1:
#         # actual kernel-size
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         # auto-pad
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p
#
# class SiLU(nn.Module):
#     # SiLU激活函数
#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)
#
# class Conv(nn.Module):
#     # 标准卷积+标准化+激活函数
#     default_act = SiLU()
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def forward_fuse(self, x):
#         return self.act(self.conv(x))
# #
# # class Bottleneck(nn.Module):
# #     # 标准瓶颈结构，残差结构
# #     # c1为输入通道数，c2为输出通道数
# #     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
# #         super().__init__()
# #         c_ = int(c2 * e)  # hidden channels
# #         self.cv1 = Conv(c1, c_, k[0], 1)
# #         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
# #         self.add = shortcut and c1 == c2
# #
# #     def forward(self, x):
# #         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# ######wrq 在bottleneck上增加simam注意力机制
# class Bottleneck(nn.Module):
#     # 标准瓶颈结构，残差结构
#     # c1为输入通道数，c2为输出通道数
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#         # self.simam = SimAM(e_lambda=1e-4)
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#         # return x + self.simam(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
#
# class C2f(nn.Module):
#     # CSPNet结构结构，大残差结构
#     # c1为输入通道数，c2为输出通道数
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         n = int(n)
#         self.c      = int(c2 * e)
#         self.cv1    = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2    = Conv(int((2 + n) * self.c), c2, 1)
#         self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#     def forward(self, x):
#         # 进行一个卷积，然后划分成两份，每个通道都为c
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
#         y.extend(m(y[-1]) for m in self.m)
#
#         return self.cv2(torch.cat(y, 1))
#
# class SPPF(nn.Module):
#     # SPP结构，5、9、13最大池化核的最大池化。
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_          = c1 // 2
#         self.cv1    = Conv(c1, c_, 1, 1)
#         self.cv2    = Conv(c_ * 4, c2, 1, 1)
#         self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#
#     def forward(self, x):
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
#
# class Backbone(nn.Module):#wrq修改过，适配。
#     # def __init__(self, base_channels, base_depth, deep_mul, phi='s', pretrained=False):
#     def __init__(self, phi='s', pretrained=False):
#         super().__init__()
#
#         depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
#         width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
#         deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
#         dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
#
#         base_channels = int(wid_mul * 64)  # 64
#         base_depth = max(round(dep_mul * 3), 1)  # 3
#         #-----------------------------------------------#
#         #   输入图片是3, 640, 640
#         #-----------------------------------------------#
#
#         # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
#         self.stem = Conv(3, base_channels, 3, 2)
#         # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
#         self.dark2 = nn.Sequential(
#             Conv(base_channels, base_channels * 2, 3, 2),
#             C2f(base_channels * 2, base_channels * 2, base_depth, True),
#         )
#         # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
#         self.dark3 = nn.Sequential(
#             Conv(base_channels * 2, base_channels * 4, 3, 2),
#             C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
#         )
#         # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
#         self.dark4 = nn.Sequential(
#             Conv(base_channels * 4, base_channels * 8, 3, 2),
#             C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
#         )
#         # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
#         self.dark5 = nn.Sequential(
#             Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
#             C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
#             SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
#         )
#         if pretrained:
#             print("backbone加载了预训练权重！！！")
#             local_weight_path = '/disk/home/wurx/yolov8_beifen/logs/202310120_200epoch_toNorway/best_epoch_weights.pth'
#             checkpoint = torch.load(local_weight_path, map_location="cpu")
#             self.load_state_dict(checkpoint, strict=False)
#             print("Loaded weights from " + local_weight_path)
#         # if pretrained:
#         #     url = {
#         #         "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
#         #         "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
#         #         "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
#         #         "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
#         #         "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
#         #     }[phi]
#         #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
#         #     self.load_state_dict(checkpoint, strict=False)
#         #     print("Load weights from " + url.split('/')[-1])
#
#     def forward(self, x):
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # x = x.to(device)
#         #输入 x  3 640 640
#         x = self.stem(x) #应该是64, 320, 320   实际却是torch.Size([32, 320, 320])
#         # print("self.stem(x)的大小", x.size())
#         x = self.dark2(x)#应该是128, 160, 160  实际却是torch.Size([64, 160, 160])  下面的特征的通道数都少了一半
#         # print("self.dark2(x)的大小", x.size())
#         #-----------------------------------------------#
#         #   dark3的输出为256, 80, 80，是一个有效特征层
#         #-----------------------------------------------#
#         x = self.dark3(x)
#         feat1 = x
#         # print("feat1的大小",feat1.size())
#         #-----------------------------------------------#
#         #   dark4的输出为512, 40, 40，是一个有效特征层
#         #-----------------------------------------------#
#         x = self.dark4(x)
#         feat2 = x  #feat2的大小 输出却是  torch.Size(([256, 40, 40]))
#         # print("feat2的大小", feat2.size())
#         #-----------------------------------------------#
#         #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
#         #-----------------------------------------------#
#         x = self.dark5(x)
#         feat3 = x #feat3的大小 输出却是  torch.Size(([512, 20, 20]))
#         # print("feat3的大小", feat3.size())
#         return feat1, feat2, feat3
#
# class SimAM(torch.nn.Module):
#     def __init__(self, e_lambda=1e-4):
#         super(SimAM, self).__init__()
#
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda
#
#     def __repr__(self):
#         s = self.__class__.__name__ + '('
#         s += ('lambda=%f)' % self.e_lambda)
#         return s
#
#     @staticmethod
#     def get_module_name():
#         return "simam"
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         n = w * h - 1
#
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#
#         return x * self.activaton(y)
#
#
# # class DCNv2(nn.Module):
# #     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
# #                  padding=1, dilation=1, groups=1, deformable_groups=1):
# #         super(DCNv2, self).__init__()
# #
# #         self.in_channels = in_channels
# #         self.out_channels = out_channels
# #         self.kernel_size = (kernel_size, kernel_size)
# #         self.stride = (stride, stride)
# #         self.padding = (padding, padding)
# #         self.dilation = (dilation, dilation)
# #         self.groups = groups
# #         self.deformable_groups = deformable_groups
# #
# #         self.weight = nn.Parameter(
# #             torch.empty(out_channels, in_channels, *self.kernel_size)
# #         )
# #         self.bias = nn.Parameter(torch.empty(out_channels))
# #
# #         out_channels_offset_mask = (self.deformable_groups * 3 *
# #                                     self.kernel_size[0] * self.kernel_size[1])
# #         self.conv_offset_mask = nn.Conv2d(
# #             self.in_channels,
# #             out_channels_offset_mask,
# #             kernel_size=self.kernel_size,
# #             stride=self.stride,
# #             padding=self.padding,
# #             bias=True,
# #         )
# #         self.bn = nn.BatchNorm2d(out_channels)
# #         self.act = Conv.default_act
# #         self.reset_parameters()
# #
# #     def forward(self, x):
# #         offset_mask = self.conv_offset_mask(x)
# #         o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
# #         offset = torch.cat((o1, o2), dim=1)
# #         mask = torch.sigmoid(mask)
# #         x = torch.ops.torchvision.deform_conv2d(
# #             x,
# #             self.weight,
# #             offset,
# #             mask,
# #             self.bias,
# #             self.stride[0], self.stride[1],
# #             self.padding[0], self.padding[1],
# #             self.dilation[0], self.dilation[1],
# #             self.groups,
# #             self.deformable_groups,
# #             True
# #         )
# #         x = self.bn(x)
# #         x = self.act(x)
# #         return x
# #
# #     def reset_parameters(self):
# #         n = self.in_channels
# #         for k in self.kernel_size:
# #             n *= k
# #         std = 1. / math.sqrt(n)
# #         self.weight.data.uniform_(-std, std)
# #         self.bias.data.zero_()
# #         self.conv_offset_mask.weight.data.zero_()
# #         self.conv_offset_mask.bias.data.zero_()
# #
# # class Bottleneck_DCN(nn.Module):
# #     # Standard bottleneck with DCN
# #     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
# #         super().__init__()
# #         c_ = int(c2 * e)  # hidden channels
# #         if k[0] == 3:
# #             self.cv1 = DCNv2(c1, c_, k[0], 1)
# #         else:
# #             self.cv1 = Conv(c1, c_, k[0], 1)
# #         if k[1] == 3:
# #             self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g)
# #         else:
# #             self.cv2 = Conv(c_, c2, k[1], 1, g=g)
# #         self.add = shortcut and c1 == c2
# #         # self.ema = EMA(channels=c2,factor=8)
# #
# #     def forward(self, x):
# #         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# #
# # class C2f_DCN(nn.Module):
# #     # CSP Bottleneck with 2 convolutions
# #     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
# #         super().__init__()
# #         self.c = int(c2 * e)  # hidden channels
# #         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
# #         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
# #         self.m = nn.ModuleList(Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
# #
# #     def forward(self, x):
# #         y = list(self.cv1(x).split((self.c, self.c), 1))
# #         y.extend(m(y[-1]) for m in self.m)
# #         return self.cv2(torch.cat(y, 1))
#
#
#
# class C2f_DWR(C2f):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(DWR(self.c) for _ in range(n))
#
#
# class DWR(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#
#         self.conv_3x3 = Conv(dim, dim // 2, 3)
#
#         self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
#         self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
#         self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)
#
#         self.conv_1x1 = Conv(dim * 2, dim, k=1)
#
#     def forward(self, x):
#         conv_3x3 = self.conv_3x3(x)
#         x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
#         x_out = torch.cat([x1, x2, x3], dim=1)
#         x_out = self.conv_1x1(x_out) + x
#         # print('C2f_DWR已经被使用')
#         return x_out

# import math
# import torch
# import torch.nn as nn
#
# def fuse_conv_and_bn(conv, bn):
#     # 混合Conv2d + BatchNorm2d 减少计算量
#     # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
#     fusedconv = nn.Conv2d(conv.in_channels,
#                           conv.out_channels,
#                           kernel_size=conv.kernel_size,
#                           stride=conv.stride,
#                           padding=conv.padding,
#                           dilation=conv.dilation,
#                           groups=conv.groups,
#                           bias=True).requires_grad_(False).to(conv.weight.device)
#
#     # 准备kernel
#     w_conv = conv.weight.clone().view(conv.out_channels, -1)
#     w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
#     fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
#
#     # 准备bias
#     b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
#     b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
#     fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
#
#     return fusedconv
#
#
# def autopad(k, p=None, d=1):
#     # kernel, padding, dilation
#     # 对输入的特征层进行自动padding，按照Same原则
#     if d > 1:
#         # actual kernel-size
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         # auto-pad
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p
#
# class SiLU(nn.Module):
#     # SiLU激活函数
#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)
#
# class Conv(nn.Module):
#     # 标准卷积+标准化+激活函数
#     default_act = SiLU()
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
#         self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def forward_fuse(self, x):
#         return self.act(self.conv(x))
#
# class Bottleneck(nn.Module):
#     # 标准瓶颈结构，残差结构
#     # c1为输入通道数，c2为输出通道数
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
# # ######wrq 在bottleneck上增加simam注意力机制
# # class Bottleneck(nn.Module):
# #     # 标准瓶颈结构，残差结构
# #     # c1为输入通道数，c2为输出通道数
# #     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
# #         super().__init__()
# #         c_ = int(c2 * e)  # hidden channels
# #         self.cv1 = Conv(c1, c_, k[0], 1)
# #         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
# #         self.add = shortcut and c1 == c2
# #         self.simam = SimAM(e_lambda=1e-4)
# #
# #     def forward(self, x):
# #         print("################################################")
# #         return x + self.simam(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
#
# class C2f(nn.Module):
#     # CSPNet结构结构，大残差结构
#     # c1为输入通道数，c2为输出通道数
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         n = int(n)
#         self.c      = int(c2 * e)
#         self.cv1    = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2    = Conv(int((2 + n) * self.c), c2, 1)
#         self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         # 进行一个卷积，然后划分成两份，每个通道都为c
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
# class SPPF(nn.Module):
#     # SPP结构，5、9、13最大池化核的最大池化。
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_          = c1 // 2
#         self.cv1    = Conv(c1, c_, 1, 1)
#         self.cv2    = Conv(c_ * 4, c2, 1, 1)
#         self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#
#     def forward(self, x):
#         x = self.cv1(x)
#         y1 = self.m(x)
#         y2 = self.m(y1)
#         return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
#
# class Backbone(nn.Module):#wrq修改过，适配。
#     # def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
#     def __init__(self, phi='s', pretrained=False):
#         super().__init__()
#         depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
#         width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
#         deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
#         dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
#
#         base_channels = int(wid_mul * 64)  # 64
#         base_depth = max(round(dep_mul * 3), 1)  # 3
#         # base_channels =64
#         # base_depth =3
#         # deep_mul = 0.5
#         #-----------------------------------------------#
#         #   输入图片是3, 640, 640
#         #-----------------------------------------------#
#
#         # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
#         self.stem = Conv(3, base_channels, 3, 2)
#         # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
#         self.dark2 = nn.Sequential(
#             Conv(base_channels, base_channels * 2, 3, 2),
#             C2f(base_channels * 2, base_channels * 2, base_depth, True),
#         )
#         # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
#         self.dark3 = nn.Sequential(
#             Conv(base_channels * 2, base_channels * 4, 3, 2),
#             C2f_DCN(base_channels * 4, base_channels * 4, base_depth * 2, True),
#
#         )
#         # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
#         self.dark4 = nn.Sequential(
#             Conv(base_channels * 4, base_channels * 8, 3, 2),
#             C2f_DCN(base_channels * 8, base_channels * 8, base_depth * 2, True),
#         )
#         # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
#         self.dark5 = nn.Sequential(
#             Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
#             C2f_DCN(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
#             SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5),
#         )
#
#         if pretrained:
#             local_weight_path = r"E:\Project\yolov8-pytorch-master\yolov8_beifen\best_epoch_weights.pth"
#             checkpoint = torch.load(local_weight_path, map_location="cpu")
#             self.load_state_dict(checkpoint, strict=False)
#             print("Loaded weights from " + local_weight_path)
#
#         # if pretrained:
#         #     url = {
#         #         "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
#         #         "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
#         #         "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
#         #         "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
#         #         "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
#         #     }[phi]
#         #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
#         #     self.load_state_dict(checkpoint, strict=False)
#         #     print("Load weights from " + url.split('/')[-1])
#
#     # wrq为了 yolov8验证的  加的def fuse(self):
#     # def fuse(self):
#     #     print('Fusing layers... ')
#     #     for m in self.modules():
#     #         if type(m) is Conv and hasattr(m, 'bn'):
#     #             m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
#     #             delattr(m, 'bn')  # remove batchnorm
#     #             m.forward = m.forward_fuse  # update forward
#     #     return self
#
#     def forward(self, x):
#         # x1=x
#         # x = x.to('cuda')  # 将输入张量移动到GPU上
#         #输入 x 没错是  3 640 640
#         x = self.stem(x) #应该是64, 320, 320   实际却是torch.Size([32, 320, 320])
#         # print("self.stem(x)的大小", x.size())
#         x = self.dark2(x)#应该是128, 160, 160  实际却是torch.Size([64, 160, 160])  下面的特征的通道数都少了一半
#         # print("self.dark2(x)的大小", x.size())
#         #-----------------------------------------------#
#         #   dark3的输出为256, 80, 80，是一个有效特征层
#         #-----------------------------------------------#
#         x = self.dark3(x)
#         feat1 = x
#         # print("feat1的大小",feat1.size())
#         #-----------------------------------------------#
#         #   dark4的输出为512, 40, 40，是一个有效特征层
#         #-----------------------------------------------#
#         x = self.dark4(x)
#         feat2 = x  #feat2的大小 输出却是  torch.Size(([256, 40, 40]))
#         # print("feat2的大小", feat2.size())
#         #-----------------------------------------------#
#         #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
#         #-----------------------------------------------#
#         x = self.dark5(x)
#         feat3 = x #feat3的大小 输出却是  torch.Size(([512, 20, 20]))
#         # print("feat3的大小", feat3.size())
#         return feat1, feat2, feat3
#
#
#
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
# class SimAM(torch.nn.Module):
#     def __init__(self, e_lambda=1e-4):
#         super(SimAM, self).__init__()
#
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda
#
#     def __repr__(self):
#         s = self.__class__.__name__ + '('
#         s += ('lambda=%f)' % self.e_lambda)
#         return s
#
#     @staticmethod
#     def get_module_name():
#         return "simam"
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         n = w * h - 1
#
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#
#         return x * self.activaton(y)
#
# class DCNv2(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=1, dilation=1, groups=1, deformable_groups=1):
#         super(DCNv2, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         self.padding = (padding, padding)
#         self.dilation = (dilation, dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#
#         self.weight = nn.Parameter(
#             torch.empty(out_channels, in_channels, *self.kernel_size)
#         )
#         self.bias = nn.Parameter(torch.empty(out_channels))
#
#         out_channels_offset_mask = (self.deformable_groups * 3 *
#                                     self.kernel_size[0] * self.kernel_size[1])
#         self.conv_offset_mask = nn.Conv2d(
#             self.in_channels,
#             out_channels_offset_mask,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             bias=True,
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = Conv.default_act
#         self.reset_parameters()
#
#     def forward(self, x):
#         offset_mask = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#         x = torch.ops.torchvision.deform_conv2d(
#             x,
#             self.weight,
#             offset,
#             mask,
#             self.bias,
#             self.stride[0], self.stride[1],
#             self.padding[0], self.padding[1],
#             self.dilation[0], self.dilation[1],
#             self.groups,
#             self.deformable_groups,
#             True
#         )
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         std = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-std, std)
#         self.bias.data.zero_()
#         self.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.bias.data.zero_()
#
# class Bottleneck_DCN(nn.Module):
#     # Standard bottleneck with DCN
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         if k[0] == 3:
#             self.cv1 = DCNv2(c1, c_, k[0], 1)
#         else:
#             self.cv1 = Conv(c1, c_, k[0], 1)
#         if k[1] == 3:
#             self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g)
#         else:
#             self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
# class C2f_DCN(nn.Module):
#     # CSP Bottleneck with 2 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))


import numpy as np
import torch
from torch import nn
from torch.nn import init
import math


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x.to('cuda')
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


#
# class Bottleneck(nn.Module):
#     # 标准瓶颈结构，残差结构
#     # c1为输入通道数，c2为输出通道数
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
######wrq 在bottleneck上增加simam注意力机制
class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        # self.simam = SimAM(e_lambda=1e-4)

    def forward(self, x):
        # return x + self.simam(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        n = int(n)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(int((2 + n) * self.c), c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Backbone(nn.Module):  # wrq修改过，适配。
    # def __init__(self, base_channels, base_depth, deep_mul, phi='s', pretrained=False):
    def __init__(self, phi='s', pretrained=False):
        super().__init__()

        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        if pretrained:
            print("backbone加载了预训练权重！！！")
            local_weight_path = '/disk/home/wurx/yolov8_beifen/logs/202310120_200epoch_toNorway/best_epoch_weights.pth'
            checkpoint = torch.load(local_weight_path, map_location="cpu")
            self.load_state_dict(checkpoint, strict=False)
            print("Loaded weights from " + local_weight_path)
        # if pretrained:
        #     url = {
        #         "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
        #         "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
        #         "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
        #         "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
        #         "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
        #     }[phi]
        #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
        #     self.load_state_dict(checkpoint, strict=False)
        #     print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("####################",device)
        x = x.to(device)  ###TODU  这里修改 使用 train和train_beifen的时候需要注意
        # 输入 x 没错是  3 640 640
        x = self.stem(x) # 应该是64, 320, 320   实际却是torch.Size([32, 320, 320])
        # print("self.stem(x)的大小", x.size())
        x = self.dark2(x)  # 应该是128, 160, 160  实际却是torch.Size([64, 160, 160])  下面的特征的通道数都少了一半
        # print("self.dark2(x)的大小", x.size())
        # -----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # print("feat1的大小",feat1.size())
        # -----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x  # feat2的大小 输出却是  torch.Size(([256, 40, 40]))
        # print("feat2的大小", feat2.size())
        # -----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x  # feat3的大小 输出却是  torch.Size(([512, 20, 20]))
        # print("feat3的大小", feat3.size())
        return feat1, feat2, feat3

# backbone = Backbone(phi="s",pretrained=False)
# for x in backbone.forward(torch.zeros(1, 3, 256, 256)):
#     print(x.shape)

class Backbone_pre(nn.Module):  # 给预训练模型用的，目的不需要每次更换backbone的时候就去重新与训练一次
    # def __init__(self, base_channels, base_depth, deep_mul, phi='s', pretrained=False):
    def __init__(self, phi='s', pretrained=False):
        super().__init__()

        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )
        if pretrained:
            print("backbone加载了预训练权重！！！")
            local_weight_path = '/disk/home/wurx/yolov8_beifen/logs/202310120_200epoch_toNorway/best_epoch_weights.pth'
            checkpoint = torch.load(local_weight_path, map_location="cpu")
            self.load_state_dict(checkpoint, strict=False)
            print("Loaded weights from " + local_weight_path)
        # if pretrained:
        #     url = {
        #         "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
        #         "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
        #         "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
        #         "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
        #         "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
        #     }[phi]
        #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
        #     self.load_state_dict(checkpoint, strict=False)
        #     print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("####################",device)
        x = x.to(device)  ###TODU  这里修改 使用 train和train_beifen的时候需要注意
        # 输入 x 没错是  3 640 640
        x = self.stem(x) # 应该是64, 320, 320   实际却是torch.Size([32, 320, 320])
        # print("self.stem(x)的大小", x.size())
        x = self.dark2(x)  # 应该是128, 160, 160  实际却是torch.Size([64, 160, 160])  下面的特征的通道数都少了一半
        # print("self.dark2(x)的大小", x.size())
        # -----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # print("feat1的大小",feat1.size())
        # -----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x  # feat2的大小 输出却是  torch.Size(([256, 40, 40]))
        # print("feat2的大小", feat2.size())
        # -----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x  # feat3的大小 输出却是  torch.Size(([512, 20, 20]))
        # print("feat3的大小", feat3.size())
        return feat1, feat2, feat3

# backbone = Backbone(phi="s",pretrained=False)
# for x in backbone.forward(torch.zeros(1, 3, 256, 256)):
#     print(x.shape)


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

#
# class DCNv2(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=1, dilation=1, groups=1, deformable_groups=1):
#         super(DCNv2, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         self.padding = (padding, padding)
#         self.dilation = (dilation, dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#
#         self.weight = nn.Parameter(
#             torch.empty(out_channels, in_channels, *self.kernel_size)
#         )
#         self.bias = nn.Parameter(torch.empty(out_channels))
#
#         out_channels_offset_mask = (self.deformable_groups * 3 *
#                                     self.kernel_size[0] * self.kernel_size[1])
#         self.conv_offset_mask = nn.Conv2d(
#             self.in_channels,
#             out_channels_offset_mask,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             bias=True,
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = Conv.default_act
#         self.reset_parameters()
#
#     def forward(self, x):
#         offset_mask = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#         x = torch.ops.torchvision.deform_conv2d(
#             x,
#             self.weight,
#             offset,
#             mask,
#             self.bias,
#             self.stride[0], self.stride[1],
#             self.padding[0], self.padding[1],
#             self.dilation[0], self.dilation[1],
#             self.groups,
#             self.deformable_groups,
#             True
#         )
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         std = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-std, std)
#         self.bias.data.zero_()
#         self.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.bias.data.zero_()
#
#
# class Bottleneck_DCN(nn.Module):
#     # Standard bottleneck with DCN
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         if k[0] == 3:
#             self.cv1 = DCNv2(c1, c_, k[0], 1)
#         else:
#             self.cv1 = Conv(c1, c_, k[0], 1)
#         if k[1] == 3:
#             self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g)
#         else:
#             self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#         self.ema = EMA(channels=c2,factor=8)
#
#     def forward(self, x):
#         # print("self.ema = EMA(channels=c2,factor=8)")
#         return x + self.ema(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
#
#
# class C2f_DCN(nn.Module):
#     # CSP Bottleneck with 2 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

class C2f_DWR(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DWR(self.c) for _ in range(n))

class DWR(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3)

        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5)

        self.conv_1x1 = Conv(dim * 2, dim, k=1)

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
