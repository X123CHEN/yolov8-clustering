from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import torch.nn.functional as F


def TripletLossCal(args, feat_ext1, feat_ext2, feat_ext3, lab1, lab2, lab3):
    criterionTri_inter = TripletLoss(margin=0.5)
    criterionTri_intra = TripletLoss(margin=0.1)
    # criterionTri_inter = TripletLoss(margin=0)
    # criterionTri_intra = TripletLoss(margin=0)

    # avgpool = nn.AvgPool2d(kernel_size=32, stride=1)
    # feat_ext_pl = avgpool(feat_ext).squeeze()

    # feat_embd1 = feat_ext[0:args.batch_size * 2]
    # feat_embd2 = feat_ext[args.batch_size * 2:args.batch_size * 4]
    # feat_embd3 = feat_ext[args.batch_size * 4:args.batch_size * 6]
    feat_embd1 = feat_ext1
    feat_embd2 = feat_ext2
    feat_embd3 = feat_ext3

    ########## 1.1 cross-domain triplet loss #########
    # 域域
    loss_Tri_12 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2], 0), torch.cat([lab1, lab2], 0))[0]
    loss_Tri_23 = criterionTri_inter(torch.cat([feat_embd2, feat_embd3], 0), torch.cat([lab2, lab3], 0))[0]
    loss_Tri_13 = criterionTri_inter(torch.cat([feat_embd1, feat_embd3], 0), torch.cat([lab1, lab3], 0))[0]
    loss_Tri_123 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2, feat_embd3], 0), torch.cat([lab1, lab2, lab3], 0))[0]
    loss_tri_inter = loss_Tri_12 + loss_Tri_23 + loss_Tri_13 + loss_Tri_123
    # loss_tri_inter = loss_Tri_123

    ########### 1.2 intra-domain triplet loss #########
    loss_Tri_1 = criterionTri_intra(feat_embd1, lab1)[0]
    loss_Tri_2 = criterionTri_intra(feat_embd2, lab2)[0]
    loss_Tri_3 = criterionTri_intra(feat_embd3, lab3)[0]
    loss_tri_intra = loss_Tri_1 + loss_Tri_2 + loss_Tri_3

    Loss_triplet = loss_tri_inter + args.W_intra * loss_tri_intra

    return Loss_triplet

def TripletLossCal_double(args, feat_ext1, feat_ext2, lab1, lab2):
    # criterionTri_inter = TripletLoss(margin=0.5)
    # criterionTri_intra = TripletLoss(margin=0.1)
    criterionTri_inter = TripletLoss(margin=0)
    criterionTri_intra = TripletLoss(margin=0)

    # avgpool = nn.AvgPool2d(kernel_size=32, stride=1)
    # feat_ext_pl = avgpool(feat_ext).squeeze()

    # feat_embd1 = feat_ext[0:args.batch_size * 2]
    # feat_embd2 = feat_ext[args.batch_size * 2:args.batch_size * 4]
    # feat_embd3 = feat_ext[args.batch_size * 4:args.batch_size * 6]

    feat_embd1 = feat_ext1
    feat_embd2 = feat_ext2
    # feat_embd3 = feat_ext3

    ########## 1.1 cross-domain triplet loss #########
    # 域域
    loss_Tri_12 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2], 0), torch.cat([lab1, lab2], 0))[0]
    # loss_Tri_23 = criterionTri_inter(torch.cat([feat_embd2, feat_embd3], 0), torch.cat([lab2, lab3], 0))[0]
    # loss_Tri_13 = criterionTri_inter(torch.cat([feat_embd1, feat_embd3], 0), torch.cat([lab1, lab3], 0))[0]
    # loss_Tri_123 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2, feat_embd3], 0), torch.cat([lab1, lab2, lab3], 0))[0]
    loss_Tri_123 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2], 0), torch.cat([lab1, lab2], 0))[0]
    loss_tri_inter = loss_Tri_12 + loss_Tri_123
    # loss_tri_inter = loss_Tri_123

    ########### 1.2 intra-domain triplet loss #########
    loss_Tri_1 = criterionTri_intra(feat_embd1, lab1)[0]
    loss_Tri_2 = criterionTri_intra(feat_embd2, lab2)[0]
    # loss_Tri_3 = criterionTri_intra(feat_embd3, lab3)[0]
    # loss_tri_intra = loss_Tri_1 + loss_Tri_2 + loss_Tri_3
    loss_tri_intra = loss_Tri_1 + loss_Tri_2

    Loss_triplet = loss_tri_inter + args.W_intra * loss_tri_intra

    return Loss_triplet


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


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        print(self.num_classes,'self.num_classes----------------------')
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1, target_fake_label=0, use_lsgan=True,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_var
        # target_tensor = None
        #
        # if len(target_is_real):  # wrq：  这个len 有待商榷
        #     # if target_is_real:   #wrq：  这个len 有待商榷
        #
        #     # 将其变形为（6, 1, 1, 1）
        #     reshaped_tensor = target_is_real.view(6, 1, 1, 1).cuda()
        #
        #     # 扩展维度为（6, 1, 4, 4）
        #     target_tensor = reshaped_tensor.expand(-1, -1, 4, 4).cuda()
        return target_tensor


    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor) # [6, 512]







#WRQ添加检测的损失函数----------------------------------------------------------------------------------------------------
import pkg_resources as pkg
def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result
TORCH_1_10 = check_version(torch.__version__, '1.10.0')
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w  = feats[i].shape
        sx          = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy          = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx      = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    # 左上右下
    lt, rb  = torch.split(distance, 2, dim)
    x1y1    = anchor_points - lt
    x2y2    = anchor_points + rb
    if xywh:
        c_xy    = (x1y1 + x2y2) / 2
        wh      = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

import numpy as np
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

import math
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # 计算IOU损失
        # weight代表损失中标签应该有的置信度，0最小，1最大
        weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        # 计算预测框和真实框的重合程度
        iou         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # 然后1-重合程度，乘上应该有的置信度，求和后求平均。
        loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 计算DFL损失
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # 一个点一般不会处于anchor点上，一般是xx.xx。如果要用DFL的话，不可能直接一个cross_entropy就能拟合
        # 所以把它认为是相对于xx.xx左上角锚点与右下角锚点的距离 如果距离右下角锚点距离小，wl就小，左上角损失就小
        #                                                   如果距离左上角锚点距离小，wr就小，右下角损失就小
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # b, n_max_boxes, 8400 -> b, 8400
    fg_mask = mask_pos.sum(-2)
    # 如果有一个anchor被指派去预测多个真实框
    if fg_mask.max() > 1:
        # b, n_max_boxes, 8400
        mask_multi_gts      = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        # 如果有一个anchor被指派去预测多个真实框，首先计算这个anchor最重合的真实框
        # 然后做一个onehot
        # b, 8400
        max_overlaps_idx    = overlaps.argmax(1)
        # b, 8400, n_max_boxes
        is_max_overlaps     = F.one_hot(max_overlaps_idx, n_max_boxes)
        # b, n_max_boxes, 8400
        is_max_overlaps     = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        # b, n_max_boxes, 8400
        mask_pos            = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask             = mask_pos.sum(-2)
    # 找到每个anchor符合哪个gt
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos

class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # roll_out_thr为64
        self.roll_out_thr = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            anc_points (Tensor) : shape(num_total_anchors, 2)
            gt_labels (Tensor)  : shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor)  : shape(bs, n_max_boxes, 4)
            mask_gt (Tensor)    : shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor)  : shape(bs, num_total_anchors)
            target_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            target_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor)        : shape(bs, num_total_anchors)
        """
        # 获得batch_size
        self.bs = pd_scores.size(0)
        # 获得真实框中的最大框数量
        self.n_max_boxes = gt_bboxes.size(1)
        # 如果self.n_max_boxes大于self.roll_out_thr则roll_out
        self.roll_out = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # b, max_num_obj, 8400
        # mask_pos      满足在真实框内、是真实框topk最重合的正样本、满足mask_gt的锚点
        # align_metric  某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
        # overlaps      所有真实框和锚点的重合程度
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        # target_gt_idx     b, 8400     每个anchor符合哪个gt
        # fg_mask           b, 8400     每个anchor是否有符合的gt
        # mask_pos          b, max_num_obj, 8400    one_hot后的target_gt_idx
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # 指定目标到对应的anchor点上
        # b, 8400
        # b, 8400, 4
        # b, 8400, 80
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 乘上mask_pos，把不满足真实框满足的锚点的都置0
        align_metric *= mask_pos
        # 每个真实框对应的最大得分
        # b, max_num_obj
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        # 每个真实框对应的最大重合度
        # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        # 把每个真实框和先验点的得分乘上最大重合程度，再除上最大得分
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # target_scores作为正则的标签
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

class dectect_Loss:
    def __init__(self, model):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # model strides
        self.nc = model.num_classes  # number of classes
        self.no = model.no
        self.reg_max = model.reg_max

        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        self.proj = torch.arange(model.reg_max, dtype=torch.float)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            # 获得图像索引
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            # 对batch进行循环，然后赋值
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # 缩放到原图大小。
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c = pred_dist.shape
            # DFL的解码
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        # 然后解码获得预测框
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        # 获得使用的device
        device = preds[1].device
        # box, cls, dfl三部分的损失
        loss = torch.zeros(3, device=device)
        # 获得特征，并进行划分
        feats = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400;
        #                                               box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # 获得batch size与dtype
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # 获得输入图片大小
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]
        # 获得anchors点和步长对应的tensor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 把一个batch中的东西弄一个矩阵
        # 0为属于第几个图片
        # 1为种类
        # 2:为框的坐标
        targets = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
        # 先进行初步的处理，对输入进来的gt进行padding，到最大数量，并把框的坐标进行缩放
        # bs, max_boxes_num, 5
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 求哪些框是有目标的，哪些是填充的
        # bs, max_boxes_num
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        # 对预测结果进行解码，获得预测框
        # bs, 8400, 4
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # 对预测框与真实框进行分配
        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # 计算分类的损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # 计算bbox的损失
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        return loss.sum()  # loss(box, cls, dfl) # * batch_size


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)