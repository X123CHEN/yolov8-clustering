from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, \
                 mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7,
                 style_cache=None):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio
        self.style_cache = style_cache
        if self.style_cache is not None and (mosaic or mixup):
            raise ValueError(
                "Offline style routing requires mosaic=False and mixup=False because "
                "synthetic images have no cached clustering distance."
            )

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

        self.bbox_attrs = 5 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:

            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)

            image, box, image1, box1, image2, box2 = self.get_random_data_with_Mosaic(lines, self.input_shape)
            # image, box= self.get_random_data_with_Mosaic(lines, self.input_shape)
            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_3, box_3, image_4, box_4, image_5, box_5 = self.get_random_data(lines[0], self.input_shape,random=self.train)
                image,  box  = self.get_random_data_with_MixUp(image, box, image_3, box_3)
                image1, box1 = self.get_random_data_with_MixUp(image1, box1, image_4, box_4)
                image2, box2 = self.get_random_data_with_MixUp(image2, box2, image_5, box_5)
        else:
            # image_data_list, box_list = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
            # image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
            image, box, image1, box1, image2, box2 = self.get_random_data(self.annotation_lines[index],
                                                                          self.input_shape, random=self.train)
            # image_trip, box_trip, image1_trip, box1_trip, image2_trip, box2_trip = self.get_random_data(self.annotation_lines[index],
            #                                                               self.input_shape, random=self.train)
        image0, labels_out0 = self.gt_preprocess(image=image, box=box)  # 日本
        image1, labels_out1 = self.gt_preprocess(image=image1, box=box1)  # 印度
        image2, labels_out2 = self.gt_preprocess(image=image2, box=box2)  # 美国

        # image0_trip, labels_out0_trip = self.gt_preprocess(image=image_trip, box=box_trip)  # 日本
        # image1_trip, labels_out1_trip = self.gt_preprocess(image=image1_trip, box=box1_trip)  # 印度
        # image2_trip, labels_out2_trip = self.gt_preprocess(image=image2_trip, box=box2_trip)  # 美国

        if self.style_cache is None:
            return image0, labels_out0, image1, labels_out1, image2, labels_out2,

        image_paths = [part.split()[0] for part in self.annotation_lines[index].split('&')]
        if len(image_paths) != 3:
            raise RuntimeError("Style training annotation must contain exactly three '&' parts")
        cached = [self.style_cache.lookup(path) for path in image_paths]
        hard_styles = np.asarray([item[0] for item in cached], dtype=np.int64)
        distance_sq = np.stack([item[1] for item in cached]).astype(np.float32)
        return (image0, labels_out0, image1, labels_out1, image2, labels_out2,
                image_paths, hard_styles, distance_sq)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def gt_preprocess(self, image, box):
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        # ---------------------------------------------------#
        #   对真实框进行预处理
        # ---------------------------------------------------#
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            # ---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            # ---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # ---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            # ---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            # ---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            # ---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]
        return image, labels_out

    def img_letterbox(self, line, input_shape, jitter=.1, hue=.1, sat=0.7, val=0.4, random=True):
        # ------------------------------  #
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            return image_data, box
        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.8, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, box

    def get_random_data(self, annotation_line, input_shape, jitter=.1, hue=.1, sat=0.7, val=0.4, random=True):
        # annotation_line = "/disk/home/wurx/yolov8_beifen/data_total/JPEGImages/Japan_000003.jpg 151,427,581,600,2 2,493,53,581,2&/disk/home/wurx/yolov8_beifen/data_total/JPEGImages/India_000014.jpg 304,529,489,691,2 1,526,226,708,2&/disk/home/wurx/yolov8_beifen/data_total/JPEGImages/United_States_000002.jpg 116,433,307,455,1 173,398,198,426,0 299,398,325,485,0 258,473,493,498,1 320,493,377,632,0\n"
        three_part = annotation_line.split('&')

        line = three_part[0].split()
        line1 = three_part[1].split()
        line2 = three_part[2].split()

        image_data,  box  = self.img_letterbox(line,  input_shape, jitter, hue, sat, val, random)
        image_data1, box1 = self.img_letterbox(line1, input_shape, jitter, hue, sat, val, random)
        image_data2, box2 = self.img_letterbox(line2, input_shape, jitter, hue, sat, val, random)

        return image_data, box, image_data1, box1, image_data2, box2  ###0是Japan ， 1是India，  2是United_States

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def Mosaic_3domain(self, lines, input_shape, jitter, hue, sat, val):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in lines:
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            # line_content = line.split()
            line_content = line
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            # ---------------------------------#
            #   图片的大小
            # ---------------------------------#
            iw, ih = image.size
            # ---------------------------------#
            #   保存框的位置
            # ---------------------------------#
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])
            # ---------------------------------#
            #   是否翻转图片
            # ---------------------------------#
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.8, 1.1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            # -----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            # -----------------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1 if index < 3 else 0
            box_data = []
            # ---------------------------------#
            #   对box进行重新处理
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)
        # ---------------------------------#
        #   将图片分割，放在一起
        # ---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        # print(image_datas)
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对框进行进一步的处理
        # ---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        return new_image, new_boxes

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):

        lines_list = []
        for lines in annotation_line:
            line = lines.split('&')
            for i in line:
                n = i.split()
                lines_list.append(n)

        new_image_list = []
        new_boxes_list = []

        index = 0  # 初始化 index
        while index <= len(lines_list):
            lines_4 = lines_list[index:index + 4]
            new_image, new_boxes = self.Mosaic_3domain(lines_4, input_shape, jitter, hue, sat, val)
            new_boxes_list.extend([new_boxes])
            new_image_list.extend([new_image])
            index += 4
            if index >= len(lines_list):
                break
        # return new_image, new_boxes
        return new_image_list[0], new_boxes_list[0], new_image_list[1], new_boxes_list[1], new_image_list[2], new_boxes_list[2]

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


# DataLoader中collate_fn使用
# def yolo_dataset_collate(batch):
#     images = []
#     bboxes = []
#     for i, (img, box) in enumerate(batch):
#         images.append(img)
#         box[:, 0] = i
#         bboxes.append(box)
#
#     images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
#     return images, bboxes

# # DataLoader中collate_fn使用
# def yolo_dataset_collate(batch):
#     images      = []
#     n_max_boxes = 0
#     bs          = len(batch)
#     for i, (img, box) in enumerate(batch):
#         images.append(img)
#         n_max_boxes = max(n_max_boxes, len(box))

#     bboxes  = torch.zeros((bs, n_max_boxes, 4))
#     labels  = torch.zeros((bs, n_max_boxes, 1))
#     masks   = torch.zeros((bs, n_max_boxes, 1))

#     for i, (img, box) in enumerate(batch):
#         _sub_length = len(box)
#         bboxes[i, :_sub_length] = box[:, :4]
#         labels[i, :_sub_length] = box[:, 4]
#         masks[i, :_sub_length]  = 1

#     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
#     return images, bboxes, labels, masks


# wrq添加的代码------------------------------------------------------------------------------------------------------------------
import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def OriImg_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    RGBimg = RGBimg.resize((720, 720))
    return RGBimg


class DatasetLoader(Dataset):
    def __init__(self, name, getreal, transform=None, oriimg_loader=OriImg_loader, root='E:/Project/MADDoG-master/datasets/'):
    # def __init__(self, name, getreal, transform=None, oriimg_loader=OriImg_loader, depthimg_loader=DepthImg_loader, root='E:/Project/MADDoG-master/datasets/'):

        self.name = name
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, self.name)
        if getreal:
            filename = 'image_list_real.txt'
        else:
            filename = 'image_list_fake.txt'

        fh = open(os.path.join(self.root, filename), 'r')

        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            label = [float(item) for item in words[1].split(',')][0]  # x是种类
            xywh_str = str([float(item) for item in words[1].split(',')][1:]).replace('[', '').replace(']', '')
            qiyu_labels = words[2:]

            # imgs.append((words[0], int(words[1]), words[2:]))
            imgs.append((words[0], int(label), xywh_str, qiyu_labels))  # imgs的 第一个是路径， 第二个是类别，第三个是类别所对应的坐标

        self.imgs = imgs
        self.transform = transform
        self.oriimg_loader = oriimg_loader

    def __getitem__(self, index):

        ori_img_dir, label, x_y_w_h, qiyu_bbox = self.imgs[index]
        ori_img_dir_all = os.path.join(self.root, ori_img_dir)
        # depth_img_dir_all = os.path.join(self.root)

        ori_rgbimg = self.oriimg_loader(ori_img_dir_all)

        if self.transform is not None:
            ori_rgbimg = self.transform(ori_rgbimg)
            # ori_hsvimg = self.transform(ori_hsvimg)
            # depth_img = self.transform(depth_img)

            ori_catimg = torch.cat([ori_rgbimg], 0)
            # ori_catimg = torch.cat([ori_rgbimg,ori_hsvimg],0)

        return ori_catimg, label, x_y_w_h, qiyu_bbox

    def __len__(self):
        return len(self.imgs)


def get_dataset_loader(name, getreal, batch_size):
    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                   mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])])

    pre_process = transforms.Compose([transforms.ToTensor()])

    # dataset and data loader
    dataset = DatasetLoader(name=name,
                            getreal=getreal,
                            transform=pre_process,
                            )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # wrq加了 collate_fn=custom_collate_fn,
        # collate_fn=yolo_dataset_collate,
        shuffle=True,
        drop_last=True)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     #wrq修改过，适配yolo数据加载时候的格式   i（定位图片第几张），labels（标签），x y w h
    #     shuffle=True,
    #     drop_last=True)

    return data_loader


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    images1 = []
    bboxes1 = []
    images2 = []
    bboxes2 = []
    image_paths = []
    hard_styles = []
    distance_sq = []

    num = 0
    for i, sample in enumerate(batch):
        img, box, img1, box1, img2, box2 = sample[:6]
            # , img_trip, box_trip, img1_trip, box1_trip, img2_trip, box2_trip) in enumerate(batch):
        images.append(img)
        box[:, 0] = num
        bboxes.append(box)

        images1.append(img1)
        box1[:, 0] = num + 1
        bboxes1.append(box1)

        images2.append(img2)
        box2[:, 0] = num + 2
        bboxes2.append(box2)

        if len(sample) == 9:
            image_paths.extend(sample[6])
            hard_styles.append(sample[7])
            distance_sq.append(sample[8])

        # images_trip.append(img_trip)
        # box_trip[:, 0] = num
        # bboxes_trip.append(box_trip)
        #
        # images1_trip.append(img1_trip)
        # box1_trip[:, 0] = num + 1
        # bboxes1_trip.append(box1_trip)
        #
        # images2_trip.append(img2_trip)
        # box2_trip[:, 0] = num + 2
        # bboxes2_trip.append(box2_trip)

        num += 3

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)

    images1 = torch.from_numpy(np.array(images1)).type(torch.FloatTensor)
    bboxes1 = torch.from_numpy(np.concatenate(bboxes1, 0)).type(torch.FloatTensor)

    images2 = torch.from_numpy(np.array(images2)).type(torch.FloatTensor)
    bboxes2 = torch.from_numpy(np.concatenate(bboxes2, 0)).type(torch.FloatTensor)

    if hard_styles:
        # Flatten in the same [sample0:S0,S1,S2, sample1:S0,S1,S2, ...]
        # order used by the Student image concatenation in utils_fit.py.
        return (images, bboxes, images1, bboxes1, images2, bboxes2,
                image_paths,
                torch.from_numpy(np.stack(hard_styles).reshape(-1)).long(),
                torch.from_numpy(np.stack(distance_sq).reshape(-1, 3)).float())
    return images, bboxes, images1, bboxes1, images2, bboxes2


# def yolo_dataset_collate(batch):
#     images = []
#     bboxes = []
#     for i, (img, box, img1,box1, img2,box2) in enumerate(batch):
#         images.append(img)
#         box[:, 0] = i
#         bboxes.append(box)
#
#     images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
#     return images, bboxes


###target dataset
def default_loader(path):
    RGBimg = Image.open(path).convert('RGB')
    # HSVimg = Image.open(path).convert('HSV')
    # return RGBimg, HSVimg
    return RGBimg


class DatasetLoader2(Dataset):
    def __init__(self, name, transform=None, loader=default_loader,
                 root='E:/Project/MADDoG-master/datasets/'):  # root地址需改动

        self.name = name
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, self.name)
        filename = 'image_list_all.txt'

        fh = open(os.path.join(self.root, filename), 'r')

        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn = os.path.join(self.root, fn)
        # rgbimg, hsvimg = self.loader(fn)
        rgbimg, _ = self.loader(fn)
        if self.transform is not None:
            rgbimg = self.transform(rgbimg)
            # hsvimg = self.transform(hsvimg)

            # catimg = torch.cat([rgbimg,hsvimg],0)
            catimg = torch.cat([rgbimg], 0)

        return catimg, label

    def __len__(self):
        return len(self.imgs)


def get_tgtdataset_loader(name, batch_size):
    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                   mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])])

    pre_process = transforms.Compose([transforms.ToTensor()])

    # dataset and data loader
    dataset = DatasetLoader2(name=name,
                             transform=pre_process,
                             )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn = yolo_dataset_collate
    )

    return data_loader


########以下是 eval时候的数据加载器
class YoloDataset_eval(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, \
                 mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7):
        super(YoloDataset_eval, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

        self.bbox_attrs = 5 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        # ---------------------------------------------------#
        #   对真实框进行预处理
        # ---------------------------------------------------#
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            # ---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            # ---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # ---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            # ---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            # ---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            # ---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.1, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.8, 1.2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            line_content = line.split()
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # ---------------------------------#
            #   图片的大小
            # ---------------------------------#
            iw, ih = image.size
            # ---------------------------------#
            #   保存框的位置
            # ---------------------------------#
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # ---------------------------------#
            #   是否翻转图片
            # ---------------------------------#
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # -----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            # -----------------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # ---------------------------------#
            #   对box进行重新处理
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # ---------------------------------#
        #   将图片分割，放在一起
        # ---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对框进行进一步的处理
        # ---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate_eval(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes
