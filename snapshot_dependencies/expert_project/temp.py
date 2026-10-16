# # from utils.dataloader import YoloDataset
# # from utils.utils import get_classes


# # train_annotation_path = './20231124_chongzu_data_train.txt'
# # classes_path = 'model_data/rdd_classes.txt'
# # input_shape = [640, 640]
# # class_names, num_classes = get_classes(classes_path)
# # with open(train_annotation_path, encoding='utf-8') as f:
# #         train_lines = f.readlines()
# # num_train = len(train_lines)

# # train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=50, \
# #                                     mosaic=False, mixup=True, mosaic_prob=0.5, mixup_prob=0.5,
# #                                     train=True, special_aug_ratio=0.5)

# # for step, (annotation_line, image0, labels_out0, image1, labels_out1, image2, labels_out2) in enumerate(train_dataset):
# #     # print(f"{step} >>>")
# #     print(labels_out0.shape, labels_out1.shape, labels_out2.shape)
# #     # print(annotation_line.strip().split("&")[0], labels_out0.shape)
# #     # print(annotation_line.strip().split("&")[1], labels_out1.shape)
# #     # print(annotation_line.strip().split("&")[2], labels_out2.shape)
# #     # # print()
# #     # if labels_out0.shape[0] == 0:
# #     #       print(annotation_line)
# #     #       print(labels_out0.shape)
# #     #       break

# # from core.loss import TripletLoss
# # import torch

# # x = torch.randn(5, 10)
# # y = torch.Tensor([1, 2, 3, 4, 1])

# # loss = TripletLoss()
# # l_v = loss(x, y)
# # print(l_v)


# # import os

# # # 设置目录路径
# # label_dir = "/cvgroup/home/wurx/Datasets/KITTI/training/label_2"
# # image_dir = "/cvgroup/home/wurx/Datasets/KITTI/data_object_image_2/training/image_2"

# # # 获取label目录下的所有.txt文件名
# # label_files = os.listdir(label_dir)

# # # 获取image目录下的所有.png文件名
# # image_files = os.listdir(image_dir)

# # # 将文件名去掉扩展名后放入集合中，方便比较
# # label_files_without_ext = {os.path.splitext(file)[0] for file in label_files}
# # image_files_without_ext = {os.path.splitext(file)[0] for file in image_files}

# # # 找到label中存在但image中不存在的文件
# # extra_label_files = label_files_without_ext - image_files_without_ext

# # # 将结果写入txt文件
# # output_file = "extra_files.txt"
# # with open(output_file, "w") as f:
# #     if extra_label_files:
# #         f.write("以下是label目录中存在但image目录中不存在的.txt文件名：\n")
# #         for file in extra_label_files:
# #             f.write(file + ".txt\n")
# #     else:
# #         f.write("label目录中的所有.txt文件都存在于image目录中。\n")

# # print(f"结果已写入到文件：{output_file}")


# ##BDD10K数据标签统计
# # import json
# # def count_categories_from_json(file_path):
# #     categories = set()
# #     with open(file_path, 'r', encoding='utf-8') as file:
# #         data = json.load(file)
# #         for entry in data:
# #             if 'labels' in entry:
# #                 for label in entry['labels']:
# #                     category = label['category']
# #                     categories.add(category)
# #     return categories

# # def print_categories(categories):
# #     print("所有种类别:")
# #     for category in categories:
# #         print(category)

# # if __name__ == "__main__":
# #     file_path = '/cvgroup/home/wurx/Datasets/BDD2/labels/det_20/det_val.json'
# #     categories = count_categories_from_json(file_path)
# #     print_categories(categories)

# ##Sim10k
# # import os
# # import xml.etree.ElementTree as ET

# # def count_categories_from_xml_dir(directory):
# #     categories = set()
# #     for filename in os.listdir(directory):
# #         if filename.endswith('.xml'):
# #             filepath = os.path.join(directory, filename)
# #             categories.update(get_categories_from_xml(filepath))
# #     return categories

# # def get_categories_from_xml(xml_file):
# #     categories = set()
# #     tree = ET.parse(xml_file)
# #     root = tree.getroot()
# #     for obj in root.findall('object'):
# #         category = obj.find('name').text
# #         categories.add(category)
# #     return categories

# # def print_categories(categories):
# #     print("所有种类别:")
# #     for category in categories:
# #         print(category)

# # if __name__ == "__main__":
# #     directory_path = '/cvgroup/home/wurx/Datasets/Sim10k/VOC2012/Annotations'
# #     categories = count_categories_from_xml_dir(directory_path)
# #     print_categories(categories)

# ##KITTI 数据标签统计
# # import os

# # def count_categories_from_txt_dir(directory):
# #     categories = set()
# #     for filename in os.listdir(directory):
# #         if filename.endswith('.txt'):
# #             filepath = os.path.join(directory, filename)
# #             categories.update(get_categories_from_txt(filepath))
# #     return categories

# # def get_categories_from_txt(txt_file):
# #     categories = set()
# #     with open(txt_file, 'r') as file:
# #         for line in file:
# #             category = line.strip().split(' ')[0]
# #             categories.add(category)
# #     return categories

# # def print_categories(categories):
# #     print("所有种类别:")
# #     for category in categories:
# #         print(category)

# # if __name__ == "__main__":
# #     directory_path = '/cvgroup/home/wurx/Datasets/KITTI/training/label_2'
# #     categories = count_categories_from_txt_dir(directory_path)
# #     print_categories(categories)



    
# # import os
# # import glob
# # import xml.etree.ElementTree as ET
# # import tqdm
 
# # def get_classes(classes_path):
# #     with open(classes_path, encoding='utf-8') as f:
# #         class_names = f.readlines()
# #     class_names = [c.strip() for c in class_names]
# #     return class_names, len(class_names)
 
 
# # def convert(size, box):
# #     dw = 1.0 / size[0]
# #     dh = 1.0 / size[1]
# #     x = (box[0] + box[1]) / 2.0
# #     y = (box[2] + box[3]) / 2.0
# #     w = box[1] - box[0]
# #     h = box[3] - box[2]
# #     x = x * dw
# #     w = w * dw
# #     y = y * dh
# #     h = h * dh
# #     return (x, y, w, h)
 
 
# # if __name__ == '__main__':
# #     # 设置xml文件的路径和要保存的txt文件路径
# #     xml_root_path = r'/cvgroup/home/wurx/Datasets/Sim10k/VOC2012/Annotations'
# #     txt_save_path = r'/cvgroup/home/wurx/Datasets/Sim10k/VOC2012/labels'
# #     if not os.path.exists(txt_save_path):
# #         os.makedirs(txt_save_path)
# #     xml_paths = glob.glob(os.path.join(xml_root_path, '*.xml'))
# #     # classes_path = 'labels.txt'
# #     classes_path = 'classes.txt'
# #     classes, _      = get_classes(classes_path)
 
# #     for xml_id in xml_paths:
# #         txt_id = os.path.join(txt_save_path, (xml_id.split('\\')[-1])[:-4] + '.txt')
# #         txt = open(txt_id, 'w')
# #         xml = open(xml_id, encoding='utf-8')
# #         tree = ET.parse(xml)
# #         root = tree.getroot()
# #         size = root.find('size')
# #         w = int(size.find('width').text)
# #         h = int(size.find('height').text)
# #         for obj in root.iter('object'):
# #             difficult = 0
# #             if obj.find('difficult') != None:
# #                 difficult = obj.find('difficult').text
# #             cls = obj.find('name').text
# #             if cls not in classes or int(difficult) == 1:
# #                 continue
# #             cls_id = classes.index(cls)
# #             xmlbox = obj.find('bndbox')
# #             b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('xmax').text)),
# #                  int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('ymax').text)))
# #             box = convert((w, h), b)
# #             txt.write(str(cls_id) + ' ' + ' '.join([str(a) for a in box]) + '\n')
# #         txt.close()

# import os

# def process_file_cityscapes(file_path):
#     # 定义一个字典来映射旧值到新值
#     mapping = {
#         '3': '0',
#         '11': '2',
#         '12': '1',
#         '14': '2',
#         '19': '0',
#         '21': '2',
#         '22': '4',
#         '24': '1',
#         '25': '3',
#         '30': '3',
#         '31': '3',
#         '35': '4',
#         '36': '2',
#         '37': '3'
#     }
    
#     # 读取文件
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # 处理每一行
#     processed_lines = []
#     for line in lines:
#         # 按空格分割每一行
#         parts = line.strip().split(' ')
#         # 如果第一个元素在映射中，替换为新值
#         if parts[0] in mapping:
#             parts[0] = mapping[parts[0]]
#             processed_lines.append(' '.join(parts))
    
#     # 将处理后的数据写回文件
#     with open(file_path, 'w') as file:
#         file.write('\n'.join(processed_lines))


# def process_file_bdd100k(file_path):
#     # 定义一个字典来映射旧值到新值
#     mapping = {
#         '0': '0',
#         '1': '1',
#         '2': '2',
#         '3': '4',
#         '4': '3',
#         '5': '3',
#         '6': '1',
#     }
    
#     # 读取文件
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # 处理每一行
#     processed_lines = []
#     for line in lines:
#         # 按空格分割每一行
#         parts = line.strip().split(' ')
#         # 如果第一个元素在映射中，替换为新值
#         if parts[0] in mapping:
#             parts[0] = mapping[parts[0]]
#             processed_lines.append(' '.join(parts))
    
#     # 将处理后的数据写回文件
#     with open(file_path, 'w') as file:
#         file.write('\n'.join(processed_lines))

# def process_file_kitti(file_path):
#     # 定义一个字典来映射旧值到新值
#     mapping = {
#         '0': '1',
#         '1': '0',
#         '2': '2',
#         '3': '3',
#         '4': '1',
#         '7': '3',
#     }
    
#     # 读取文件
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # 处理每一行
#     processed_lines = []
#     for line in lines:
#         # 按空格分割每一行
#         parts = line.strip().split(' ')
#         # 如果第一个元素在映射中，替换为新值
#         if parts[0] in mapping:
#             parts[0] = mapping[parts[0]]
#             processed_lines.append(' '.join(parts))
    
#     # 将处理后的数据写回文件
#     with open(file_path, 'w') as file:
#         file.write('\n'.join(processed_lines))

# def process_file_sim10k(file_path):
#     # 定义一个字典来映射旧值到新值
#     mapping = {
#         '0': '0',
#         '1': '1',
#         '2': '4',
#     }
    
#     # 读取文件
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     # 处理每一行
#     processed_lines = []
#     for line in lines:
#         # 按空格分割每一行
#         parts = line.strip().split(' ')
#         # 如果第一个元素在映射中，替换为新值
#         if parts[0] in mapping:
#             parts[0] = mapping[parts[0]]
#             processed_lines.append(' '.join(parts))
    
#     # 将处理后的数据写回文件
#     with open(file_path, 'w') as file:
#         file.write('\n'.join(processed_lines))


# def process_folder(folder_path):
#     # 遍历文件夹下的每一个txt文件
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith('.txt'):
#                 file_path = os.path.join(root, file)
#                 process_file_sim10k(file_path)

# # 调用函数来处理文件夹下的每一个txt文件
# folder_path = "/cvgroup/home/wurx/Datasets/Sim10k/VOC2012/labels"
# process_folder(folder_path)




# import os
# # 设置要重命名的文件夹路径
# folder_path = "/cvgroup/home/wurx/Datasets/cityscapes/JPEGImages_beta_0.02"
# # 获取文件夹中所有的文件名
# file_names = os.listdir(folder_path)
# # 遍历文件夹中的每个文件
# for file_name in file_names:
#     # 检查文件是否以'.png'结尾
#     if file_name.endswith('.png'):
#         # 去除后缀'.png'
#         new_name = file_name[:-4]
#         # 按照'_'分隔文件名
#         parts = new_name.split('_')
#         # 删除倒数第三个'_'后的内容
#         if len(parts) >= 3:
#             new_name = '_'.join(parts[:-3]) + '.png'
#         # 构造新的文件路径
#         new_path = os.path.join(folder_path, new_name)
#         # 重命名文件
#         os.rename(os.path.join(folder_path, file_name), new_path)
#         print(f"重命名文件: {file_name} -> {new_name}")


# import os
# import shutil

# # 源文件夹路径
# source_folder = "/cvgroup/home/wurx/Datasets/cityscapes/JPEGImages"

# # 目标文件夹路径字典，键为目标文件夹的关键字，值为目标文件夹的路径
# target_folders = {
#     "beta_0.005": "/cvgroup/home/wurx/Datasets/cityscapes/JPEGImages_beta_0.005",
#     "beta_0.01": "/cvgroup/home/wurx/Datasets/cityscapes/JPEGImages_beta_0.01",
#     "beta_0.02": "/cvgroup/home/wurx/Datasets/cityscapes/JPEGImages_beta_0.02"
# }

# # 遍历源文件夹中的文件
# for file_name in os.listdir(source_folder):
#     # 遍历目标文件夹路径字典中的键值对
#     for target_key, target_folder in target_folders.items():
#         # 检查文件名中是否包含目标关键字
#         if target_key in file_name:
#             # 构造目标文件的完整路径
#             target_path = os.path.join(target_folder, file_name)
#             # 移动文件
#             shutil.move(os.path.join(source_folder, file_name), target_path)
#             print(f"移动文件: {file_name} -> {target_path}")
#             # 移动到目标文件夹后跳出内循环
#             break



import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# 指定XML文件夹路径
xml_folder = "/cvgroup/home/wurx/Datasets/Sim10k/VOC2012/Annotations"

# 初始化标签字典
label_counts = defaultdict(int)

# 遍历XML文件夹中的每个文件
for filename in os.listdir(xml_folder):
    if filename.endswith(".xml"):
        # 解析XML文件
        tree = ET.parse(os.path.join(xml_folder, filename))
        root = tree.getroot()
        
        # 遍历XML文件中的所有标签
        for obj in root.findall('object'):
            # 获取标签名称并增加计数
            label_name = obj.find('name').text
            label_counts[label_name] += 1

# 输出标签及其数量
print("Label Counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")


# import os
# import xml.etree.ElementTree as ET

# # 指定XML文件夹路径
# xml_folder = "/cvgroup/home/wurx/Datasets/Sim10k/VOC2012/Annotations"

# # 初始化已修改文件列表
# modified_files = []

# # 遍历XML文件夹中的每个文件
# for filename in os.listdir(xml_folder):
#     if filename.endswith(".xml"):
#         # 解析XML文件
#         tree = ET.parse(os.path.join(xml_folder, filename))
#         root = tree.getroot()
        
#         # 遍历XML文件中的所有标签
#         for obj in root.findall('object'):
#             # 查找名为'pedestrian'的标签
#             name_element = obj.find('name')
#             if name_element is not None and name_element.text == 'pedestrian':
#                 print(filename)
#                 # 将标签内容修改为'person'
#                 name_element.text = 'person'

#         # 保存修改后的XML文件
#         tree.write(os.path.join(xml_folder, filename))

#         # 添加已修改文件到列表中
#         modified_files.append(filename)

# 输出已修改文件的名称
# print("已修改的文件:")
# for file in modified_files:
#     print(file)
