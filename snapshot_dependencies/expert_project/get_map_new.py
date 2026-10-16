import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# ---------------------------------------------------------------------------#
#   一次运行：同一套 yolo.py 权重，依次在 7 个域的 test.txt 上评估 mAP
#   权重在 yolo.py 的 model_path / phi 中配置；切换消融实验时改 MAP_OUT_BASE
# ---------------------------------------------------------------------------#
DATASET_ROOT = '/data4/czh/code/yolov8-pytorch-master-main/datasets/RDD2022'
TEST_LIST_REL = os.path.join('ImageSets', 'Main', 'test.txt')
MAP_OUT_BASE = './map_out/best/run1/train2'

COUNTRIES = [
    'Japan',
    'India',
    'United_States',
    'Norway',
    'new_Czech',
    'new_China_Drone',
    'new_China_MotorBike',
]


def ensure_map_dirs(map_out_path):
    os.makedirs(map_out_path, exist_ok=True)
    for sub in ('ground-truth', 'detection-results', 'images-optional'):
        os.makedirs(os.path.join(map_out_path, sub), exist_ok=True)


if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    map_mode = 0
    classes_path = 'model_data/rdd_classes.txt'
    MINOVERLAP = 0.5
    confidence = 0.001
    nms_iou = 0.5
    score_threhold = 0.5
    map_vis = False

    class_names, _ = get_classes(classes_path)

    yolo = None
    if map_mode == 0 or map_mode == 1:
        print("Load model (once for all domains).")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

    for country in COUNTRIES:
        VOCdevkit_path = os.path.join(DATASET_ROOT, country)
        map_out_path = os.path.join(MAP_OUT_BASE, country)
        test_list_path = os.path.join(VOCdevkit_path, TEST_LIST_REL)

        print('\n' + '=' * 60)
        print('Domain:', country)
        print('VOCdevkit_path:', VOCdevkit_path)
        print('map_out_path:', map_out_path)
        print('=' * 60)

        if not os.path.isdir(VOCdevkit_path):
            raise FileNotFoundError('Dataset not found: %s' % VOCdevkit_path)
        if not os.path.isfile(test_list_path):
            raise FileNotFoundError('test list not found: %s' % test_list_path)

        image_ids = open(test_list_path, encoding='utf-8').read().strip().split()
        if not image_ids:
            raise ValueError('Empty test list: %s' % test_list_path)

        ensure_map_dirs(map_out_path)

        if map_mode == 0 or map_mode == 1:
            print("Get predict result.")
            for image_id in tqdm(image_ids, desc=country + ' predict'):
                image_path = os.path.join(VOCdevkit_path, 'JPEGImages', image_id + '.jpg')
                image = Image.open(image_path)
                if map_vis:
                    image.save(os.path.join(map_out_path, 'images-optional', image_id + '.jpg'))
                yolo.get_map_txt(image_id, image, class_names, map_out_path)
            print("Get predict result done.")

        if map_mode == 0 or map_mode == 2:
            print("Get ground truth result.")
            for image_id in tqdm(image_ids, desc=country + ' gt'):
                with open(os.path.join(map_out_path, 'ground-truth', image_id + '.txt'), 'w', encoding='utf-8') as new_f:
                    ann_path = os.path.join(VOCdevkit_path, 'Annotations', image_id + '.xml')
                    root = ET.parse(ann_path).getroot()
                    for obj in root.findall('object'):
                        difficult_flag = False
                        if obj.find('difficult') is not None:
                            difficult = obj.find('difficult').text
                            if int(difficult) == 1:
                                difficult_flag = True
                        obj_name = obj.find('name').text
                        if obj_name not in class_names:
                            continue
                        bndbox = obj.find('bndbox')
                        left = bndbox.find('xmin').text
                        top = bndbox.find('ymin').text
                        right = bndbox.find('xmax').text
                        bottom = bndbox.find('ymax').text

                        if difficult_flag:
                            new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                        else:
                            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            print("Get ground truth result done.")

        if map_mode == 0 or map_mode == 3:
            print("Get map.")
            get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
            print("Get map done.")

        if map_mode == 4:
            print("Get map (COCO).")
            get_coco_map(class_names=class_names, path=map_out_path)
            print("Get map done.")

    print('\nAll %d domains finished. Results under: %s' % (len(COUNTRIES), MAP_OUT_BASE))
