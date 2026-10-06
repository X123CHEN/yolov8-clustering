import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes, seed_everything
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# ---------------------------------------------------------------------------#
#   一次运行：加载同一份 yolo.py 权重，依次在 7 个域的 test_all.txt 上评估 mAP。
#   权重和模型规模仍在 yolo.py 的 model_path / phi 中配置。
# ---------------------------------------------------------------------------#
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'datasets', 'RDD2022')
TEST_LIST_REL = os.path.join('ImageSets', 'Main', 'test_all.txt')

# 切换实验时修改该目录，七个域的结果会分别保存到对应的子目录中。
MAP_OUT_BASE = './map_out/Style_soft/260712_test/exp5_soft_top2_conf_tau1.0_floor0.2_adv0.7'

USE_SEED = False
SEED = 66

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
    for sub_dir in ('ground-truth', 'detection-results', 'images-optional'):
        os.makedirs(os.path.join(map_out_path, sub_dir), exist_ok=True)


if __name__ == '__main__':
    # map_mode: 0=完整流程，1=仅预测，2=仅生成真实框，3=仅VOC mAP，4=COCO mAP。
    map_mode = 0
    classes_path = 'model_data/rdd_classes.txt'
    MINOVERLAP = 0.5
    confidence = 0.001
    nms_iou = 0.5
    score_threhold = 0.5
    map_vis = False

    if USE_SEED:
        seed_everything(SEED)
        print('Evaluation seed:', SEED)

    class_names, _ = get_classes(classes_path)

    # 七个域共用同一个模型，避免每个域重复加载权重。
    yolo = None
    if map_mode in (0, 1):
        print('Load model (once for all domains).')
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print('Load model done.')

    for country in COUNTRIES:
        vocdevkit_path = os.path.join(DATASET_ROOT, country)
        map_out_path = os.path.join(MAP_OUT_BASE, country)
        test_list_path = os.path.join(vocdevkit_path, TEST_LIST_REL)

        print('\n' + '=' * 60)
        print('Domain:', country)
        print('VOCdevkit_path:', vocdevkit_path)
        print('map_out_path:', map_out_path)
        print('=' * 60)

        if not os.path.isdir(vocdevkit_path):
            raise FileNotFoundError('Dataset not found: %s' % vocdevkit_path)
        if not os.path.isfile(test_list_path):
            raise FileNotFoundError('Test list not found: %s' % test_list_path)

        with open(test_list_path, encoding='utf-8') as test_file:
            image_ids = test_file.read().strip().split()
        if not image_ids:
            raise ValueError('Empty test list: %s' % test_list_path)

        print('Test images:', len(image_ids))
        ensure_map_dirs(map_out_path)

        if map_mode in (0, 1):
            print('Get predict result.')
            for image_id in tqdm(image_ids, desc=country + ' predict'):
                image_path = os.path.join(vocdevkit_path, 'JPEGImages', image_id + '.jpg')
                with Image.open(image_path) as image:
                    if map_vis:
                        image.save(os.path.join(map_out_path, 'images-optional', image_id + '.jpg'))
                    yolo.get_map_txt(image_id, image, class_names, map_out_path)
            print('Get predict result done.')

        if map_mode in (0, 2):
            print('Get ground truth result.')
            for image_id in tqdm(image_ids, desc=country + ' gt'):
                ground_truth_path = os.path.join(
                    map_out_path, 'ground-truth', image_id + '.txt')
                annotation_path = os.path.join(
                    vocdevkit_path, 'Annotations', image_id + '.xml')

                with open(ground_truth_path, 'w', encoding='utf-8') as new_f:
                    root = ET.parse(annotation_path).getroot()
                    for obj in root.findall('object'):
                        difficult = obj.find('difficult')
                        difficult_flag = difficult is not None and int(difficult.text) == 1
                        obj_name = obj.find('name').text
                        if obj_name not in class_names:
                            continue

                        bndbox = obj.find('bndbox')
                        left = bndbox.find('xmin').text
                        top = bndbox.find('ymin').text
                        right = bndbox.find('xmax').text
                        bottom = bndbox.find('ymax').text
                        difficult_suffix = ' difficult' if difficult_flag else ''
                        new_f.write('%s %s %s %s %s%s\n' % (
                            obj_name, left, top, right, bottom, difficult_suffix))
            print('Get ground truth result done.')

        if map_mode in (0, 3):
            print('Get map.')
            get_map(MINOVERLAP, True, score_threhold=score_threhold,
                    path=map_out_path)
            print('Get map done.')

        if map_mode == 4:
            print('Get map (COCO).')
            get_coco_map(class_names=class_names, path=map_out_path)
            print('Get map done.')

    print('\nAll %d domains finished. Results under: %s' % (
        len(COUNTRIES), MAP_OUT_BASE))
