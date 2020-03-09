import cv2
import os
import numpy as np
import json
from random import randint
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# from google.colab.patches import cv2_imshow

# Register to Detectron2 a custom dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

import random

classes = {
    'Car' : 0,
    'Van' : 1,
    'Truck' : 2,
    'Pedestrian' : 3,
    'Person_sitting' : 4,
    'Cyclist' : 5,
    'Tram' : 6,
    'Misc' : 7,
    'DontCare' : 8
}
model = "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml"
print(model)

def kitti_dataset(dataset):
    image_id = 0
    dataset_dicts = []
    for gt_file, image_file in dataset:
        height, width = cv2.imread(image_file).shape[:2]

        record = {}
        record['file_name'] = image_file
        record['height'] = height
        record['width'] = width
        record['image_id'] = image_id

        objs = []
        with open(gt_file) as fh:
            for line in fh:
                gt_file_content = line.strip()
                gt = gt_file_content.split(' ')

                category = gt[0]
                bbox_left = float(gt[4])
                bbox_top = float(gt[5])
                bbox_right = float(gt[6])
                bbox_bottom = float(gt[7])

                obj = {
                    "bbox" : [bbox_left, bbox_top, bbox_right, bbox_bottom],
                    "bbox_mode" : BoxMode.XYXY_ABS,
                    "category_id" : classes[category]
                }
                objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
        image_id += 1
    return dataset_dicts

from random import shuffle
from operator import itemgetter
import datetime
def get_datasets(training=0.8):
    images_location = '/home/grupo06/KITTI/data_object_image_2/training/image_2/'
    gt_location = '/home/grupo06/KITTI/training/label_2/'

    dataset = []
    for gt_file, image_file in zip(sorted(os.listdir(gt_location)), sorted(os.listdir(images_location))):
        dataset.append(('{}{}'.format(gt_location, gt_file), '{}{}'.format(images_location, image_file)))

    shuffle(dataset)
    training = dataset[:int(len(dataset)*0.8)]
    val = dataset[int(len(dataset)-len(training))+1:]

    return training, val

def get_testing_dataset():
    testing_location = '/home/grupo06/KITTI/data_object_image_2/testing/'

    test = []
    for f in os.listdir(testing_location):
        test.append('{}{}'.format(testing_location, f))
    return test

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
if __name__ == "__main__":
    training, val = get_datasets()

    dataset_type = 'training'
    DatasetCatalog.register("kitti_{}".format(dataset_type), lambda dataset_type = dataset_type : kitti_dataset(training))
    MetadataCatalog.get("kitti_{}".format(dataset_type)).set(thing_classes=list(classes.keys()))

    dataset_type = 'val'
    DatasetCatalog.register("kitti_{}".format(dataset_type), lambda dataset_type = dataset_type : kitti_dataset(val))
    MetadataCatalog.get("kitti_{}".format(dataset_type)).set(thing_classes=list(classes.keys()))

    kitti_metadata = MetadataCatalog.get("kitti_training")

    print('4')
    print('1000')
    print('512')
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("kitti_training",)
    cfg.DATASETS.TEST = ('kitti_val',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    i = 0
    examples = kitti_dataset(val)
    for image_file in random.sample(examples, 10):
        im = cv2.imread(image_file['file_name'])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=kitti_metadata,
                    scale=0.8
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        now = datetime.datetime.now()
        cv2.imwrite('example_{}_{}_{}.png'.format(now.hour, now.minute, now.second), v.get_image()[:, :, ::-1])
        i += 1

    evaluator = COCOEvaluator("kitti_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, 'kitti_val')
    inference_on_dataset(trainer.model, val_loader, evaluator)