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

def kitti_dataset(img_dir):
    dataset_location = '/home/mcv/datasets/KITTI/data_object_image_2/{}/image_2'.format(img_dir)
    gt_location = '/home/mcv/datasets/KITTI/training/label_2/'
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

    gt_files = []
    for gt_file in os.listdir(gt_location):
        gt_files.append('{}{}'.format(gt_location, gt_file))
    # https://detectron2.readthedocs.io/tutorials/datasets.html
    dataset_dicts = []
    image_id = 0
    for image in os.listdir(dataset_location):
        if 'mat' not in image:
            filename = '{}/{}'.format(dataset_location, image)
            height, width = cv2.imread(filename).shape[:2]
            
            record = {}
            record['filen_name'] = filename
            record['height'] = height
            record['width'] = width
            record['image_id'] = image_id
            record['iscrowd'] = 0
            
            gt_file = gt_files[image_id]
            gt_file_content = open(gt_file, 'r').readline().strip()
            gt = gt_file_content.split(' ')

            category = gt[0]
            bbox_left = gt[4]
            bbox_top = gt[5]
            bbox_right = gt[6]
            bbox_bottom = gt[7]
            
            annotations = [
                {
                    "bbox" : [bbox_left, bbox_top, bbox_right, bbox_bottom],
                    "bbox_mode" : BoxMode.XYXY_ABS,
                    "category_id" : classes[category]
                }
            ]
            record['annotations'] = annotations
            dataset_dicts.append(record)
    return dataset_dicts

# Register to Detectron2 a custom dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
for dataset_type in ['training']:
    DatasetCatalog.register("kitti_{}".format(dataset_type), lambda dataset_type = dataset_type : kitti_dataset(dataset_type))
    MetadataCatalog.get("kitti_{}".format(dataset_type)).set(
        thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 
                       'Cyclist', 'Tram', 'Misc', 'DontCare'])
model = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.DATASETS.TRAIN = ("kitti_training",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 

trainer.resume_or_load(resume=False)
trainer.train()