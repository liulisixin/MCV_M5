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

def kitti_dataset(img_dir):
    #dataset_location = '../../KITTI/data_object_image_2/{}/image_2'.format(img_dir)
    dataset_location = '../../KITTI/data_object_image_2/{}'.format(img_dir)
    gt_location = '../../KITTI/training/label_2/'
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
    
    """
    gt_files = []
    for gt_file in os.listdir(gt_location):
        gt_files.append('{}{}'.format(gt_location, gt_file))
    """
    # https://detectron2.readthedocs.io/tutorials/datasets.html
    dataset_dicts = []

    for image_id, image in enumerate(os.listdir(dataset_location)):
        if 'mat' not in image:
            filename = '{}/{}'.format(dataset_location, image)
            height, width = cv2.imread(filename).shape[:2]
            
            record = {}
            record['file_name'] = filename
            record['height'] = height
            record['width'] = width
            record['image_id'] = image_id
            
            """
            gt_file = gt_files[image_id]
            objs = []
            with open(gt_file) as fh:
                for line in fh:
                    gt_file_content =line.strip()
                    #gt_file_content = open(gt_file, 'r').readline().strip()
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
            """
            dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    for dataset_type in ['testing/image_2']:
        DatasetCatalog.register("kitti_{}".format(dataset_type), lambda dataset_type = dataset_type : kitti_dataset(dataset_type))
        MetadataCatalog.get("kitti_{}".format(dataset_type)).set(
            thing_classes=['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
                           'Cyclist', 'Tram', 'Misc', 'DontCare'])
    kitti_metadata = MetadataCatalog.get("kitti_testing/image_2")

    model = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ("kitti_training",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("kitti_val",)
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    dataset_dicts = kitti_dataset("testing/image_2")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=kitti_metadata,
                       scale=0.8,
                       # instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('evaluation{}.png'.format(d["file_name"].split("/")[-1]), v.get_image()[:, :, ::-1])

        