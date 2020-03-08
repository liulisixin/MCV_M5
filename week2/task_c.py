import cv2
import os
import numpy as np
import json
import datetime
from random import randint
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

prefix = 'retina_r_101_fpn_x3_th_50'

def write_images_into_disk(images):
    for idx, image in enumerate(images):
        now = datetime.datetime.now()
        cv2.imwrite('output/task_c_sample_{}_{}_{}_{}_{}.png'.format(prefix, idx, now.hour, now.minute, now.second), image)

def read_samples_from_disk(samples):
    images = []
    for sample in samples:
        images.append(cv2.imread(sample))
    return images

def inference(images, model):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    predictor = DefaultPredictor(cfg)
    
    for idx, image in enumerate(images):        
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        now = datetime.datetime.now()
        cv2.imwrite('output/task_c_inference_{}_{}_{}_{}_{}.png'.format(prefix, idx, now.hour, now.minute, now.second), v.get_image()[:, :, ::-1])

samples = [
    '/home/mcv/m5/datasets/MIT_split/train/Opencountry/fie23.jpg',
    '/home/mcv/m5/datasets/MIT_split/train/street/art976.jpg',
    '/home/mcv/m5/datasets/MIT_split/train/forest/cdmc12.jpg',
    '/home/mcv/m5/datasets/MIT_split/train/inside_city/gre102.jpg',
    '/home/mcv/m5/datasets/MIT_split/train/highway/urb681.jpg']

# List of models for retina: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#retinanet
model = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

images = read_samples_from_disk(samples)
write_images_into_disk(images)
inference(images, model)