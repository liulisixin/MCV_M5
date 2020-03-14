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

def write_images_into_disk(images):
    for idx, image in enumerate(images):
        cv2.imwrite('output/task_b_sample_{}.png'.format(idx), image)

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
        cv2.imwrite('output/task_b_inference_{}.png'.format(idx), v.get_image()[:, :, ::-1])

samples = [
    '../../KITTI/data_object_image_2/training/image_2/000000.png',
    '../../KITTI/data_object_image_2/training/image_2/000001.png',]

