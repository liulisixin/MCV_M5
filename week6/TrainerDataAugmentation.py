import glob
import json
import cv2
import os
from random import shuffle
import shutil
import numpy as np
import torch
import random
import logging

import copy
from detectron2.structures import BoxMode
from detectron2.config import get_cfg, CfgNode
from pathlib import Path
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from pycocotools.mask import toBbox, decode
from pycocotools import coco
from LossEvalHook import *
from PlotTrainer import *
from plotter import plot_loss_curve
import sys
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog

class TrainerDataAugmentation(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=CustomDatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))

class CustomDatasetMapper(DatasetMapper):
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.resize = cfg.RESIZE
        self.enable_angles = cfg.ANGLE
        self.angles = cfg.ANGLES
        self.expand = cfg.EXPAND
        self.enable_crop = cfg.CROP
        self.crop_type = cfg.CROP_TYPE
        self.crop_size = cfg.CROP_SIZE
        self.enable_flip = cfg.FLIP
        self.flip_prob = cfg.FLIP_PROB
        self.flip_horizontal = cfg.FLIP_HORIZONTAL
        self.flip_vertical = cfg.FLIP_VERTICAL

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
                        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2]

        transformations = []

        if self.is_train and self.enable_angles:
            transform = T.RandomRotation(self.angles, self.expand)
            transformations.append(transform)
        if self.is_train and self.enable_crop:
            transform = T.RandomCrop(self.crop_type, self.crop_size)
            transformations.append(transform)
        if self.is_train and self.enable_flip:
            transform = T.RandomFlip(self.flip_prob, horizontal=self.flip_horizontal, vertical=self.flip_vertical)
            transformations.append(transform)

        image, transforms = T.apply_transform_gens(transformations, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict