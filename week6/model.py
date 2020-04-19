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
from TrainerDataAugmentation import TrainerDataAugmentation

class Model():
    def __init__(self, name, training_dataset, validation_dataset, test_dataset, gt, classes):
        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset
        self._gt = gt
        self._name = name
        self._classes = classes
        self._cfg = None
    
    def train(self, zoo_model_name, zoo_model, lr, iterations, th, data_agumentation=None):
        
        DatasetCatalog.register('validation_{}'.format(self._name), lambda : self.__register_dataset(self._validation_dataset))
        MetadataCatalog.get('validation_{}'.format(self._name)).set(thing_classes=self._classes)

        DatasetCatalog.register('training_{}'.format(self._name), lambda : self.__register_dataset(self._training_dataset))
        MetadataCatalog.get('training_{}'.format(self._name)).set(thing_classes=self._classes)
    
        if self._test_dataset is not None:
            DatasetCatalog.register('test_{}'.format(self._name), lambda : self.__register_dataset(self._test_dataset))
            MetadataCatalog.get('test_{}'.format(self._name)).set(thing_classes=self._classes)

        if data_agumentation is None:
            data_agumentation = {'no-da' : {
                'size' : [(0, 0)],
                'crop' : False,
                'crop_type' : 'relative',
                'crop_size' : [],
                'angle' : False,
                'angles' : [],
                'expand' : False,
                'flip' : False,
                'flip_prob' : 0.0,
                'flip_horizontal' : False,
                'flip_vertical' : False
            }}

        for key, data in data_agumentation.items():
            print(data)
            output_path = './output-{}-{}'.format(self._name, key)
            if (os.path.exists(output_path)):
                os.system('rm -rf {}'.format(output_path))

            self._cfg = get_cfg()
            self._cfg.merge_from_file(model_zoo.get_config_file(zoo_model))
            self._cfg.DATASETS.TRAIN = ('training_{}'.format(self._name),)
            if self._test_dataset is not None:
                self._cfg.DATASETS.TEST = ('validation_{}'.format(self._name),)
            else:
                self._cfg.DATASETS.TEST = ()
            self._cfg.DATALOADER.NUM_WORKERS = 4
            self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_model)
            self._cfg.SOLVER.IMS_PER_BATCH = 4
            self._cfg.SOLVER.BASE_LR = lr
            self._cfg.SOLVER.MAX_ITER = iterations
            self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
            self._cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            self._cfg.OUTPUT_DIR = output_path
            os.makedirs(output_path, exist_ok=True)

            # Set Data augmentation values:
            self._cfg.RESIZE = data['size']
            self._cfg.CROP = data['crop']
            self._cfg.CROP_TYPE = data['crop_type']
            self._cfg.CROP_SIZE = data['crop_size']
            self._cfg.ANGLE = data['angle']
            self._cfg.ANGLES = data['angles']
            self._cfg.EXPAND = data['expand']
            self._cfg.FLIP = data['flip']
            self._cfg.FLIP_PROB = data['flip_prob']
            self._cfg.FLIP_HORIZONTAL = data['flip_horizontal']
            self._cfg.FLIP_VERTICAL = data['flip_vertical']

            trainer = TrainerDataAugmentation(self._cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            
            self._cfg.MODEL.WEIGHTS = os.path.join(self._cfg.OUTPUT_DIR, 'model_final.pth')
            self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
            
            if self._test_dataset is not None:
                self._cfg.DATASETS.TEST = ('test_{}'.format(self._name),)                
                evaluator = COCOEvaluator('test_{}'.format(self._name), self._cfg, False, output_dir=output_path)
            else:
                self._cfg.DATASETS.TEST = ('validation_{}'.format(self._name),)            
                evaluator = COCOEvaluator('validation_{}'.format(self._name), self._cfg, False, output_dir=output_path)
            
            trainer.test(self._cfg, trainer.model, evaluators=[evaluator])
            
            inference_output_path = './output-inference-{}-{}'.format(self._name, key)
            images_path = '/home/mcv/datasets/KITTI-MOTS/training/image_02/0000'
            if (os.path.exists(inference_output_path)):
                os.system('rm -rf {}'.format(inference_output_path))
            os.makedirs(inference_output_path, exist_ok=True)

            predictor = DefaultPredictor(self._cfg)
            for idx, image in enumerate(os.listdir(images_path)):
                im = cv2.imread(os.path.join(images_path, image))
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(self._cfg.DATASETS.TEST[0]), scale=1.2)
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite('{}/sample_{}.png'.format(inference_output_path, idx), v.get_image()[:, :, ::-1])

    def filter_preds(self, preds, cat_mapping_coco_kitti):
        for pred in preds:
            pred['instances'] = [i for i in pred['instances'] if i['category_id'] in cat_mapping_coco_kitti.keys()]
            for instance in pred['instances']:
                instance['category_id'] = cat_mapping_coco_kitti[instance['category_id']]
        return preds

    def eval_model_using_coco(self, zoo_model_name, zoo_model, lr, iterations, th):
        output_path = './output-eval-{}'.format(self._name)
        if (os.path.exists(output_path)):
            os.system('rm -rf {}'.format(output_path))    

        DatasetCatalog.register('validation_{}'.format(self._name), lambda : self.__register_dataset(self._validation_dataset))
        MetadataCatalog.get('validation_{}'.format(self._name)).set(thing_classes=self._classes)

        DatasetCatalog.register('training_{}'.format(self._name), lambda : self.__register_dataset(self._training_dataset))
        MetadataCatalog.get('training_{}'.format(self._name)).set(thing_classes=self._classes)
        
        self._cfg = get_cfg()
        self._cfg.merge_from_file(model_zoo.get_config_file(zoo_model))
        self._cfg.DATASETS.TRAIN = ('training_{}'.format(self._name),)
        self._cfg.DATASETS.TEST = ('validation_{}'.format(self._name),)
        self._cfg.DATALOADER.NUM_WORKERS = 4
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_model)
        self._cfg.SOLVER.IMS_PER_BATCH = 8
        self._cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self._cfg.OUTPUT_DIR = output_path
        os.makedirs(self._cfg.OUTPUT_DIR, exist_ok=True)    

        trainer = DefaultTrainer(self._cfg)
        trainer.resume_or_load(resume=False)

        evaluator = COCOEvaluator('validation_{}'.format(self._name), self._cfg, False, output_dir=output_path)
        val_loader = build_detection_test_loader(self._cfg, 'validation_{}'.format(self._name))
        inference_on_dataset(trainer.model, val_loader, evaluator)

        preds = evaluator._predictions

        filtered_preds = self.filter_preds(preds, {2 : 0, 0 : 1})
        evaluator._predictions = filtered_preds

        evaluator.evaluate()

    def __register_dataset(self, dataset):
        dataset_dicts = []
        for idx, (img_file, data) in enumerate(dataset.items()):
            if img_file in self._gt:
                record = {}
                record['file_name'] = data['path']
                record['image_id'] = idx
                record['height'] = self._gt[img_file]['height']
                record['width'] = self._gt[img_file]['width']
                for x in self._gt[img_file]['obj']:
                    x['bbox_mode'] = BoxMode.XYXY_ABS
                record['annotations'] = self._gt[img_file]['obj']
                dataset_dicts.append(record)
        return dataset_dicts
        