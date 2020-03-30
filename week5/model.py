import glob
import json
import cv2
import os
from random import shuffle
import shutil
import numpy as np
import random
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from pathlib import Path
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from pycocotools.mask import toBbox, decode
from pycocotools import coco
import sys
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from loader import load_kitti_mots_gt, split_kitti_mots_training_dataset

def register_dataset(dataset, gt):
    dataset_dicts = []
    for idx, (img_filename, img_file) in enumerate(dataset):
        if img_file in gt:
            record = {}
            record['file_name'] = img_filename
            record['image_id'] = idx
            record['height'] = gt[img_file]['height']
            record['width'] = gt[img_file]['width']
            for x in gt[img_file]['obj']:
                x['bbox_mode'] = BoxMode.XYXY_ABS
            record['annotations'] = gt[img_file]['obj']
            dataset_dicts.append(record)
    return dataset_dicts

def train_model(zoo_model_name, 
                zoo_model, 
                training,
                validation, 
                gt, 
                dataset_name,
                classes,
                lr=0.00025, 
                iterations=1000, 
                th=0.7):
                
    output_path = './output-training-{}-{}'.format(dataset_name, zoo_model_name)
    if (os.path.exists(output_path)):
        os.system('rm -rf {}'.format(output_path))    
    
    DatasetCatalog.register('validation', lambda : register_dataset(validation, gt))
    DatasetCatalog.register('training', lambda : register_dataset(training, gt))   
    MetadataCatalog.get('validation').set(thing_classes=classes)
    MetadataCatalog.get('training').set(thing_classes=classes)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_model))
    cfg.DATASETS.TRAIN = ("training",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_model)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = output_path

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.DATASETS.TEST = ('validation',)
    
    evaluator = COCOEvaluator("validation", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, 'validation')
    inference_on_dataset(trainer.model, val_loader, evaluator)

    predictor = DefaultPredictor(cfg)
    i = 0
    for image in os.listdir('/home/mcv/datasets/KITTI-MOTS/training/image_02/0019'):
        im = cv2.imread('/home/mcv/datasets/KITTI-MOTS/training/image_02/0019/{}'.format(image))
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('{}/sample_{}.png'.format(output_path, i), v.get_image()[:, :, ::-1])
        i += 1

def filter_preds(preds, cat_mapping_coco_kitti):
    for pred in preds:
        pred['instances'] = [i for i in pred['instances'] if i['category_id'] in cat_mapping_coco_kitti.keys()]
        for instance in pred['instances']:
            instance['category_id'] = cat_mapping_coco_kitti[instance['category_id']]

    return preds

def eval_model_using_coco(zoo_model_name, 
                          zoo_model, 
                          training,
                          validation,
                          gt,
                          dataset_name,
                          classes,
                          lr=0.00025, 
                          iterations=1000, 
                          th=0.7):

    output_path = './output-eval-{}-{}'.format(dataset_name, zoo_model_name)
    if (os.path.exists(output_path)):
        os.system('rm -rf {}'.format(output_path))    

    DatasetCatalog.register('training', lambda : register_dataset(training, gt))
    MetadataCatalog.get('training').set(thing_classes=classes)
    DatasetCatalog.register('validation', lambda : register_dataset(validation, gt))
    MetadataCatalog.get('validation').set(thing_classes=classes)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_model))
    cfg.DATASETS.TRAIN = ('training',)
    cfg.DATASETS.TEST = ('validation',)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_model)
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.OUTPUT_DIR = output_path

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)    

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator("validation", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "validation")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    preds = evaluator._predictions

    filtered_preds = filter_preds(preds, {2 : 0, 0 : 1})
    evaluator._predictions = filtered_preds

    evaluator.evaluate()
    
    predictor = DefaultPredictor(cfg)
    i = 0
    for image in os.listdir('/home/mcv/datasets/KITTI-MOTS/training/image_02/0019'):
        im = cv2.imread('/home/mcv/datasets/KITTI-MOTS/training/image_02/0019/{}'.format(image))
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('{}/sample_{}.png'.format(output_path, i), v.get_image()[:, :, ::-1])
        i += 1


def inference(zoo_model_name, zoo_model, dataset_name):
    output_path = './output-eval-{}-{}'.format(dataset_name, zoo_model_name)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(zoo_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_model)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)    
    
    predictor = DefaultPredictor(cfg)
    im = cv2.imread('/home/mcv/datasets/MOTSChallenge/train/images/0002/000470.jpg')
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('output/sample.png'.format(output_path), v.get_image()[:, :, ::-1])

def eval_model_using_kitti_mots(zoo_model_name, 
                                zoo_model, 
                                validation_dataset, 
                                training_dataset,
                                validation_gt,
                                training_gt,
                                dataset_name,
                                validation_classes,
                                training_classes,
                                lr=0.00025, 
                                iterations=5000, 
                                th=0.8):

    output_path = './output-eval-{}-{}'.format(dataset_name, zoo_model_name)
    if (os.path.exists(output_path)):
        os.system('rm -rf {}'.format(output_path))    

    DatasetCatalog.register('training', lambda : register_dataset(training_dataset, training_gt))   
    MetadataCatalog.get('training').set(thing_classes=training_classes)
    DatasetCatalog.register('validation', lambda : register_dataset(validation_dataset, validation_gt))
    MetadataCatalog.get('validation').set(thing_classes=validation_classes)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_model))
    cfg.DATASETS.TRAIN = ("training",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_model)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = output_path

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()    

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.DATASETS.TEST = ('validation',)
    
    evaluator = COCOEvaluator("validation", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, 'validation')
    inference_on_dataset(trainer.model, val_loader, evaluator)
    
    predictor = DefaultPredictor(cfg)
    im = cv2.imread('/home/mcv/datasets/MOTSChallenge/train/images/0002/000470.jpg')
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('{}/sample.png'.format(output_path), v.get_image()[:, :, ::-1])
