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


"""
Task (a): Apply pre-trained Mask-RCNN models to KITTI-MOTS validation set
    •Get quantitative and qualitative results for both object detection and object segmentation --> Done
    •Analyze the different configurations depending on:
        –Number of layers (ResNet-50 vs ResNet-101)
        –Backbone configuration (C4 vs DC5)
        –Use of Feature Pyramid Network (C4 vs FPN)
        –Use of training data (COCO vs COCO+Cityscapes
"""

def register_dataset(dataset):
    dataset_dicts = []
    image_id = 1
    for img_filename, img_file in dataset:
        gt_file = '{}.json'.format(img_file[:-4])
        if os.path.exists('./gt_KITTI-MOTS/{}'.format(gt_file)):
            with open('./gt_KITTI-MOTS/{}'.format(gt_file), 'r') as f:
                gt = json.load(f)                    
            record = {}
            record['file_name'] = img_filename
            record['image_id'] = image_id
            record['height'] = gt['height']
            record['width'] = gt['width']
            for x in gt['obj']:
                x['bbox_mode'] = BoxMode.XYXY_ABS
            record['annotations'] = gt['obj']
            dataset_dicts.append(record)
            image_id += 1
    return dataset_dicts

def generate_gt(include_segmentation = False):
    """
    # https://github.com/facebookresearch/Detectron/issues/100
    # https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html
    KITTI-MOTS annotation for classes:
        NAME    KITTI   COCO
        Person  2       0
        Car     1       2
        Other   X       1
    """
    gt_files = sorted([f for f in glob.glob(config['gt'] + '**/*.txt')])
    gt_content = {}
    for gt_file in gt_files:
        gt_name = gt_file.split('/')[-1][:-4]
        with open(gt_file, 'r') as lines:
            for line in lines:
                splited_line = line.split(' ')
                class_id = int(splited_line[2])
                frame = int(splited_line[0])
                height = int(splited_line[3])
                width = int(splited_line[4])
                if class_id == 2: # PEDESTRIAN
                    class_id = 0
                elif class_id == 1: # CAR
                    class_id = 1
                else:
                    class_id = 2
                rle = {'size': [height, width], 'counts': splited_line[5].strip()}
                
                bbox = list(toBbox(rle))
                bbox[0] += bbox[2]
                bbox[1] += bbox[3]

                if include_segmentation:
                    mask = coco.maskUtils.decode(rle)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
                    segmentation = []
                    for contour in contours:
                        contour = contour.flatten().tolist()
                        if len(contour) > 4:
                            segmentation.append(contour)

                    if len(segmentation) == 0:
                        continue
                
                frame_name = '{0:06}'.format(frame)
                key_name = '{}_{}'.format(gt_name, frame_name)

                if key_name not in gt_content:
                    gt_content[key_name] = {}
                    gt_content[key_name]['height'] = height
                    gt_content[key_name]['width'] = width
                    gt_content[key_name]['obj'] = []

                if include_segmentation:
                    gt_content[key_name]['obj'].append({
                            'category_id' : class_id,
                            'bbox' : bbox,
                            'segmentation' : segmentation
                    })
                else:
                    gt_content[key_name]['obj'].append({
                            'category_id' : class_id,
                            'bbox' : bbox
                    })

    for filename, content in gt_content.items():        
        with open('./gt_KITTI-MOTS/{}.json'.format(filename), 'w') as f:
            json.dump(content, f, indent=4)

def split_training_dataset(config):
    dataset = []

    for frame in sorted(os.listdir(config['training_dataset'])):        
        for img_file in sorted(os.listdir('{}/{}'.format(config['training_dataset'], frame))):
            dataset.append(('{}/{}/{}'.format(config['training_dataset'], frame, img_file), 
                            '{}_{}'.format(frame, img_file)))
    shuffle(dataset)

    training = dataset[:int(len(dataset)*config['task_a']['training'])]    
    validation = dataset[-int(len(dataset)*(1.0 - config['task_a']['training']) + 1):]    
    return training, validation

def pre_trained_model(model_name, model, th, model_type, validation_dataset, training_dataset, print_samples=True):    
    MetadataCatalog.get('validation').set(thing_classes=['Pedestrian', 'Car', 'Other'])
    MetadataCatalog.get('training').set(thing_classes=['Pedestrian', 'Car', 'Other'])

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ('training',)
    cfg.DATASETS.TEST = ('validation',)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.OUTPUT_DIR = './{}/{}'.format(model_type, model_name)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    
    build = build_model(cfg)
    DetectionCheckpointer(build).load(cfg.MODEL.WEIGHTS)
    
    trainer = DefaultTrainer(cfg)

    # Eval model
    evaluator = COCOEvaluator("validation", cfg, False, output_dir='./{}/{}'.format(model_type, model_name))
    #val_loader = build_detection_test_loader(cfg, "validation")
    #inference_on_dataset(predictor.model, val_loader, evaluator)

    trainer = DefaultTrainer(cfg)
    trainer.test(cfg, build, evaluators=[evaluator])

def print_examples(model_type, model_name, model, th):    
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    predictor = DefaultPredictor(cfg)

    Path('./output_task_a/{}/{}'.format(model_type, model_name)).mkdir(parents=True, exist_ok=True)
    images_to_print = [
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0017/000137.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0019/000220.png',
        '/home/mcv/datasets/KITTI-MOTS/training/image_02/0002/000097.png'
    ]

    for idx, image in enumerate(images_to_print):
        im = cv2.imread(image)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('./output_task_a/{}/{}/sample_{}.png'.format(model_type, model_name, idx), v.get_image()[:, :, ::-1])

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    print('Configuration: ')
    print(json.dumps(config['task_a'], indent=4))
    
    training, validation = split_training_dataset(config)

    for folder in os.listdir('./segmentation'):
        for f in os.listdir('./segmentation/{}'.format(folder)):
            os.remove('./segmentation/{}/{}'.format(folder, f))    
    for folder in os.listdir('./selection'):
        for f in os.listdir('./selection/{}'.format(folder)):
            os.remove('./selection/{}/{}'.format(folder, f))    
    os.system('rm -rf cityscapes')
    
    def register_validation() : return register_dataset(validation)
    DatasetCatalog.register('validation', register_validation)
    
    def register_training() : return register_dataset(training)
    DatasetCatalog.register('training', register_validation)

    #os.system('rm ./gt_KITTI-MOTS/*.json')
    #generate_gt(include_segmentation=False)
    
    for model_name, model in config['faster_rcnn_models'].items():
        #print_examples('segmentation', model_name, model, config['task_a']['th_segmentation'])
        pre_trained_model(model_name, model, config['task_a']['th_segmentation'], 'selection', validation, training)
        
    #print_examples('cityscapes', 'cityscapes', config['cityscapes'], config['task_a']['th_segmentation'])
    #pre_trained_model('cityscapes', config['cityscapes'], config['task_a']['th_selection'], 'cityscapes', validation, training)
    