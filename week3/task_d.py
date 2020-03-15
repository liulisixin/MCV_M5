import glob
import json
import cv2
import os
from random import shuffle
import shutil
import numpy as np
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from pycocotools.mask import toBbox, decode
from detectron2.data import DatasetCatalog, MetadataCatalog

def register_dataset(dataset, dataset_name, config):
    dataset_dicts = []
    for idx, (img_filename, img_file) in enumerate(dataset):
        gt_file = '{}.txt'.format(img_file[:-4])

        record = {}
        record['file_name'] = img_filename
        record['image_id'] = idx
        objs = []
        if os.path.exists('./gt_{}/{}'.format(dataset_name, gt_file)):
            with open('./gt_{}/{}'.format(dataset_name, gt_file)) as f:
                for line in f:
                    gt_file_content = line.split(' ')
                    record['height'] = int(gt_file_content[3])
                    record['width'] = int(gt_file_content[4])

                    obj = {
                        'bbox' : [
                            float(gt_file_content[5]),
                            float(gt_file_content[6]),
                            float(gt_file_content[7]),
                            float(gt_file_content[8])],
                        'bbox_mode' : BoxMode.XYXY_ABS,
                        'category_id' : int(gt_file_content[2])
                    }
                    objs.append(obj)
            record['annotations'] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def add_dataset_catalog(dataset, dataset_name, dataset_type, config):
    classes = list(config['classes'].values())

    name = '{}_{}'.format(dataset_name, dataset_type)
    def register() : return register_dataset(dataset, dataset_name, config)

    DatasetCatalog.register(name, register)
    MetadataCatalog.get(name).set(thing_classes=classes)

def split_training_dataset(config, dataset_name):
    dataset = []

    for frame in sorted(os.listdir(config['training_datasets'][dataset_name])):        
        for img_file in sorted(os.listdir('{}/{}'.format(config['training_datasets'][dataset_name], frame))):
            dataset.append(('{}/{}/{}'.format(config['training_datasets'][dataset_name], frame, img_file), 
                            '{}_{}'.format(frame, img_file)))
    shuffle(dataset)

    training = dataset[:int(len(dataset)*config['task_d']['training'])]    
    validation = dataset[-int(len(dataset)*(1.0 - config['task_d']['training']) + 1):]    
    return training, validation

def get_gt_files(config, dataset_name):
    return sorted([f for f in glob.glob(config['gt'][dataset_name] + '**/*.txt')])

def generate_gt(gt_files, dataset_name):
    for gt_file in gt_files:
        gt_name = gt_file.split('/')[-1][:-4]
        last_frame = -1
        with open(gt_file, 'r') as lines:
            for line in lines:                
                splited_line = line.split(' ')
                frame = int(splited_line[0])
                object_id = int(splited_line[1])
                class_id =int(splited_line[2])
                height = int(splited_line[3])
                width = int(splited_line[4])
                mask = {'size': [height, width], 'counts': splited_line[5].encode(encoding='UTF-8')}
                bbox = list(toBbox(mask))

                filename = '{0:06}'.format(frame)
                with open('gt_{}/{}_{}.txt'.format(dataset_name, gt_name, filename), 'a') as f:
                    if last_frame == frame:
                        f.write('\n{} {} {} {} {} {} {} {} {}'.format(
                            frame, 
                            object_id, 
                            class_id, 
                            height, 
                            width, 
                            bbox[0], 
                            bbox[1], 
                            bbox[0] + bbox[2], 
                            bbox[1] + bbox[3]))
                    else:
                        f.write('{} {} {} {} {} {} {} {} {}'.format(
                            frame, 
                            object_id, 
                            class_id, 
                            height, 
                            width, 
                            bbox[0], 
                            bbox[1], 
                            bbox[0] + bbox[2], 
                            bbox[1] + bbox[3]))
                last_frame = frame

def generate_gt_for_validation(dataset, dataset_name):
    with open('./validation/gt.txt', 'w') as f:
        for idx, (img_filename, img_file) in enumerate(dataset):                
            gt_file = '{}.txt'.format(img_file[:-4])
            if os.path.exists('./gt_{}/{}'.format(dataset_name, gt_file)):
                lines = []
                with open('./gt_{}/{}'.format(dataset_name, gt_file)) as gt_file:
                    for line in gt_file:
                        gt_file_content = line.split(' ')
                        bbox = [float(gt_file_content[5]), 
                                float(gt_file_content[6]), 
                                float(gt_file_content[7]),
                                float(gt_file_content[8])]
                        category_id = int(gt_file_content[2])
                        if category_id == 1:
                            category = 'Pedestrian'
                        else:
                            category = 'Car'
                        line = write_detection_format(category, bbox)
                        lines.append(line)                
                with open('./validation/label_2/{}'.format(gt_file), 'w') as validation_file:
                    for line in lines[:-1]:
                        validation_file.write(line + '\n')
                    validation_file.write(line[-1])

def write_detection_format(class_str, box, score=None):
    line = []
    line.append(class_str)
    line.append(-1)
    line.append(-1)
    line.append(-10)
    line.append(box[0])
    line.append(box[1])
    line.append(box[2])
    line.append(box[3])
    line.append(-1)
    line.append(-1)
    line.append(-1)
    line.append(-1000)
    line.append(-1000)
    line.append(-1000)
    line.append(-1000)
    if score != None:
        line.append(score)
    return line

def generate_model(config):
    gt_files = get_gt_files(config, config['task_d']['dataset_train'])
    generate_gt(gt_files, config['task_d']['dataset_train'])
    training, validation = split_training_dataset(config, config['task_d']['dataset_train'])
    add_dataset_catalog(training, config['task_d']['dataset_train'], 'training', config)
    add_dataset_catalog(validation, config['task_d']['dataset_train'], 'validation', config)

    cfg = train_model(config)
    validate_model(config, cfg, validation)

def train_model(config):
    task_d_model = config['task_d']['model']
    if task_d_model in config['faster_rcnn_models']:
        model = config['faster_rcnn_models'][task_d_model]
    elif task_d_model in config['retina_models']:
        model = config['retina_models'][task_d_model]
    else:
        raise NotImplementedError()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATASETS.TRAIN = ('{}_training'.format(config['task_d']['dataset_train']),)
    cfg.DATASETS.TEST = ('{}_validation'.format(config['task_d']['dataset_train']),)

    if config['task_d']['dataset_train'] == 'KITTI-MOTS':
        cfg.DATALOADER.NUM_WORKERS = config['task_d']['kitti_mots_static_training']['num_workers']
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  
        cfg.SOLVER.IMS_PER_BATCH = config['task_d']['kitti_mots_static_training']['img_per_batch']
        cfg.SOLVER.BASE_LR = config['task_d']['kitti_mots_static_training']['LR']
        cfg.SOLVER.MAX_ITER = config['task_d']['kitti_mots_static_training']['iterations']
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['task_d']['kitti_mots_static_training']['batch_size']
    else:
        cfg.DATALOADER.NUM_WORKERS = config['task_d']['MOTSChallenge_static_training']['num_workers']
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  
        cfg.SOLVER.IMS_PER_BATCH = config['task_d']['MOTSChallenge_static_training']['img_per_batch']
        cfg.SOLVER.BASE_LR = config['task_d']['MOTSChallenge_static_training']['LR']
        cfg.SOLVER.MAX_ITER = config['task_d']['MOTSChallenge_static_training']['iterations']
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['task_d']['MOTSChallenge_static_training']['batch_size']
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()
    return cfg   

def validate_model(config, cfg, validation_dataset):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    if config['task_d']['dataset_train'] == 'KITTI-MOTS':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['task_d']['kitti_mots_static_training']['th']
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['task_d']['MOTSChallenge_static_training']['th']

    predictor = DefaultPredictor(cfg)

    generate_gt_for_validation(validation_dataset, config['task_d']['dataset_train'])
    for img_filename, img_file in validation_dataset:
        im = cv2.imread(img_filename)
        outputs = predictor(im)

        classes = outputs["instances"].pred_classes.to("cpu").numpy()
        boxes = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()
        scores = outputs["instances"].scores.to("cpu").numpy()
        
        filename = '{}/{}'.format('./detection/data', img_file.replace(".png", ".txt"))
        f = open(filename, 'w+')
        person_index = np.where(classes==1)[0]
        car_index = np.where(classes==2)[0]
        if len(person_index) > 0:
            for k in range(person_index.shape[0]):
                line = write_detection_format("Pedestrian", boxes[person_index[k]], scores[person_index[k]])
                print(" ".join(str(i) for i in line), file=f)
        if len(car_index) > 0:
            for k in range(car_index.shape[0]):
                line = write_detection_format("Car", boxes[car_index[k]], scores[car_index[k]])
                print(" ".join(str(i) for i in line), file=f)
        f.close()

def cleanup():
    if os.path.isdir('output'):
        shutil.rmtree('./output')
    for f in glob.glob('detection/' + '**/*.txt'):
        os.remove(f)
    for f in glob.glob('gt_KITTI-MOTS' + '**/*.txt'):
        os.remove(f)
    for f in glob.glob('gt_MOTSChanllenge' + '**/*.txt'):
        os.remove(f)

if __name__ == "__main__":
    cleanup()

    with open('config.json', 'r') as f:
        config = json.load(f)
    print('Configuration: ')
    print(json.dumps(config['task_d'], indent=4))

    generate_model(config)