from loader import (
    load_kitti_mots_gt, 
    split_kitti_mots_training_dataset,
    load_mots_challenge_gt, 
    load_mots_challenge_dataset
)
import json
from model import train_model, inference
from model import eval_model_using_kitti_mots, eval_model_using_coco
from detectron2 import model_zoo
import argparse

"""
Task (a): Apply pre-trained and finetuned Mask-RCNN models to MOTSChallenge training set
    • Get quantitative and qualitative results for both object detection and object segmentation
    • Analyze the different configurations depending on:
        – Use of training data 
            • COCO
            • COCO+Cityscapes
            • COCO+KITTI-MOTS
            • COCO+Cityscapes+KITTI-MOTS
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="Name of the model do you want to train")
    parser.add_argument("--model", help="Model do you want to train")
    args = parser.parse_args()
    
    print('Model name:', args.model_name)
    print('Model: ', args.model)

    """
    gt = load_kitti_mots_gt(load_from_disk=True)
    dataset = split_kitti_mots_training_dataset()     
    train_model(args.model_name, args.model, dataset, gt, 'KITTI-MOTS', iterations=2500, th=0.8)
    """

    # Train KITTI-MOTS
    
    kitti_mots_gt = load_kitti_mots_gt(load_from_disk=True)
    kitti_training, kitti_validation = split_kitti_mots_training_dataset()
    
    mots_challenge_gt = load_mots_challenge_gt(load_from_disk=True)
    mots_training, mots_validation = load_mots_challenge_dataset()
    

    # Merge KITTI-MOTS and MOTS-Challenge
    
    merged_gt = {}
    for frame, data in mots_challenge_gt.items():
        prefix, suffix = frame.split('_')
        new_frame = '00{}'.format(str(int(prefix) + 21))
        new_key = '{}_{}'.format(new_frame, suffix)
        if (new_key in merged_gt):
            print('Ho noes')
        merged_gt[new_key] = data
    for frame, data in kitti_mots_gt.items():
        if frame in merged_gt:
            print('Ho noes')
        merged_gt[frame] = data
    

    merged_training_dataset = []
    for filename, frame in mots_training:
        prefix, suffix = frame.split('_')
        new_frame = '00{}'.format(str(int(prefix) + 21))
        merged_training_dataset.append((filename, '{}_{}'.format(new_frame, suffix)))
    
    for filename, frame in kitti_training:
        merged_training_dataset.append((filename, frame))
    

    merged_validation_dataset = []
    for filename, frame in mots_validation:
        prefix, suffix = frame.split('_')
        new_frame = '00{}'.format(str(int(prefix) + 21))
        merged_validation_dataset.append((filename, '{}_{}'.format(new_frame, suffix)))
    
    for filename, frame in kitti_validation:
        merged_validation_dataset.append((filename, frame))

    train_model(zoo_model_name=args.model_name, 
                zoo_model=args.model, 
                training=mots_training,
                validation=mots_validation,
                gt=mots_challenge_gt, 
                dataset_name='MOTSChallenge-TEST',
                classes=['Car', 'Pedestrian'],
                iterations=5000, 
                th=0.8)
    

    """
    train_model(zoo_model_name=args.model_name, 
                zoo_model=args.model, 
                training=mots_training,
                validation=mots_validation,
                gt=mots_challenge_gt, 
                dataset_name='MOTSChallenge-TEST',
                classes=['Car', 'Pedestrian'],
                iterations=5000, 
                th=0.8)
    """

    # Eval MOTSChallenge with COCO    
    
    
    """
    eval_model_using_coco(zoo_model_name=args.model_name,
               zoo_model=args.model,
               training=mots_training,
               validation=kitti_validation,
               gt=kitti_mots_gt,
               dataset_name='KITTI-MOTS-COCO',
               classes=['Car', 'Pedestrian'])
    """
    

    
    # Eval MOTSChallenge with COCO + KITTI-MOTS / + Cityscapes
    """
    eval_model_using_kitti_mots(zoo_model_name=args.model_name,
               zoo_model=args.model,
               validation_dataset=mots_challenge_dataset,
               validation_gt=mots_challenge_gt,
               training_dataset=kitti_mots_dataset,
               training_gt=kitti_mots_gt,
               dataset_name='MOTSChallenge',
               validation_classes=['Car', 'Pedestrian'],
               training_classes=['Car', 'Pedestrian'])    
    """