import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from random import shuffle
import pycocotools.mask as rletools
import PIL.Image as Image
from pycocotools.mask import toBbox, decode
from pycocotools import coco

"""
Key offset      Seq     Offset
----------      ---     ------

KITTI-MOTS      0-20    0-20    +0
MOTS-CHALLENGE  0-20    42-62   +42 
VKITTI          0-20    21-41   +21

"""

def split_frame(frame):
    splitted_frame = frame.split(' ')
    return int(splitted_frame[0]), int(splitted_frame[2]), int(splitted_frame[3]), int(splitted_frame[4]), splitted_frame[5]

def get_bbox(rle):
    bbox = list(toBbox(rle))
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox

def get_segmentation_by_mask(mask):    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    return segmentation

def get_segmentation(rle):
    return get_segmentation_by_mask(coco.maskUtils.decode(rle))

def split_dataset(dataset, training_offset, split_randomly=False):
    keys = list(dataset.keys())
    if split_randomly:
        shuffle(keys)

    training = {}
    validation = {}
    for key in keys[:training_offset]:
        training[key] = dataset[key]
    for key in keys[training_offset:]:
        validation[key] = dataset[key]

    return training, validation

def remove_from_dictionary(gt, remove_offset, remove_randomly=False):    
    keys = list(gt.keys())
    if remove_randomly:
        shuffle(keys)        
    keys = keys[remove_offset:]
    new_gt = {}
    for key in keys:
        new_gt[key] = gt[key] 
    return new_gt

def load_gt(dataset_type, load_from_disk=False):
    """
    Car = 0
    Pedestrian = 1
    """
    def load_mots_gt(gt_path, offset):
        gt = {}
        for frames in sorted(os.listdir(gt_path)):
            filename = os.path.join(gt_path, frames)
            with open(filename, 'r') as f:
                for raw_frame in f:
                    frame, category, height, width, rle = split_frame(raw_frame)
                    if category not in [1,2]:
                        continue

                    rle = {'size': [height, width], 'counts': rle.strip()}
                    segmentation = get_segmentation(rle)
                    
                    if len(segmentation) == 0:
                        continue

                    category -= 1

                    bbox = get_bbox(rle)

                    frame = '{0:06}'.format(frame)
                    offset_key = '{0:04}'.format(int(frames.replace('.txt', '')) + offset)
                    key = '{}_{}'.format(offset_key, frame)

                    if key not in gt:
                        gt[key] = {'height' : height, 'width' : width, 'obj': []}
                    
                    gt[key]['obj'].append({
                        'category_id' : category, 
                        'bbox' : bbox, 
                        'segmentation' : segmentation
                    })
        return gt    

    def load_vkitti_gt(gt_path, offset):
        gt = {}

        for frame_folder in sorted(os.listdir(gt_path)):
            frame_path = os.path.join(gt_path, frame_folder, 'clone/frames/instanceSegmentation/Camera_0')
            for mask_file in sorted(os.listdir(frame_path)):
                mask_path = os.path.join(frame_path, mask_file)

                img = np.array(Image.open(mask_path))
                height, width = img.shape[:2]

                category = 0 # Always be a car
                frame = mask_file.replace('instancegt_', '').replace('.png', '')
                instances = np.unique(img)
                mask = np.zeros(img.shape, dtype=np.uint8, order="F")

                for obj_id in instances[1:]:
                    mask.fill(0)
                    mask[np.where(img == obj_id)] = 255 # Highlight the elements
                    segmentation = get_segmentation_by_mask((mask).astype(np.uint8))
                    
                    if len(segmentation) == 0:
                        continue

                    RLEs = rletools.frPyObjects(segmentation, height, width)
                    rle = rletools.merge(RLEs)
                    bbox = get_bbox(rle)

                    offset_key = '{0:04}'.format(int(frame_folder) + offset)
                    key = '{}_{}'.format(offset_key, frame)

                    if key not in gt:
                        gt[key] = {'height' : height, 'width' : width, 'obj': []}
                    
                    gt[key]['obj'].append({
                        'category_id' : category,
                        'bbox' : bbox, 
                        'segmentation' : segmentation
                    })
        return gt

    is_vkitti = False
    offset = 0
    if dataset_type.lower() == 'kitti-mots':
        offset = 0
        gt_path = '/home/mcv/datasets/KITTI-MOTS/instances_txt'
        if load_from_disk and os.path.exists('./kitti-mots-gt.pickle'):
            with open('./kitti-mots-gt.pickle', 'rb') as f:
                return pickle.load(f)                
    elif dataset_type.lower() == 'mots-challenge':
        offset = 42
        gt_path = '/home/mcv/datasets/MOTSChallenge/train/instances_txt'
        if load_from_disk and os.path.exists('./mots-challenge-gt.pickle'):
            with open('./mots-challenge-gt.pickle', 'rb') as f:
                return pickle.load(f)
    elif dataset_type.lower() == 'vkitti':
        offset = 21
        is_vkitti = True
        gt_path = '/home/grupo06/vKITTI'
        if load_from_disk and os.path.exists('./vkitti-gt.pickle'):
            with open('./vkitti-gt.pickle', 'rb') as f:
                return pickle.load(f)
    else:
        raise NotImplementedError('The dataset type is not correct')

    if is_vkitti:
        gt = load_vkitti_gt(gt_path, offset)
    else:
        gt = load_mots_gt(gt_path, offset)

    if dataset_type.lower() == 'kitti-mots':
        with open('./kitti-mots-gt.pickle', 'wb') as f:
            pickle.dump(gt, f)
    elif dataset_type.lower() == 'mots-challenge':
        with open('./mots-challenge-gt.pickle', 'wb') as f:
            pickle.dump(gt, f)
    elif dataset_type.lower() == 'vkitti':
        with open('./vkitti-gt.pickle', 'wb') as f:
            pickle.dump(gt, f)
    else:
        raise NotImplementedError('The dataset type is not correct')

    return gt

def load_dataset(dataset_type, filter_by_sequence=None):
    def load_vkitti(dataset_path, offset):
        dataset = {}
        for frame in sorted(os.listdir(dataset_path)):
            frame_path = os.path.join(dataset_path, frame, 'clone/frames/rgb/Camera_0')            

            for img_file in os.listdir(frame_path):
                if '.png' in img_file or '.jpg' in img_file:
                    img_path = os.path.join(frame_path, img_file)
                    img_file_without_extension = img_file.replace('.png', '').replace('.jpg', '').replace('rgb_', '')
                    offset_key = '{0:04}'.format((int(frame) + offset))
                    key = '{}_{}'.format(offset_key, img_file_without_extension)
                    dataset[key] = {'path' : img_path}

        return dataset

    def load_mots(dataset_path, filter_by_sequence, offset):            
        dataset = {}

        for frame in sorted(os.listdir(dataset_path)):
            frame_path = os.path.join(dataset_path, frame)

            if filter_by_sequence is not None and int(frame) not in filter_by_sequence:
                continue                

            for img_file in os.listdir(frame_path):
                if '.png' in img_file or '.jpg' in img_file:
                    img_path = os.path.join(dataset_path, frame, img_file)
                    img_file_without_extension = img_file.replace('.png', '').replace('.jpg', '')
                    offset_key = '{0:04}'.format((int(frame) + offset))
                    key = '{}_{}'.format(offset_key, img_file_without_extension)
                    dataset[key] = {'path' : img_path}
        return dataset

    is_vkitti = False
    offset = 0
    if dataset_type.lower() == 'kitti-mots':
        dataset_path = '/home/mcv/datasets/KITTI-MOTS/training/image_02'
        offset = 0
    elif dataset_type.lower() == 'mots-challenge':
        dataset_path = '/home/mcv/datasets/MOTSChallenge/train/images'
        offset = 42
    elif dataset_type.lower() == 'vkitti':
        dataset_path = '/home/grupo06/vKITTI'
        is_vkitti = True
        offset = 21
    else:
        raise NotImplementedError('The dataset type is not correct')

    if is_vkitti:
        return load_vkitti(dataset_path, offset)
    return load_mots(dataset_path, filter_by_sequence, offset)
