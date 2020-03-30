import os
import cv2
import pickle
from random import shuffle
from pycocotools.mask import toBbox, decode
from pycocotools import coco

def split_frame(frame):
    splitted_frame = frame.split(' ')
    return int(splitted_frame[0]), int(splitted_frame[2]), int(splitted_frame[3]), int(splitted_frame[4]), splitted_frame[5]

def get_bbox(rle):
    bbox = list(toBbox(rle))
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    return bbox

def get_segmentation(rle):
    mask = coco.maskUtils.decode(rle)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    return segmentation

def load_gt(path):
    # Pedestrian    = 2 --> 1
    # Car           = 1 --> 0    
    gt = {}
    for frames in sorted(os.listdir(path)):
        filename = os.path.join(path, frames)
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
                key = '{}_{}'.format(frames.replace('.txt', ''), frame)

                if key not in gt:
                    gt[key] = {'height' : height, 'width' : width, 'obj': []}
                
                gt[key]['obj'].append({
                    'category_id' : category, 
                    'bbox' : bbox, 
                    'segmentation' : segmentation,
                    'iscrowd' : 0
                })
    return gt

def split_training_dataset(path):
    dataset = []

    for frame in sorted(os.listdir(path)):
        frame_path = os.path.join(path, frame)   
        for img_file in sorted(os.listdir(frame_path)):
            if '.jpg' in img_file or '.png' in img_file:
                img_path = os.path.join(path, frame, img_file)
                dataset.append((img_path, '{}_{}'.format(frame, img_file.replace('.png', '').replace('.jpg', ''))))
    shuffle(dataset)

    training = dataset[:int(len(dataset)*0.8)]    
    validation = dataset[-int(len(dataset)*0.2 + 1):]    
    return training, validation

def load_kitti_mots_gt(load_from_disk=False):
    if load_from_disk and os.path.exists('./kitti-mots-gt.pickle'):
        with open('./kitti-mots-gt.pickle', 'rb') as f:
            gt = pickle.load(f)
    else:
        path = '/home/mcv/datasets/KITTI-MOTS/instances_txt'
        gt = load_gt(path)
        with open('./kitti-mots-gt.pickle', 'wb') as f:
            pickle.dump(gt, f)
    return gt

def load_mots_challenge_gt(load_from_disk=False):
    if load_from_disk and os.path.exists('./mots-challenge-gt.pickle'):
        with open('./mots-challenge-gt.pickle', 'rb') as f:
            gt = pickle.load(f)
    else:
        path = '/home/mcv/datasets/MOTSChallenge/train/instances_txt'
        gt = load_gt(path)
        with open('./mots-challenge-gt.pickle', 'wb') as f:
            pickle.dump(gt, f)
    return gt

def load_mots_challenge_dataset():
    return split_training_dataset('/home/mcv/datasets/MOTSChallenge/train/images')


def split_kitti_mots_training_dataset():
    return split_training_dataset('/home/mcv/datasets/KITTI-MOTS/training/image_02')
