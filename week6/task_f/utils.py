import os
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm
from copy import deepcopy
import imageio


def bb_iou(bboxA, bboxB):
    # This implements a function to compute the intersection over union of two bounding boxes, also known as the Jaccard Index.
    # I've adapted this code from the M1 project code we implemented. The Format of the bboxes is [tlx, tly, brx, bry, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # computing coordinates of the intersection rectangle

    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])

    # compute the area of intersection rectangle

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both predicted and ground-truth bboxes

    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)

    try:
        iou = interArea / float(bboxAArea + bboxBArea - interArea)
    except ZeroDivisionError:
        iou = 0.0

    # returns the intersection over union value
    return iou


def addBboxesToFrames_gif(framesPath, detections, groundTruth, start_frame, end_frame, name):
    #Show GT bboxes and detections
    #Preprocess detections and GT
    for detection in detections:
        detection['isGT'] = False
    for item in groundTruth:
        item['isGT'] = True

    combinedList = detections + groundTruth

    combinedList = sortDetectionsByKey(combinedList, 'frame')

    images = []
    firstFrame = combinedList[0]['frame']

    prevFrame = 0
    filename = "{}{}.jpg".format(framesPath, str(start_frame).zfill(5))
    frameMat = cv2.imread(filename)
    for item in tqdm(combinedList):
        frame = item['frame']
        if frame < start_frame or frame > end_frame:
            continue
        if frame != prevFrame:
            resized = cv2.resize(frameMat, (480, 270), interpolation=cv2.INTER_AREA)
            images.append(resized)
            filename = "{}{}.jpg".format(framesPath, str(item['frame']).zfill(5))
            frameMat = cv2.imread(filename)
        startPoint = (int(item['left']), int(item['top']))
        endPoint = (int(startPoint[0] + item['width']), int(startPoint[1] + item['height']))
        color = (255, 0, 0) if item['isGT'] is True else (0, 0, 255)
        frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
        prevFrame = frame
    imageio.mimsave(name + '.gif', images)


def addBboxesToFrames_avi(framesPath, detections_ori, groundTruth_ori, name):
    """
    This function produce the video in the format of .avi.
    :param framesPath:
    :param detections_ori:
    :param groundTruth_ori:
    :param name:
    :return:
    """
    # the following operation will change the elements in detections and gt.
    # so deepcopy and protect the original data.
    groundTruth = deepcopy(groundTruth_ori)
    detections = deepcopy(detections_ori)
    #Show GT bboxes and detections
    #Preprocess detections and GT
    for detection in detections:
        detection['isGT'] = False
    for item in groundTruth:
        item['isGT'] = True

    combinedList = detections + groundTruth

    combinedList.sort(key=lambda x: x['frame'])

    frameFiles = sorted(glob.glob(framesPath + '/*.png'))
    frame_size = cv2.imread(frameFiles[0]).shape
    size = (frame_size[1], frame_size[0])
    fps = 10
    out = cv2.VideoWriter(name + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    prevFrame = 0
    frameMat = cv2.imread(frameFiles[0])
    for item in tqdm(combinedList):
        frame = item['frame']
        if frame != prevFrame:
            out.write(frameMat)
            frameMat = cv2.imread(frameFiles[frame])
        startPoint = (int(item['left']), int(item['top']))
        endPoint = (int(startPoint[0] + item['width']), int(startPoint[1] + item['height']))
        color = (255, 0, 0) if item['isGT'] is True else (0, 0, 255)
        frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
        prevFrame = frame
    out.release()


def box(o):
    return [o['left'], o['top'], o['left'] + o['width'], o['top'] + o['height']]
def calculate_mAP(groundtruth_list_original, detections_list_original, IoU_threshold=0.5, have_confidence = True, verbose = False):

    groundtruth_list = deepcopy(groundtruth_list_original)
    detections_list = deepcopy(detections_list_original)

    # Sort detections by confidence
    if have_confidence:
        detections_list.sort(key=lambda x: x['confidence'], reverse=True)
    # Save number of groundtruth labels
    groundtruth_size = len(groundtruth_list)

    TP = 0
    FP = 0
    FN = 0

    precision = list(); recall = list()


    # to compute mAP
    threshold = 1
    checkpoint = 0
    temp = 1000

    for n, detection in enumerate(detections_list):
        match_flag = False
        if threshold != temp:

            #print(threshold)

            temp = threshold

        # Get groundtruth of the target frame
        gt_on_frame = [x for x in groundtruth_list if x['frame'] == detection['frame']]
        gt_bboxes = [(box(o), o['confidence'] if ('confidence' in o) else 1) for o in gt_on_frame]


        #print(gt_bboxes)

        for gt_bbox in gt_bboxes:
            iou = bb_iou(gt_bbox[0], box(detection))
            if iou > IoU_threshold and gt_bbox[1] > 0.9:
                match_flag = True
                TP += 1

                gt_used = next((x for x in groundtruth_list if x['frame'] == detection['frame'] and box(x) == gt_bbox[0]), None)
                gt_used['confidence'] = 0
                break

        if match_flag == False:
            FP += 1

        # Save metrics

        precision.append(TP/(TP+FP))
        if groundtruth_size:
            recall.append(TP/groundtruth_size)

    recall_step = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    precision_step = [0] * 11
    max_precision_i = 0.0
    for i in range(len(recall)):
        recall_i = recall[-(i+1)]
        precision_i = precision[-(i+1)]
        max_precision_i = max(max_precision_i, precision_i)
        for j in range(len(recall_step)):
            if recall_i >= recall_step[j]:
                precision_step[j] = max_precision_i
            else:
                break
    if verbose:
        plt.figure(1)

        plt.plot(recall, precision,'r--')
        plt.xlim((0, 1.0))
        plt.ylim((0, 1.0))
        plt.title('Precision - recall curve')
        plt.plot(recall_step, precision_step,'g--')
        plt.show()

    # Check false negatives
    FN = len(detections_list) - TP
    # groups = defaultdict(list)
    # for obj in groundtruth_list:
    #     groups[obj['frame']].append(obj)
    # grouped_groundtruth_list = groups.values()
    #
    # for groundtruth in grouped_groundtruth_list:
    #     detection_on_frame = [x for x in detections_list if x['frame'] == groundtruth[0]['frame']]
    #     detection_bboxes = [box(o) for o in detection_on_frame]
    #
    #     groundtruth_bboxes = [box(o) for o in groundtruth]
    #
    #     results = get_single_frame_results(detection_bboxes, groundtruth_bboxes, IoU_threshold)
    #     FN_temp = results['false_neg']
    #
    #     FN += FN_temp

    if verbose:
        print("TP={} FN={} FP={}".format(TP, FN, FP))

    if TP > 0:
        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        F1_score = 2 * recall * precision / (recall + precision)

    #print(TP+FP)
    #print("precision:{}".format(precision))
    #print("recall:{}".format(recall))
    mAP = sum(precision_step)/11
    if verbose:
        print("mAP: {}".format(mAP))

    #return precision, recall, precision_step, F1_score, mAP
    return mAP