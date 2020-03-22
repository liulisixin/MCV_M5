import os
from pycocotools.mask import toBbox, decode
import cv2


def read_gt_txt(gt_txt, gt_path):
    gt_index_str = gt_txt.split(".")[0]
    gt_file = "{}{}".format(gt_path, gt_txt)
    gt = []
    gt_one_frame = []
    time_frame_last = 0
    for line in open(gt_file):
        fields = line.split()
        time_frame = int(fields[0])

        one_instance = [gt_index_str]
        one_instance.extend(fields)

        # print(instance)
        if time_frame == time_frame_last:
            gt_one_frame.append(one_instance)
        else:
            gt.append(gt_one_frame)
            # if time_frame != len(gt):
            #     #raise Exception("time_frame != len(gt)")
            #     print("error")
            gt_one_frame = []
            gt_one_frame.append(one_instance)

        time_frame_last = time_frame

    if len(gt_one_frame) > 0:
        gt.append(gt_one_frame)

    return gt



gt_path = "../../KITTI-MOTS/instances_txt/"
thing_classes = ['Car', 'Pedestrian', 'DontCare']
gt_all = []
for gt_id, gt_txt in enumerate(sorted(os.listdir(gt_path))):
    gt = read_gt_txt(gt_txt, gt_path)
    gt_all.extend(gt)


from random import shuffle
shuffle(gt_all)
train_ratio = 0.8
training = gt_all[:int(len(gt_all)*train_ratio)]
validation = gt_all[-int(len(gt_all)*(1.0 - train_ratio) + 1):]



# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import os
import numpy as np
import json
from detectron2.structures import BoxMode


def register_from_dataset(dataset_path, dataset):

    dataset_dicts = []
    for idx, v in enumerate(dataset):
        instance_0 = v[0]
        record = {}

        filename = "{}/{}/{}.png".format(dataset_path, instance_0[0], instance_0[1].zfill(6))

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = instance_0[4]
        record["width"] = instance_0[5]

        objs = []
        for instance in v:
            # do like https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py,
            # mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
            mask = {'size': [int(instance[4]), int(instance[5])], 'counts': instance[6].encode(encoding='UTF-8')}
            box = toBbox(mask)

            ori_class = int(instance[3])
            if ori_class == 1:
                transform_class = 0
            elif ori_class == 2:
                transform_class = 1
            else:
                transform_class = 2

            obj = {
                "bbox": [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": mask,
                "category_id": transform_class,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog

dataset_path = "../../KITTI-MOTS/training/image_02"
thing_classes = ['Car', 'Pedestrian', 'DontCare']

# g = lambda: register_from_dataset(dataset_path, training)
# g()

DatasetCatalog.register("KITTI_MOTS_training", lambda: register_from_dataset(dataset_path, training))
MetadataCatalog.get("KITTI_MOTS_training").set(thing_classes=thing_classes)
DatasetCatalog.register("KITTI_MOTS_val", lambda: register_from_dataset(dataset_path, validation))
MetadataCatalog.get("KITTI_MOTS_val").set(thing_classes=thing_classes)

KITTI_MOTS_metadata = MetadataCatalog.get("KITTI_MOTS_training")

