# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


import os


def write_detection_format(class_str, box, score):
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
    line.append(score)
    return line


if __name__ == "__main__":
    dataset_location = "../../KITTI-MOTS/training/image_02/"
    detection_output_path = "./detection/data/"
    scores_threshold = 0.5
    # Pre_trained_model = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    Pre_trained_model = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(Pre_trained_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(Pre_trained_model)
    predictor = DefaultPredictor(cfg)



    for sequence_id, sequence in enumerate(os.listdir(dataset_location)):
        sequence_location = "{}{}/".format(dataset_location, sequence)
        for frame_id, frame in enumerate(os.listdir(sequence_location)):
            frame_filename = "{}{}".format(sequence_location, frame)
            im = cv2.imread(frame_filename)
            outputs = predictor(im)

            # print(outputs)
            classes = outputs["instances"].pred_classes.to("cpu").numpy()
            boxes = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()
            scores = outputs["instances"].scores.to("cpu").numpy()
            filename = '{}{}_{}'.format(detection_output_path, sequence, frame.replace(".png", ".txt"))
            f = open(filename, 'w+')
            person_index = np.where(classes==0)[0]
            car_index = np.where(classes==2)[0]
            if len(person_index) > 0:
                for k in range(person_index.shape[0]):
                    if scores[person_index[k]] > scores_threshold:
                        line = write_detection_format("Pedestrian", boxes[person_index[k]], scores[person_index[k]])
                        print(" ".join(str(i) for i in line), file=f)
            if len(car_index) > 0:
                for k in range(car_index.shape[0]):
                    if scores[car_index[k]] > scores_threshold:
                        line = write_detection_format("Car", boxes[car_index[k]], scores[car_index[k]])
                        print(" ".join(str(i) for i in line), file=f)
            f.close()



"""
The detection of pretrained model is like:
['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 ...

"""
