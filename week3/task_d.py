import glob
import json
import os
from pycocotools.mask import toBbox, decode

def get_predictor(model, th):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    return DefaultPredictor(cfg)

def preditct(predictor, dataset):
    pass

def split_training_dataset(config):
    dataset = []
    
    frames = []
    for frame in sorted(os.listdir(config['training_datasets']['KITTI-MOTS'])):
        frames.append(frame)
    
    
    

def get_kitti_gt_files(config):
    return sorted([f for f in glob.glob(config['gt']['KITTI-MOTS'] + '**/*.txt')])

def generate_gt(gt_files):
    for f in glob.glob('gt' + '**/*.txt'):
        os.remove(f)

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
                with open('gt/{}_{}.txt'.format(gt_name, filename), 'a') as f:
                    if last_frame == frame:
                        f.write('\n{} {} {} {} {} {}'.format(frame, object_id, class_id, height, width, bbox))
                    else:
                        f.write('{} {} {} {} {} {}'.format(frame, object_id, class_id, height, width, bbox))
                last_frame = frame

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    gt_files = get_kitti_gt_files(config)
    generate_gt(gt_files)
    split_training_dataset(config)


    
    

    