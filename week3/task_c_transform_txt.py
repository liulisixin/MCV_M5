import os
from pycocotools.mask import toBbox, decode


def read_txt(gt_file):
    gt = []
    gt_one_frame = []
    time_frame_last = 0
    for line in open(gt_file):
        fields = line.split()
        time_frame = int(fields[0])
        object_id = fields[1]
        class_id = fields[2]
        # do like https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_common/io.py,
        # mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
        mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
        # print(mask)
        # https://github.com/VisualComputingInstitute/mots_tools/blob/master/mots_vis/visualize_mots.py
        # x, y, w, h = rletools.toBbox(obj.mask)
        box = toBbox(mask)
        instance = {}
        instance["class_id"] = int(class_id)
        instance["bbox_left"] = box[0]
        instance["bbox_top"] = box[1]
        instance["bbox_right"] = box[0] + box[2]
        instance["bbox_bottom"] = box[1] + box[3]

        # print(instance)
        if time_frame == time_frame_last:
            gt_one_frame.append(instance)
        else:
            gt.append(gt_one_frame)
            # if time_frame != len(gt):
            #     #raise Exception("time_frame != len(gt)")
            #     print("error")
            gt_one_frame = []
            gt_one_frame.append(instance)

        time_frame_last = time_frame

    if len(gt_one_frame) > 0:
        gt.append(gt_one_frame)

    return gt


if __name__ == "__main__":
    # gt_file = "../../KITTI-MOTS/instances_txt/0000.txt"
    # gt = read_txt(gt_file)
    gt_path = "../../KITTI-MOTS/instances_txt/"
    gt_transformed_path = "./gt_transformed/"
    thing_classes = ['Car', 'Pedestrian', 'DontCare']
    for gt_id, gt_txt in enumerate(os.listdir(gt_path)):
        gt_index_str = gt_txt.split(".")[0]
        gt_file = "{}{}".format(gt_path, gt_txt)
        gt = read_txt(gt_file)
        for frame_id in range(len(gt)):
            frame = gt[frame_id]
            filename = '{}{}_{}.txt'.format(gt_transformed_path, gt_index_str, str(frame_id).zfill(6))
            f = open(filename, 'w+')
            for i in range(len(frame)):
                line = []
                if frame[i]["class_id"] == 1:
                    line.append(thing_classes[0])
                elif frame[i]["class_id"] == 2:
                    line.append(thing_classes[1])
                else:
                    line.append(thing_classes[2])
                line.append(-1)
                line.append(-1)
                line.append(-10)
                line.append(frame[i]["bbox_left"])
                line.append(frame[i]["bbox_top"])
                line.append(frame[i]["bbox_right"])
                line.append(frame[i]["bbox_bottom"])
                line.append(-1)
                line.append(-1)
                line.append(-1)
                line.append(-1000)
                line.append(-1000)
                line.append(-1000)
                line.append(-1000)
                # line.append(scores[i])
                print(" ".join(str(i) for i in line), file=f)
            f.close()

        pass


