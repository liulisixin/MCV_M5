"""
example
{'frame': 100, 'left': 931, 'top': 78, 'width': 82, 'height': 68, 'confidence': 0.99}
Note, frame starts from 1.
"""
import pickle
from utils import addBboxesToFrames_avi, calculate_mAP, addBboxesToFrames_gif
from utils_tracking import tracking_filter, calculate_idf1, addTracksToFrames, addTracksToFrames_gif
from utils_read import read_gt_txt, transform_gt
from utils_maximum_overlap import find_tracking_maximum_overlap
import os

from utils_kalman import kalman_filter_tracking_2

if __name__ == "__main__":
    detections_filename = "./detections_all_test_seq.pkl"
    test_path = "../../../KITTI-MOTS/training/image_02/"
    gt_path = "../../../KITTI-MOTS/instances_txt/"

    print("Reading pkl")
    with open(detections_filename, 'rb') as p:
        detections_all_seq = pickle.load(p)
        p.close()

    track_type = "maximum overlap"
    # track_type = "kalman filter"
    # kalman_filter_mode = 0

    thing_classes = ['Car', 'Pedestrian']
    num_classes = len(thing_classes)
    test_seq = [4, 5, 7, 8, 9, 11, 15]
    # test_seq = [4]
    video_length_list = {}
    for id_seq in test_seq:
        count = 0
        for fn in os.listdir(test_path+str(id_seq).zfill(4)):
            count = count + 1
        video_length_list[id_seq] = count

    idf1_list_list = []
    num_gt_tracks_list_list = []
    for id_seq in test_seq:
        print("id_seq = ", id_seq)

        detections_all_labels = detections_all_seq[id_seq]
        print("Reading gt...")
        gt_all_labels = read_gt_txt('{}{}.txt'.format(gt_path, str(id_seq).zfill(4)))

        # get video_length
        video_length = video_length_list[id_seq]

        detections_tracks_all_labels = []
        tracks_gt_list_all_labels = []

        idf1_list = []
        num_gt_tracks_list = []
        for label_id in range(num_classes):
            detections = [x for x in detections_all_labels if x['label'] == label_id]
            gt = [x for x in gt_all_labels if x['label'] == (label_id+1)]
            print("class {}: number of detections = {}, number of gt = {}".format(label_id, len(detections), len(gt)))

            tracks_gt_list = transform_gt(gt)
            num_gt_tracks_list.append(len(tracks_gt_list))

            # print("calculate mAP...")
            # mAP = calculate_mAP(gt, detections, IoU_threshold=0.5, have_confidence=True, verbose=True)
            # print("mAP = ", mAP)

            # addBboxesToFrames_avi('{}{}'.format(test_path, str(id_seq).zfill(4)), detections, gt, "test")
            # addBboxesToFrames_gif(video_path, detections, groundTruth, start_frame=210, end_frame=260, name="test")

            # sort detections for following operations.
            detections.sort(key=lambda x: x['frame'])

            missing_chance = 2
            lou_max_threshold = 0.05

            if track_type == "maximum overlap":
                # maximum overlap
                detections_tracks = find_tracking_maximum_overlap(detections, video_length,
                                                                  missing_chance=missing_chance,
                                                                  lou_max_threshold=lou_max_threshold)
            elif track_type == "kalman filter":
                # kalman
                # mode: 0: velocity; 1: acceleration
                detections_tracks = kalman_filter_tracking_2(detections, video_length, kalman_filter_mode)
            else:
                raise Exception("Wrong track type")

            # filter the track
            # detections_tracks = tracking_filter(detections_tracks)

            idf1 = calculate_idf1(gt, detections_tracks, video_length)
            idf1_list.append(idf1)

            for track_one in detections_tracks:
                track_one.detections.sort(key=lambda x: x['frame'])

            detections_tracks_all_labels.extend(detections_tracks)
            tracks_gt_list_all_labels.extend(tracks_gt_list)

        # addTracksToFrames('{}{}'.format(test_path, str(id_seq).zfill(4)), detections_tracks_all_labels,
        #                   tracks_gt_list_all_labels, start_frame=0, end_frame=video_length,
        #                   name="test_track"+str(id_seq).zfill(4))
        # addTracksToFrames_gif(video_path, detections_tracks, tracks_gt_list, start_frame=210, end_frame=390, name="test")

        idf1_list_list.append(idf1_list)
        num_gt_tracks_list_list.append(num_gt_tracks_list)

    for x in idf1_list_list:
        print(x)

    for x in num_gt_tracks_list_list:
        print(x)
