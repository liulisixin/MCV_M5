from tqdm import tqdm
import numpy as np
import cv2
import imageio
import motmetrics as mm


def center_of_detection(detection):
    return (int(detection['left'] + 0.5 * detection['width']), int(detection['top'] + 0.5 * detection['height']))


def write_one_frame(detections_tracks, frame_id, frameMat, color):
    """
    tool for addTracksToFrames
    :param detections_tracks: this can be detections or ground truth
    :param frame_id: which frame
    :param frameMat: frame picture
    :param color: rectangle and line color
    :return: frameMat
    """
    for track_one in detections_tracks:
        index = 0
        flag_shoot = False
        for index, detection in enumerate(track_one.detections):
            # write the rectangle
            if detection['frame'] == frame_id:
                startPoint = (int(detection['left']), int(detection['top']))
                endPoint = (int(startPoint[0] + detection['width']), int(startPoint[1] + detection['height']))
                frameMat = cv2.rectangle(frameMat, startPoint, endPoint, color, 2)
                flag_shoot = True
                break
        if flag_shoot:
            shoot_index = index
            # write the line
            for index in range(shoot_index):
                startPoint = center_of_detection(track_one.detections[index])
                endPoint = center_of_detection(track_one.detections[index + 1])
                frameMat = cv2.line(frameMat, startPoint, endPoint, color, 2)
    return frameMat


def addTracksToFrames(framesPath, detections_tracks, tracks_gt_list, start_frame = 1, end_frame = 2141, name = "test"):
    """
    write the video of the tracking result in the format .avi
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    # size = (1920, 1080)
    filename = "{}/{}.png".format(framesPath, str(start_frame).zfill(6))
    frame_size = cv2.imread(filename).shape
    size = (frame_size[1], frame_size[0])
    fps = 10
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for frame_id in tqdm(range(start_frame, end_frame)):
        filename = "{}/{}.png".format(framesPath, str(frame_id).zfill(6))
        frameMat = cv2.imread(filename)
        color_detection = (0, 0, 255)
        write_one_frame(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        write_one_frame(tracks_gt_list, frame_id, frameMat, color_gt)
        out.write(frameMat)
    out.release()


def addTracksToFrames_gif(framesPath, detections_tracks, tracks_gt_list, start_frame = 1, end_frame = 10, name = "test"):
    """
    write the gif of the tracking result
    :param framesPath: path of frames
    :param detections_tracks: detections in list of Track. Detections in Track should be sorted.
    :param tracks_gt_list: ground truth in list of Track. Detections in Track should be sorted.
    :param name: name of video.
    :return: None
    """
    scale = 0.5
    filename = "{}/{}.png".format(framesPath, str(start_frame).zfill(6))
    frame_size = cv2.imread(filename).shape
    size = (int(scale*frame_size[1]), int(scale*frame_size[0]))

    images = []

    for frame_id in tqdm(range(start_frame, end_frame)):
        filename = "{}/{}.png".format(framesPath, str(frame_id).zfill(6))
        frameMat = cv2.imread(filename)
        color_detection = (0, 0, 255)
        write_one_frame(detections_tracks, frame_id, frameMat, color_detection)
        color_gt = (255, 0, 0)
        write_one_frame(tracks_gt_list, frame_id, frameMat, color_gt)
        resized = cv2.resize(frameMat, size, interpolation=cv2.INTER_AREA)
        images.append(resized)
        cv2.imwrite("{}.png".format(str(frame_id)), frameMat)
    # imageio.mimsave(name + '.gif', images, duration=0.5)
    imageio.mimsave(name + '.gif', images)


def find_frame_in_track(tracks, frame_id):
    object_id_list = []
    box_list = []
    for track_one in tracks:
        for index, detection in enumerate(track_one.detections):
            # write the rectangle
            if detection['frame'] == frame_id:
                object_id_list.append(track_one.id)
                box_list.append([detection['left'], detection['top'], detection['width'], detection['height']])
                break
    return object_id_list, box_list

def find_frame_in_gt(gt, frame_id):
    object_id_list = []
    box_list = []
    for instance in gt:
        if instance['frame'] == frame_id:
            object_id_list.append(instance['ID'])
            box_list.append([instance['left'], instance['top'], instance['width'], instance['height']])
    return object_id_list, box_list


def calculate_idf1(gt, detections_tracks, video_length, IoU_threshold=0.5, verbose = False):
    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(video_length):
        frame_id = i + 1
        gt_ids, gt_bboxes = find_frame_in_gt(gt, frame_id)
        detections_ids, detections_bboxes = find_frame_in_track(detections_tracks, frame_id)

        distances_gt_det = mm.distances.iou_matrix(gt_bboxes, detections_bboxes, max_iou=1.)
        acc.update(gt_ids, detections_ids, distances_gt_det)

    # print(acc.mot_events)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    print(summary)

    return summary.idf1.acc


def tracking_filter(detections_tracks):
    detections_tracks_filtered = []
    for track_one in detections_tracks:
        detection_first = track_one.detections[0]
        X_1 = detection_first['left'] + 0.5*detection_first['width']
        Y_1 = detection_first['top'] + 0.5*detection_first['height']
        detection_last = track_one.detections[-1]
        X_2 = detection_last['left'] + 0.5 * detection_last['width']
        Y_2 = detection_last['top'] + 0.5 * detection_last['height']
        vec1 = np.array([X_1, Y_1])
        vec2 = np.array([X_2, Y_2])
        if (detection_last['frame']-detection_first['frame']) > 10:
            if np.linalg.norm(vec1-vec2) > 100:
                detections_tracks_filtered.append(track_one)
    return detections_tracks_filtered

