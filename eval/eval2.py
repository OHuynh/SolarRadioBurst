# coding: utf-8


import tensorflow as tf
import cv2
import os
import datetime
import argparse
from pathlib import Path
from data_proc.read_utils import *
from data_proc.preprocess import remove_artifactsC
from data_proc.plot import plot_bursts
from criteria import *

show_detections_cnn = True
show_detections_whole_day = True

thresholds_for_curve = np.arange(0.5, 1.0, 0.05)
params_detect = {2: {'window': 2700,
                     'stride': 1350,  # ballayage, image qu'on reformate pour fit le modèle
                     'width_px': 320,
                     'height_px': 160,
                     'near_distance_merge': 300,  # merge depuis les boites trouvées
                     'nms_iou_threshold': 0.1,  # idem
                     'full_height': False},
                 3: {'window': 500,
                     'stride': 250,
                     'width_px': 480,
                     'height_px': 160,
                     'near_distance_merge': 0,
                     'nms_iou_threshold': 0.5,
                     'full_height': False},
                 4: {'window': 28800,
                     'stride': 28801,
                     'width_px': 320,
                     'height_px': 160,
                     'near_distance_merge': 1000,
                     'nms_iou_threshold': 0.1,
                     'full_height': False}}


class Area:
    def __init__(self, l_x=0, r_x=28800, t_y=80, b_y=10):
        self.l_x = l_x
        self.r_x = r_x
        self.t_y = t_y
        self.b_y = b_y


def nms_bbox_threshold(areas, threshold, iou_threshold):
    for idx in range(len(areas)):
        boxes = areas[idx]['detections']
        boxes = boxes[boxes[:, 4] >= threshold]
        selected_indices = tf.image.non_max_suppression(boxes[:, :4], boxes[:, 4], 100,
                                                        iou_threshold=iou_threshold)
        boxes = tf.gather(boxes, selected_indices).numpy()
        tmp = boxes.copy()
        boxes[:, 0] = tmp[:, 1]
        boxes[:, 1] = tmp[:, 3]
        boxes[:, 2] = tmp[:, 0]
        boxes[:, 3] = tmp[:, 2]
        areas[idx]['detections'] = boxes
    return areas


def merge_near_boxes(areas, distance=0):
    if distance == 0:
        return areas
    for idx in range(len(areas)):
        boxes = areas[idx]['detections']
        idx = 0
        while idx < len(boxes):
            for other_idx in range(len(boxes)):
                if idx == other_idx:
                    continue
                # minimal time between two types 2
                if boxes[other_idx, 0] - boxes[idx, 1] < distance and boxes[other_idx, 0] > boxes[idx, 1]:
                    boxes[idx, 1] = boxes[other_idx, 1]  # merge and extend
                    boxes = np.delete(boxes, other_idx, axis=0)
                    idx = idx - 1
                    break
            idx += 1
        areas[idx]['detections'] = boxes
    return areas


def conv_hours_to_sec(time):
    hours = 0
    minutes = 0
    seconds = 0
    exist = len(time)
    end = (str(time)).split(":")
    if exist >= 2:
        hours = int(end[0]) * 3600
    if exist >= 5:
        minutes = int(end[1]) * 60
    if exist >= 8:
        seconds = int(end[2])
    return hours + minutes + seconds


def cnn_detect_with_slide(img, area, model, type, params_detect, end_day, freq, base_threshold=0.5,
                          show_detection=False,
                          show_whole_area=True):  # tous les params qui balayent la journée
    burst_count = 0
    if img.shape[1] < area.r_x:
        area.r_x = img.shape[1]
    boxes = np.empty(shape=(0, 8))
    idx_img = 0
    mean_for_peak = []
    print("Burst type : ", type)
    for i in range(area.l_x, area.r_x, params_detect['stride']):
        img_windowed = np.zeros(shape=(img.shape[0], params_detect['window']))

        right_x = np.min([i + params_detect['window'], np.min([area.r_x, img.shape[1]])]) - i

        img_windowed[:, :right_x] = img[:, i:i + right_x]

        img_windowed = cv2.resize(img_windowed, dsize=(params_detect['width_px'], params_detect['height_px']))
        img_to_show = np.repeat(np.expand_dims(img_windowed.copy().astype(dtype=np.uint8), axis=2), 3, axis=2)
        img_windowed = np.reshape(img_windowed, (1, params_detect['height_px'], params_detect['width_px'], 1))
        detections = model(img_windowed)

        bboxes = detections['detection_boxes'][0].numpy()  # si c'est supérieur à un certain threshold, c'est détecté
        bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
        bscores = detections['detection_scores'][0].numpy()
        # print(bscores)
        FP = False
        for idx in range(len(bboxes)):
            if bclasses[idx] == 1:
                if bscores[
                    idx] >= base_threshold:  # on resize ce qui est détecté, et on dessine le rectangle sur l'image to show
                    y_min = int(bboxes[idx][0] * params_detect['height_px'])
                    x_min = int(bboxes[idx][1] * params_detect['width_px'])
                    y_max = int(bboxes[idx][2] * params_detect['height_px'])
                    x_max = int(bboxes[idx][3] * params_detect['width_px'])
                    boxes = np.concatenate([boxes,
                                            np.array([[int(bboxes[idx][0] * 70.0 + 10.0),
                                                       int(bboxes[idx][1] * params_detect['window']) + i,
                                                       int(bboxes[idx][2] * 70.0 + 10.0),
                                                       int(bboxes[idx][3] * params_detect['window']) + i,
                                                       float(bscores[idx]),
                                                       i,
                                                       x_min + i + end_day - area.r_x,
                                                       x_max - x_min]])], axis=0)
                    if show_detection or FP:
                        cv2.rectangle(img_to_show, (x_min, y_min),
                                      (x_max, y_max), (0, 255, 0), 1)

        if area.l_x + params_detect['window'] > area.r_x:
            break
    if show_whole_area:  # merge sur toute la journée
        boxes_ = boxes.copy()
        boxes_[:, 0] = 11.0
        boxes_[:, 2] = 79.0
        selected_indices = tf.image.non_max_suppression(boxes_[:, :4], boxes_[:, 4], 100, iou_threshold=0.1)
        boxes_ = tf.gather(boxes_, selected_indices).numpy()
        tmp = boxes_.copy()
        boxes_[:, 0] = tmp[:, 1]
        boxes_[:, 1] = tmp[:, 3]
        boxes_[:, 2] = tmp[:, 0]
        boxes_[:, 3] = tmp[:, 2]
        """
        idx = 0
        while idx < len(boxes_):
            for other_idx in range(len(boxes_)):
                if idx == other_idx:
                    continue
                #minimal time between two types 2
                if boxes_[other_idx, 0] - boxes_[idx, 1] < 300.0 and boxes_[other_idx, 0] > boxes_[idx, 1]:
                    boxes_[idx, 1] = boxes_[other_idx, 1] # merge and extend
                    boxes_ = np.delete(boxes_, other_idx, axis=0)
                    idx = idx - 1
                    break
            idx += 1
        """
        plot_bursts(img, boxes_)
    sortedboxes_ = boxes_[(boxes_[:, 6]).argsort()]
    for idx in range(len(sortedboxes_)):
        burst_count = burst_count + 1
        print("Burst Number : ", burst_count)
        print("Detection score :", sortedboxes_[idx, 4])
        print("Burst_start : ", str(timedelta(seconds=sortedboxes_[idx, 6])))
        print("Burst duration : ", sortedboxes_[idx, 7])

        mean_for_peak = np.sum([freq[int(sortedboxes_[idx, 6]), :]], axis=0) / len(freq)
        max_value = np.max(mean_for_peak)
        peak = np.where(mean_for_peak == max_value) % sortedboxes_[idx, 7]
        print("Burst peak : ", str(timedelta(seconds=sortedboxes_[idx, 6] + int(peak))))
    if burst_count == 0:
        print("No burst found for this type")
    return boxes


def append_detections_CNN():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_RT1', type=Path)
    parser.add_argument('--path_to_model_2', type=Path)
    parser.add_argument('--path_to_model_3', type=Path)
    parser.add_argument('--path_to_model_4', type=Path)
    args = parser.parse_args()
    img, _, _, _, end_of_day, freq = read_data(args.path_to_RT1)
    end_day = conv_hours_to_sec(end_of_day)
    filtered_img = remove_artifactsC(img)
    test_area_2 = Area()
    test_area_3 = Area()
    test_area_4 = Area()

    print('Loading models...')
    model_type_2 = tf.saved_model.load(str(args.path_to_model_2), tags=None, options=None)
    print('Model type 2 loaded !')
    model_type_3 = tf.saved_model.load(str(args.path_to_model_3), tags=None, options=None)
    print('Model type 3 loaded !')
    model_type_4 = tf.saved_model.load(str(args.path_to_model_4), tags=None, options=None)
    print('Model type 4 loaded !')

    cnn_detect_with_slide(filtered_img, test_area_2, model_type_2, 2, params_detect[2], end_day, freq,
                          show_detection=show_detections_cnn, show_whole_area=show_detections_whole_day)
    cnn_detect_with_slide(filtered_img, test_area_3, model_type_3, 3, params_detect[3], end_day, freq,
                          show_detection=show_detections_cnn, show_whole_area=show_detections_whole_day)
    cnn_detect_with_slide(filtered_img, test_area_4, model_type_4, 4, params_detect[4], end_day, freq,
                          show_detection=show_detections_cnn, show_whole_area=show_detections_whole_day)


def main():
    append_detections_CNN()
    # eval_complete_days_with_merging()


if __name__ == "__main__":
    main()
