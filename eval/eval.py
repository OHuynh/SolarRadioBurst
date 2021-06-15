# coding: utf-8


import tensorflow as tf
import cv2
import os
from data_proc.read_utils import *
from data_proc.preprocess import remove_artifactsC
from data_proc.plot import plot_bursts
from criteria import *

show_detections_cnn = True
show_detections_whole_day = True
test_type = 4

thresholds_for_curve = np.arange(0.5, 1.0, 0.05)
params_detect = {2: {'window': 2700,
                     'stride': 1350, #ballayage, image qu'on reformate pour fit le modèle
                     'width_px': 320,
                     'height_px': 160,
                     'near_distance_merge': 300, #merge depuis les boites trouvées
                     'nms_iou_threshold': 0.1, #idem
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

def nms_bbox_threshold(areas, threshold, iou_threshold):
    for idx in range(len(areas)):
        boxes = areas[idx]['detections']
        boxes = boxes[boxes[:, 4] >= threshold]
        selected_indices = tf.image.non_max_suppression(boxes[:,:4], boxes[:, 4], 100,
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
                #minimal time between two types 2
                if boxes[other_idx, 0] - boxes[idx, 1] < distance and boxes[other_idx, 0] > boxes[idx, 1]:
                    boxes[idx, 1] = boxes[other_idx, 1] # merge and extend
                    boxes = np.delete(boxes, other_idx, axis=0)
                    idx = idx - 1
                    break
            idx += 1
        areas[idx]['detections'] = boxes
    return areas

def cnn_detect_with_slide(img, area, model, type, params_detect, base_threshold=0.5, show_detection=False, show_whole_area=False): #tous les params qui balayent la journée
    if img.shape[1] < area.r_x:
        area.r_x = img.shape[1]
    boxes = np.empty(shape=(0, 6))
    idx_img = 0
    for i in range(area.l_x, area.r_x, params_detect['stride']):
        img_windowed = np.zeros(shape=(img.shape[0], params_detect['window']))

        right_x = np.min([i + params_detect['window'], np.min([area.r_x, img.shape[1]])]) - i

        img_windowed[:, :right_x] = img[:, i:i + right_x]

        img_windowed = cv2.resize(img_windowed, dsize=(params_detect['width_px'], params_detect['height_px']))
        img_to_show = np.repeat(np.expand_dims(img_windowed.copy().astype(dtype=np.uint8), axis=2), 3, axis=2)
        img_windowed = np.reshape(img_windowed, (1, params_detect['height_px'], params_detect['width_px'], 1))
        detections = model(img_windowed)
        if show_detection: #ssd prend en entrée une image et en sortie plein de boites. Ces boites sont données dans un tableau, à des ancres
            for positive in area.get_positives():#image_windowed : 1 batch, 160,320 (taille conventionnée), 1 couleur
                x_min = int((positive[0] - i) / params_detect['window'] * params_detect['width_px'])
                x_max = int((positive[1] - i) / params_detect['window'] * params_detect['width_px'])
                y_min = int((positive[2] - 10.0) / 70.0 * params_detect['height_px'])
                y_max = int((positive[3] - 10.0) / 70.0 * params_detect['height_px'])
                tl_img = (np.min([np.max([x_min, 0]), params_detect['width_px'] - 1]), y_min)
                br_img = (np.min([np.max([x_max, 0]), params_detect['width_px'] - 1]), y_max)
                #if tl_img[0] != br_img[0]:
                #    cv2.rectangle(img_to_show, tl_img, br_img, (0, 0, 255), 1)
                print("{}/{}/{}".format(area.year, area.month, area.day))
        bboxes = detections['detection_boxes'][0].numpy() #travailler avec detection_boxes, detection_classes, detection_scores #si c'est supérieur à un certain threshold, c'est détecté
        bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
        bscores = detections['detection_scores'][0].numpy()
        #print(bscores)
        FP = False
        for idx in range(len(bboxes)):
            if bclasses[idx] == 1:
                if bscores[idx] >= base_threshold: #on resize ce qui est détecté, et on dessine le rectangle sur l'image to show
                    print(bscores[idx])
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
                             type]])], axis=0)
                    if area.get_positives().shape[0] == 0:
                        #FP = True
                        FP = False
                    if show_detection or FP:
                        cv2.rectangle(img_to_show, (x_min, y_min),
                                      (x_max, y_max), (0, 255, 0), 1)
        if show_detection or FP:
            idx_img += 1
            #cv2.imwrite('G:/Projets/SolarBurst/testing/plot/{}.png'.format(idx_img), img_to_show)
            cv2.imshow('detection', cv2.resize(img_to_show, None, fx=4.0, fy=4.0))
            cv2.waitKey(-1)

        if area.l_x + params_detect['window'] > area.r_x:
            break
    if show_whole_area: #merge sur toute la journée
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
        plot_bursts(img,
                    boxes_,
                    '{:02d}/{:02d}/{}'.format(area.day, area.month, area.year))

    return boxes


def append_detections_in_tests_areas_CNN(type, test_areas, params_detect):
    print('Loading model...')
    model = tf.saved_model.load(
        "C:/Users/zacch/Desktop/Stage_Orlean/SolarRadioBurst/bucket/output/exported{}/saved_model".format(type),
        tags=None,
        options=None
    )
    print('Model loaded !')
    for idx in range(len(test_areas)): #il va chercher le fichier RT1
        test_area = test_areas[idx]['area']
        filename = os.path.join('C:/Users/zacch/Desktop/Stage_Orlean/SolarRadioBurst/Solar_Interface/Image_Filters/Read_Data/Data',
                                str(test_area.year) + '%02d' % test_area.month)
        extension = "S" + str(test_area.year)[2:4] + '%02d' % test_area.month + '%02d' % test_area.day + ".RT1"
        filename = os.path.join(filename, extension)
        if not os.path.exists(filename):
            continue
        img, _, _, _, _ = read_data(filename)
        filtered_img = remove_artifactsC(img)

        test_areas[idx]['detections'] = np.concatenate([test_areas[idx]['detections'], #on stocke les résulstats de la détection
                                                       cnn_detect_with_slide(filtered_img,
                                                                             test_area,
                                                                             model,
                                                                             type,
                                                                             params_detect[type],
                                                                             show_detection=show_detections_cnn,
                                                                             show_whole_area=show_detections_whole_day)],
                                                       axis=0)
        print('{}/{} tests for type {}'.format(idx, len(test_areas), type))
        #if idx > 5:
        #    break
    return test_areas

def append_detections_in_tests_areas_from_file(type, test_areas):

    samples = read_annotations(path='../object-detection/results_from_matlab/Type{}/1/output.txt'.format(type), dim=9)
    for idx in range(len(test_areas)):
        area = test_areas[idx]['area']
        np_area = np.array([area.year, area.month, area.day])
        samples_for_the_day = samples[(samples[:, 5:8] == np_area).all(axis=1), :]
        tmp_area = Area(area.year, area.month, area.day, 0, l_x=area.l_x, r_x=area.r_x,b_y=area.b_y, t_y=area.t_y)
        tmp_area.add_valid_annot(samples_for_the_day, with_height=False)
        for positive in tmp_area.get_positives():
            test_areas[idx]['detections'] = np.concatenate([test_areas[idx]['detections'],
                                                            np.array([[int(positive[2]), #ymin
                                                                       int(positive[0]), #xmin
                                                                       int(positive[3]), #ymax
                                                                       int(positive[1]), #xmax
                                                                       positive[4],
                                                                       type]])
                                                            ], axis=0)
    return test_areas

def collect_results(type, areas, params_detect):
    #keep only areas in test_data
    test_areas = get_subset_areas_dict(areas[type],
                                       "../data_proc/dataset_type_{}_test_areas.txt".format(type, type))
    CNN_MODE = True
    if CNN_MODE:
        test_areas = append_detections_in_tests_areas_CNN(type, test_areas, params_detect)
    else:
        test_areas = append_detections_in_tests_areas_from_file(type, test_areas)

    recall = []
    precision = []
    sensitivity = []
    specificity = []
    false_positive_rate = []
    threshold_overlap = 0.3
    for threshold in thresholds_for_curve:
        test_areas_merged = []
        for test_area in test_areas:
            test_areas_merged.append({'area':test_area['area'], 'detections': test_area['detections'].copy()})
        test_areas_merged = nms_bbox_threshold(test_areas_merged, threshold, params_detect[type]['nms_iou_threshold'])
        #test_areas_merged = merge_near_boxes(test_areas_merged, params_detect[type]['near_distance_merge'])

        show = False
        if show:
            for test_area_merged in test_areas_merged:
                filename = os.path.join('../data/', str(test_area_merged['area'].year) + '%02d' % test_area_merged['area'].month)
                extension = "S" + str(test_area_merged['area'].year)[2:4] + '%02d' % test_area_merged['area'].month + '%02d' % test_area_merged['area'].day + ".RT1"
                filename = os.path.join(filename, extension)
                if not os.path.exists(filename):
                    continue
                img, _, _, _, _ = read_data(filename)
                filtered_img = remove_artifactsC(img)
                plot_bursts(filtered_img, test_area_merged['area'].get_positives(), '{:02d}/{:02d}/{}'.format(test_area_merged['area'].day, test_area_merged['area'].month, test_area_merged['area'].year))

                plot_bursts(filtered_img, test_area_merged['detections'], '{:02d}/{:02d}/{}'.format(test_area_merged['area'].day, test_area_merged['area'].month, test_area_merged['area'].year))

        recall_, precision_, positives_ = eval_recall_precision(test_areas_merged, threshold_overlap)
        print('recall : {:.5f} precision : {:.5f} positives : {} test areas : {}'.format(recall_,
                                                                                         precision_,
                                                                                         positives_,
                                                                                         len(test_areas)))
        sensitivity_, specificity_, false_positive_rate_, total_time_ = eval_roc(test_areas_merged, threshold_overlap)
        print('sensitivity : {:.5f} specificity : {:.5f} false_positive_rate : {:.5f} threshold : {:.2f} total_time : {}'
              .format(sensitivity_,
                      specificity_,
                      false_positive_rate_,
                      threshold,
                      total_time_))
        recall.append(recall_)
        precision.append(precision_)
        sensitivity.append(sensitivity_)
        specificity.append(specificity_)
        false_positive_rate.append(false_positive_rate_)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(precision, recall)

    ax.set(xlabel='precision', ylabel='recall',
           title='Recall/Precision')
    ax.grid()

    fig.savefig("precision_recall.png")
    plt.show()

    return test_areas

def eval_in_test_areas():
    years_available = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    sdate = date(2011, 1, 1)
    edate = date(2020, 12, 31)
    delta = edate - sdate
    all_data = read_annotations(path='C:/Users/zacch/Desktop/Stage_Orlean/SolarRadioBurst/Solar_Interface/Image_Filters/Read_Data/Data/Annotation.txt')
    all_data[all_data[:, 4] == 6, 4] = 4

    areas_type_2, areas_type_3, areas_type_4, _, _, _ = get_data_in_areas(all_data, sdate, edate)
    areas = {2: areas_type_2,
             3: areas_type_3,
             4: areas_type_4}

    if test_type == 2:
        collect_results(type=2, areas=areas, params_detect=params_detect)
    elif test_type == 3:
        collect_results(type=3, areas=areas, params_detect=params_detect)
    else:
        collect_results(type=4, areas=areas, params_detect=params_detect)

def merge_detections(test_areas):
    for idx in range(len(test_areas)):
        area = test_areas[idx]
        #type 3 inside type 2/4 are removed
        type_3_indices = np.argwhere(area['detections'][:, 5] == 3)
        indices_to_delete = []
        for type_3_index in type_3_indices:
            for other_box in area['detections'][[area['detections'][:, 5] != 3], :]:
                box = area['detections'][type_3_index]
                if iou_area(box, other_box) > 0.5:
                    indices_to_delete.append(type_3_index)
                    break
        test_areas[idx]['detections'] = np.delete(area['detections'], indices_to_delete, axis=0)
        #type 4 inside type 2 are removed?
        type_4_indices = np.argwhere(area['detections'][:, 5] == 4)
        indices_to_delete = []
        for type_4_index in type_4_indices:
            for other_box in area['detections'][[area['detections'][:, 5] == 2], :]:
                box = area['detections'][type_4_index]
                if iou_area(box, other_box) > 0.5:
                    indices_to_delete.append(type_4_index)
                    break
        test_areas[idx]['detections'] = np.delete(area['detections'], indices_to_delete, axis=0)
        # TODO merge neighbourhood detections?
    return test_areas


def eval_complete_days_with_merging():
    sdate = date(2012, 1, 1)
    edate = date(2012, 12, 31)
    all_data = read_annotations(path='../data/Annotation.txt')
    all_data[all_data[:, 4] == 6, 4] = 4
    areas_type_2, areas_type_3, areas_type_4, _, _, _ = get_data_in_areas(all_data, sdate, edate)

    delta = edate - sdate
    test_areas = []
    for i in range(delta.days + 1):
        day_date = sdate + timedelta(days=i)
        test_areas.append({'area': Area(day_date.year, day_date.month, day_date.day, 0),
                           'detections': np.empty(shape=(0, 6))})

    #test_areas = append_detections_in_tests_areas_CNN(2, test_areas, params_detect)
    #test_areas = append_detections_in_tests_areas_CNN(3, test_areas, params_detect[3])
    #test_areas = append_detections_in_tests_areas_CNN(4, test_areas, params_detect)

    #test_areas = merge_detections(test_areas)

    #test 2/3/4 only in safe areas
    #test_areas[]


def main():
    eval_in_test_areas()
    #eval_complete_days_with_merging()


if __name__ == "__main__":
    main()
