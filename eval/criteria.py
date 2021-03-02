from data_proc.safe_areas import *

def eval_recall_precision(test_areas, threshold_overlap=0.5):

    #TODO : how to deal with several detections for one positive ?
    false_positives = 0
    true_positives = 0
    positives = 0
    for test_area in test_areas:
        for detection in test_area['detections']:
            found = False
            for positive in test_area['area'].get_positives():
                if iou_area([detection[0], 10.0, detection[1], 80.0],
                            [positive[0], 10.0, positive[1], 80.0]) > threshold_overlap:
                    found = True
                    break
            if not found:
                false_positives += 1

        for positive in test_area['area'].get_positives():
            positives += 1
            found = False
            for detection in test_area['detections']:
                if iou_area([detection[0], 10.0, detection[1], 80.0],
                            [positive[0], 10.0, positive[1], 80.0]) > threshold_overlap:
                    found = True
                    break
            if found:
                true_positives += 1
    recall = float(true_positives) / positives
    precision = float(true_positives) / (true_positives + false_positives)
    return recall, precision, positives


def eval_roc(test_areas, threshold_overlap=0.5):
    # compute TP/FP/TN/FN along time axis
    # 0 : NA
    # 1 : TP
    # 2 : FP
    # 3 : TN
    # 4 : FN
    results = np.zeros(shape=(len(test_areas), 28800,), dtype=np.int)
    for idx, test_area in enumerate(test_areas):
        area = test_area['area']
        #TN
        results[idx, area.l_x:area.r_x] = 3
        #FN
        for positive in area.get_positives():
            results[idx, int(positive[0]):int(positive[1])] = 4
        #FP
        for detection in test_area['detections']:
            # consider false positives which don't overlap at all an event
            exclude_overlap_fp = True
            if exclude_overlap_fp:
                found = False
                for positive in test_area['area'].get_positives():
                    if iou_area([detection[0], 10.0, detection[1], 80.0],
                                [positive[0], 10.0, positive[1], 80.0]) > threshold_overlap:
                        found = True
                        break
                if not found:
                    results[idx, int(detection[0]):int(detection[1])] = 2
            else:
                results[idx, int(detection[0]):int(detection[1])] = 2
        #TP
        for detection in test_area['detections']:
            for positive in area.get_positives():
                results[idx, int(np.max([detection[0], positive[0]])):int(np.min([detection[1], positive[1]]))] = 1

    # calculate rates
    sensitivity = (results == 1).sum() / ((results == 1).sum() + (results == 4).sum())
    specificity = (results == 3).sum() / ((results == 3).sum() + (results == 2).sum())
    false_positive_rate = (results == 2).sum() / ((results == 2).sum() + (results == 3).sum())

    return sensitivity, specificity, false_positive_rate, (results != 0).sum()
