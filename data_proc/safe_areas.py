import numpy as np
from datetime import date, timedelta

def iou_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class Area:
    def __init__(self, year, month, day, type, l_x=0, r_x=28800, t_y=80, b_y=10):
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)
        self.type = type
        self.l_x = l_x
        self.r_x = r_x
        self.t_y = t_y
        self.b_y = b_y
        self.annots = np.empty(shape=(0, 8))

    def set_offset(self, offset):
        self.annots[:, :2] = self.annots[:, :2] + offset

    def add_valid_annot(self, annots, with_height=True, allow_partial=False):
        if annots.size == 0:
            return 0, np.array([])

        if allow_partial:
            args = np.bitwise_or(np.bitwise_or(np.bitwise_and(annots[:, 0] > self.l_x, annots[:, 1] < self.r_x),
                                               np.bitwise_and(annots[:, 0] < self.l_x, annots[:, 1] > self.l_x)),
                                 np.bitwise_and(annots[:, 0] < self.r_x, annots[:, 1] > self.r_x))

        else:
            args = np.bitwise_and(annots[:, 0] > self.l_x, annots[:, 1] < self.r_x)

        if with_height:
            args = np.bitwise_and(args,
                                  np.bitwise_and(annots[:, 2] > self.b_y, annots[:, 3] < self.t_y))

        self.annots = annots[args, :]

        return self.annots.shape[0], args

    def gen_perturbed_positives(self, nb_per_pos, perturbations):
        if self.annots.size == 0:
            return np.empty(shape=(0, 8))

        positives = np.repeat(self.annots, nb_per_pos, axis=0)
        perturb_x = perturbations['x']
        perturb_y = perturbations['y']

        offsets_x = (positives[:, 1] - positives[:, 0]) * perturb_x
        offsets_y = (positives[:, 3] - positives[:, 2]) * perturb_y

        offsets = np.array([-offsets_x, offsets_x, -offsets_y, offsets_y]).transpose() + positives[:, :4]
        inner_offsets = np.array([offsets_x, -offsets_x, offsets_y, -offsets_y]).transpose() + positives[:, :4]
        #apply safe area
        offsets = np.expand_dims(offsets, axis=2)
        safe = np.expand_dims(np.repeat(np.array([[self.l_x, self.r_x, self.b_y, self.t_y]]), positives.shape[0], axis=0), axis=2)

        limits = np.concatenate((offsets, safe), axis=2)
        limits = np.stack(([np.max(limits[:, 0], axis=1), inner_offsets[:, 0]],
                           [inner_offsets[:, 1], np.min(limits[:, 1], axis=1)],
                           [np.max(limits[:, 2], axis=1), inner_offsets[:, 2]],
                           [inner_offsets[:, 3], np.min(limits[:, 3], axis=1)])).transpose()

        dists_available = limits[:, 1, :] - limits[:, 0, :]

        perturbed_pos = np.random.random(dists_available.shape) * dists_available + limits[:, 0, :]
        positives[:, :4] = perturbed_pos
        return positives

    def gen_negatives(self, nb, neg_parameters):

        negatives = np.empty(shape=(0, 8))
        height_min = neg_parameters['height_min']
        width_min = neg_parameters['width_min']
        for i in range(nb):
            # Skip otherwise ? Safe area too small
            if self.r_x - self.l_x < neg_parameters['width_min']:
                print('Warning : width min adjusted for {}/{}/{} type : {}'.format(self.year,
                                                    self.month,
                                                    self.day,
                                                    self.type))
                width_min = self.r_x - self.l_x
                if width_min - neg_parameters['width_random'] < 0:
                    return negatives

            if self.t_y - self.b_y < neg_parameters['height_min']:
                print('Warning : height min adjusted for {}/{}/{} type : {}'.format(self.year,
                                                    self.month,
                                                    self.day,
                                                    self.type))
                height_min = self.t_y - self.b_y
                if height_min - neg_parameters['height_random'] < 0:
                    return negatives
                #height_mean = neg_parameters['height_mean'] - 1
                #if height_mean - neg_parameters['height_random'] <= 11:
                #    return negatives

            height_random_range = np.max([np.min([(self.t_y - self.b_y)
                                                  - height_min, neg_parameters['height_random'] / 2]), 0]) + neg_parameters['height_random'] / 2

            width_random_range = np.max([np.min([(self.r_x - self.l_x)
                                                 - width_min, neg_parameters['width_random'] / 2]), 0]) + neg_parameters['width_random'] / 2

            for _ in range(1000):
                height = np.random.random() * height_random_range - neg_parameters['height_random'] / 2
                width = np.random.random() * width_random_range - neg_parameters['width_random'] / 2
                width = width + width_min
                height = height + height_min

                l_x = self.l_x + np.random.random() * (self.r_x - self.l_x - width)
                b_y = self.b_y + np.random.random() * (self.t_y - self.b_y - height)

                overlap = False
                for annot in self.annots:
                    #if iou_area([l_x, b_y, l_x + width, b_y + height], annot[:4]) > 0.5:
                    #check overlap only on x
                    if iou_area([l_x, 10.0, l_x + width, 80.0],
                                [annot[0],  10.0, annot[1], 80.0]) > 0.3:
                        overlap = True
                        break
                if not overlap:
                    negatives = np.concatenate([negatives, np.array([[l_x, l_x + width, b_y, b_y + height,
                                                                    0, self.year, self.month, self.day]])], axis=0)
                    break

        return negatives

    def get_positives(self):
        return self.annots

def get_data_in_areas(all_data, sdate, edate, samples_to_exclude_3 = np.empty(shape=(0,8)),
                                              samples_to_exclude_4 = np.empty(shape=(0, 8))):
    delta = edate - sdate
    def create_areas_2_4(all_data, type):
        areas = []
        total_stored = 0
        samples = all_data[all_data[:, 4] == type]
        for i in range(delta.days + 1):
            day_date = sdate + timedelta(days=i)
            np_area = np.array([day_date.year, day_date.month, day_date.day])
            samples_for_the_day = samples[(samples[:, 5:8] == np_area).all(axis=1), :]
            area = Area(day_date.year, day_date.month, day_date.day, type)

            if type == 4:
                args_to_check_to_exclude = (samples_to_exclude_4[:, 5:8] == np_area).all(axis=1)
                samples_to_exclude_for_the_day = samples_to_exclude_4[args_to_check_to_exclude, :]
                if samples_to_exclude_for_the_day.shape[0] > 0:
                    valid_samples_added, args = area.add_valid_annot(samples_to_exclude_for_the_day,
                                                                     with_height=False)
                    # skip this safe area
                    if valid_samples_added > 0:
                        continue

            annots_stored, _ = area.add_valid_annot(samples_for_the_day)
            if annots_stored != samples_for_the_day.shape[0]:
                area.add_valid_annot(samples_for_the_day)
            total_stored += annots_stored
            areas.append(area)
        #assert(total_stored == samples.shape[0])
        return areas, total_stored, samples.shape[0]

    areas_type_2, nb_samples_2, nb_annotated_2 = create_areas_2_4(all_data, 2)
    areas_type_4, nb_samples_4, nb_annotated_4 = create_areas_2_4(all_data, 4)

    areas_type_3 = []
    safe_areas = all_data[all_data[:, 4] == 5]
    samples = all_data[all_data[:, 4] == 3]
    nb_samples_3 = 0
    args_found = np.empty(shape=(0, 1))
    for safe_area in safe_areas:
        date_area = date(int(safe_area[5]), int(safe_area[6]), int(safe_area[7]))
        if date_area < sdate or date_area > edate:
            continue
        args_to_check = (samples[:, 5:8] == safe_area[5:8]).all(axis=1)
        samples_for_the_day = samples[args_to_check, :]
        area_type_3 = Area(safe_area[5], safe_area[6], safe_area[7], 3, l_x=int(safe_area[0]), r_x=int(safe_area[1]), b_y=int(safe_area[2]), t_y=int(safe_area[3]))

        args_to_check_to_exclude = (samples_to_exclude_3[:, 5:8] == safe_area[5:8]).all(axis=1)
        samples_to_exclude_for_the_day = samples_to_exclude_3[args_to_check_to_exclude, :]
        if samples_to_exclude_for_the_day.shape[0] > 0:
            valid_samples_added, args = area_type_3.add_valid_annot(samples_to_exclude_for_the_day, with_height=False)
            #skip this safe area
            if valid_samples_added > 0:
                continue

        valid_samples_added, args = area_type_3.add_valid_annot(samples_for_the_day, with_height=False)
        if len(args) == 0:
            continue
        args_found = np.concatenate([args_found, np.argwhere(args_to_check)[args]])
        nb_samples_3 += valid_samples_added
        areas_type_3.append(area_type_3)
    args_found = args_found.astype(dtype=np.int)
    samples_not_in_safe = samples[np.bitwise_not((args_found == np.arange(samples.shape[0]).reshape(1, -1)).any(axis=0)), :]
    print(samples_not_in_safe.astype(dtype=np.int))

    print("Type 2 : {} stored, {} annotated".format(nb_samples_2, nb_annotated_2))
    print("Type 3 : {} stored, {} annotated".format(nb_samples_3, samples.shape[0]))
    print("Type 4 : {} stored, {} annotated".format(nb_samples_4, nb_annotated_4))

    return areas_type_2, areas_type_3, areas_type_4, nb_samples_2, nb_samples_3, nb_samples_4
