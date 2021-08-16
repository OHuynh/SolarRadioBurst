import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from data_proc.read_utils import *
from data_proc.preprocess import remove_artifactsC
from data_proc.plot import plot_bursts
from data_proc.safe_areas import get_data_in_areas
import random
import math
import os

perturb_positives_per_sample = {2:3, #Positives generated for each sample
                                3:1,
                                4:3}

perturbations = {2:{'x':0.2, 'y':0.2},#offset
                 3:{'x':0.2, 'y':0.2},
                 4:{'x':0.2, 'y':0.2}}

neg_parameters = {2: {'width_min': 863,
                      'width_random': 475,
                      'height_min': 44,
                      'height_random': 12},
                  3: {'width_min': 60,
                      'width_random': 29,
                      'height_min': 54,
                      'height_random': 10},
                  4: {'width_min': 10708,
                      'width_random': 8130,
                      'height_min': 46,
                      'height_random': 10}}


all_data = read_annotations(path='../Solar_Interface/Image_Filters/Read_Data/Data/Annotation.txt')

years_available = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
sdate = date(2011, 1, 1)
edate = date(2020, 12, 31)
delta = edate - sdate
areas_type_2, areas_type_3, areas_type_4, _, _, _ = get_data_in_areas(all_data, sdate, edate)

# load a subset of areas (for example train test from tf record generation)
areas_type_2 = get_subset_areas_array(areas_type_2, "../data_proc/dataset_type_2_test_areas.txt")
areas_type_3 = get_subset_areas_array(areas_type_3, "../data_proc/dataset_type_3_test_areas.txt")
areas_type_4 = get_subset_areas_array(areas_type_4, "../data_proc/dataset_type_4_test_areas.txt")

# Shuffle areas
random.shuffle(areas_type_2)
random.shuffle(areas_type_3)
random.shuffle(areas_type_4)

def generate_positives(type, areas): #s'assurer que la pertubation ne dépasse pas de la zone safe
    positives = np.empty(shape=(0, 8))
    for area in areas:
        positives = np.concatenate([positives, area.gen_perturbed_positives(perturb_positives_per_sample[type],
                                                             perturbations[type])], axis=0)
        positives = np.concatenate([positives, area.get_positives()], axis=0)
    return positives

pos_2 = generate_positives(2, areas_type_2)
pos_3 = generate_positives(3, areas_type_3)
pos_4 = generate_positives(4, areas_type_4)

def generate_negatives(areas, negs, neg_parameters):
    negs_per_area = int(math.ceil(max(negs / len(areas) , 1)))
    idx_area = 0
    negatives = np.empty(shape=(0, 8))
    while True:
        area = areas[idx_area]
        negatives = np.concatenate([negatives, area.gen_negatives(negs_per_area, neg_parameters)], axis=0)
        if negatives.shape[0] >= negs:
            break
        idx_area = idx_area + 1
        if idx_area >= len(areas) - 1:
            idx_area = 0
    return negatives
"""
neg_2 = generate_negatives(areas_type_2, 1, neg_parameters[2])
neg_3 = generate_negatives(areas_type_3, 1, neg_parameters[3])
neg_4 = generate_negatives(areas_type_4, 1, neg_parameters[4])
"""

neg_2 = generate_negatives(areas_type_2, pos_2.shape[0], neg_parameters[2])
neg_3 = generate_negatives(areas_type_3, pos_3.shape[0], neg_parameters[3])
neg_4 = generate_negatives(areas_type_4, pos_4.shape[0], neg_parameters[4])


show_samples = True
def write_annots(pos, neg, file):
    fileW = open(file, "w")
    for i in range(delta.days + 1):
        d_date = sdate + timedelta(days=i)
        day_date = [float(d_date.year), float(d_date.month), float(d_date.day)]
        samples = np.concatenate([pos[np.all(pos[:, 5:] == day_date, axis=1)],
                                  neg[np.all(neg[:, 5:] == day_date, axis=1)]])

        if samples.shape[0] > 0:
            samples = np.around(samples, decimals=2)
            csv_line = '{},{:02d},{:02d};'.format(d_date.year, d_date.month, d_date.day)
            csv_line = csv_line + ';'.join([','.join(list(sample[:5].astype(np.str))) for sample in list(samples)]) + '\n'
            if show_samples:
                filename = os.path.join('../data/', str(d_date.year) + '%02d' % d_date.month)
                extension = "S" + str(d_date.year)[2:4] + '%02d' % d_date.month + '%02d' % d_date.day + ".RT1"
                filename = os.path.join(filename, extension)
                if not os.path.exists(filename):
                    continue
                img, _, _, _, _ = read_data(filename)
                filtered_img = remove_artifactsC(img)
                plot_bursts(filtered_img, samples, '{:02d}/{:02d}/{}'.format(d_date.day, d_date.month, d_date.year))
            fileW.write(csv_line)

write_annots(pos_2, neg_2, 'annotation_2.txt')
write_annots(pos_3, neg_3, 'annotation_3.txt')
write_annots(pos_4, neg_4, 'annotation_4.txt')
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from data_proc.read_utils import *
from data_proc.preprocess import remove_artifactsC
from data_proc.plot import plot_bursts
from data_proc.safe_areas import get_data_in_areas
import random
import math
import os

perturb_positives_per_sample = {2:3, #Positives generated for each sample
                                3:1,
                                4:3}

perturbations = {2:{'x':0.2, 'y':0.2},#offset
                 3:{'x':0.2, 'y':0.2},
                 4:{'x':0.2, 'y':0.2}}

neg_parameters = {2: {'width_min': 863,
                      'width_random': 475,
                      'height_min': 44,
                      'height_random': 12},
                  3: {'width_min': 60,
                      'width_random': 29,
                      'height_min': 54,
                      'height_random': 10},
                  4: {'width_min': 10708,
                      'width_random': 8130,
                      'height_min': 46,
                      'height_random': 10}}


all_data = read_annotations(path='../Solar_Interface/Image_Filters/Read_Data/Data/Annotation.txt')

years_available = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
sdate = date(2011, 1, 1)
edate = date(2020, 12, 31)
delta = edate - sdate
areas_type_2, areas_type_3, areas_type_4, _, _, _ = get_data_in_areas(all_data, sdate, edate)

# load a subset of areas (for example train test from tf record generation)
areas_type_2 = get_subset_areas_array(areas_type_2, "../data_proc/dataset_type_2_test_areas.txt")
areas_type_3 = get_subset_areas_array(areas_type_3, "../data_proc/dataset_type_3_test_areas.txt")
areas_type_4 = get_subset_areas_array(areas_type_4, "../data_proc/dataset_type_4_test_areas.txt")

# Shuffle areas
random.shuffle(areas_type_2)
random.shuffle(areas_type_3)
random.shuffle(areas_type_4)

def generate_positives(type, areas): #s'assurer que la pertubation ne dépasse pas de la zone safe
    positives = np.empty(shape=(0, 8))
    for area in areas:
        positives = np.concatenate([positives, area.gen_perturbed_positives(perturb_positives_per_sample[type],
                                                             perturbations[type])], axis=0)
        positives = np.concatenate([positives, area.get_positives()], axis=0)
    return positives

pos_2 = generate_positives(2, areas_type_2)
pos_3 = generate_positives(3, areas_type_3)
pos_4 = generate_positives(4, areas_type_4)

def generate_negatives(areas, negs, neg_parameters):
    negs_per_area = int(math.ceil(max(negs / len(areas) , 1)))
    idx_area = 0
    negatives = np.empty(shape=(0, 8))
    while True:
        area = areas[idx_area]
        negatives = np.concatenate([negatives, area.gen_negatives(negs_per_area, neg_parameters)], axis=0)
        if negatives.shape[0] >= negs:
            break
        idx_area = idx_area + 1
        if idx_area >= len(areas) - 1:
            idx_area = 0
    return negatives
"""
neg_2 = generate_negatives(areas_type_2, 1, neg_parameters[2])
neg_3 = generate_negatives(areas_type_3, 1, neg_parameters[3])
neg_4 = generate_negatives(areas_type_4, 1, neg_parameters[4])
"""

neg_2 = generate_negatives(areas_type_2, pos_2.shape[0], neg_parameters[2])
neg_3 = generate_negatives(areas_type_3, pos_3.shape[0], neg_parameters[3])
neg_4 = generate_negatives(areas_type_4, pos_4.shape[0], neg_parameters[4])


show_samples = True
def write_annots(pos, neg, file):
    fileW = open(file, "w")
    for i in range(delta.days + 1):
        d_date = sdate + timedelta(days=i)
        day_date = [float(d_date.year), float(d_date.month), float(d_date.day)]
        samples = np.concatenate([pos[np.all(pos[:, 5:] == day_date, axis=1)],
                                  neg[np.all(neg[:, 5:] == day_date, axis=1)]])

        if samples.shape[0] > 0:
            samples = np.around(samples, decimals=2)
            csv_line = '{},{:02d},{:02d};'.format(d_date.year, d_date.month, d_date.day)
            csv_line = csv_line + ';'.join([','.join(list(sample[:5].astype(np.str))) for sample in list(samples)]) + '\n'
            if show_samples:
                filename = os.path.join('../data/', str(d_date.year) + '%02d' % d_date.month)
                extension = "S" + str(d_date.year)[2:4] + '%02d' % d_date.month + '%02d' % d_date.day + ".RT1"
                filename = os.path.join(filename, extension)
                if not os.path.exists(filename):
                    continue
                img, _, _, _, _ = read_data(filename)
                filtered_img = remove_artifactsC(img)
                plot_bursts(filtered_img, samples, '{:02d}/{:02d}/{}'.format(d_date.day, d_date.month, d_date.year))
            fileW.write(csv_line)

write_annots(pos_2, neg_2, 'annotation_2.txt')
write_annots(pos_3, neg_3, 'annotation_3.txt')
write_annots(pos_4, neg_4, 'annotation_4.txt')
