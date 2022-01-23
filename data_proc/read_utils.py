import numpy as np

from datetime import datetime

def read_data(filename, rt1_before_2010=False):
    file_head = 405;
    spec_head_size = 5;
    spec_size = 400;

    f = open(filename, "rb")
    header = f.read(file_head)

    if rt1_before_2010:
        header = header.decode('utf-8')
        f_min = header[2:3]
        f_max = header[4:5]
        date_str = header[50:59]
        end_of_the_day = header[59:64]
        date = datetime.strptime(date_str.replace(' ', ''), '%d/%m/%y')
    else:
        end_of_the_day = None
    data = f.read()
    data = np.frombuffer(data, dtype=np.uint8).reshape(int(len(data) / (spec_head_size + spec_size)), -1)
    specs, _, data = np.split(data, np.array([4,4]), axis=1)
    specs_L, specs_R = specs[::2], specs[1::2]
    data_L, data_R = data[::2], data[1::2]

    data_L = data_L.copy().transpose()[:-1, :]
    data_R = data_R.copy().transpose()[:-1, :]

    f.close()
    return data_L, data_R, specs_L, specs_R, end_of_the_day

def read_annotations(path, dim=8):
    list_input_raw = open(path, "r")

    all_data = np.empty(shape=(0, dim))
    for line in list_input_raw:
        annots = line.replace("\n", "").split(';')
        year, month, day = list(map(int, annots[0].split(',')))
        all_data = np.concatenate([all_data, np.array([list(map(float, annot.split(','))) +
                                                       ([year, month, day] + (dim - 8) * [0])
                                                       for annot in annots[1:]]).reshape(-1, dim)])

    return all_data

# Todo change the way partial annots are stored to reuse the same scheme with annots
def read_partial_annotations(path):
    list_input_raw = open(path, "r")

    all_data = np.empty(shape=(0, 5))
    for line in list_input_raw:
        annots = line.replace("\n", "").split(',')
        all_data = np.concatenate([all_data, np.array([list(map(float, annots))]).astype(dtype=np.int)])

    return all_data

def get_subset_areas_array(areas, file):
    test_areas_raw = read_partial_annotations(file)
    test_areas = []
    for test in test_areas_raw:
        for area in areas:
            if test[0] == area.year and test[1] == area.month and test[2] == area.day and \
                    test[3] >= area.l_x and area.r_x >= test[4]:
                test_areas.append(area)
                break
    return test_areas

def get_subset_areas_dict(areas, file):
    test_areas_raw = read_partial_annotations(file)
    test_areas = []
    for test in test_areas_raw:
        for area in areas:
            if test[0] == area.year and test[1] == area.month and test[2] == area.day and \
                    test[3] >= area.l_x and area.r_x >= test[4]:
                test_areas.append({'area': area, 'detections': np.empty(shape=(0, 6))})
                break
    return test_areas