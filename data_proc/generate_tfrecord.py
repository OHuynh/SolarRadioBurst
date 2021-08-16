from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf
from data_proc.read_utils import *
from data_proc.preprocess import remove_artifactsC
from data_proc.plot import plot_bursts
from data_proc.safe_areas import *
import cv2
import random
import io
from PIL import Image

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

params = {2: {'window': 2700,
              'large_window': 5000,
              'width_px': 320,
              'height_px': 160,
              'full_height': False},
          3: {'window': 500,
              'large_window': 2400,
              'width_px': 480,
              'height_px': 160,
              'full_height': False},
          4: {'window': 28800,
              'large_window': 28801, #~8 hours
              'width_px': 512,
              'height_px': 256,
              'full_height': False}}

def generate_tfrecord(areas, label, params, prefix_file, ratio_train_test, ratio_of_neg, ratio_of_neg_in_pos_areas, total_pos, variant_pos=1, show=True):
    writer_train = tf.compat.v1.python_io.TFRecordWriter('{}_train.tfrecord'.format(prefix_file))
    writer_test = tf.compat.v1.python_io.TFRecordWriter('{}_test.tfrecord'.format(prefix_file))
    sample_stored = 0
    store_in_test = False

    trains_pos = []
    tests_pos = []
    trains_neg = []
    tests_neg = []
    trains_areas = []
    tests_areas = []

    neg_in_pos_areas = int(ratio_of_neg_in_pos_areas * total_pos)
    neg_per_pos_area = np.max([1, int(ratio_of_neg_in_pos_areas)])
    neg_areas = 0

    #positives
    for area in areas:
        added_samples = 0
        for idx in range(len(area.get_positives())):
            for _ in range(variant_pos):
                filename = os.path.join('../Solar_Interface/Image_Filters/Read_Data/Data', str(area.year) + '%02d' % area.month)
                extension = "S" + str(area.year)[2:4] + '%02d' % area.month + '%02d' % area.day + ".RT1"
                filename = os.path.join(filename, extension)
                if not os.path.exists(filename):
                    continue
                img, _, _, _, _, _ = read_data(filename)

                #sometimes, daily measures are clipped
                if label != 3 and img.shape[1] != area.r_x:
                    area.r_x = img.shape[1] - 1

                positive = area.get_positives()[idx].copy()
                # get a safe area to randomly select a window containing the sample
                # if area is too small, we pad it
                width = (area.r_x - area.l_x)
                pad_coord = [area.l_x, area.r_x]
                if width < params['large_window']:
                    pad = params['large_window'] - width
                    pad_coord = [area.l_x - pad / 2, area.r_x + pad / 2]
                    # odd number
                    if pad_coord[1] - pad_coord[0] != params['large_window']:
                        pad_coord[0] += 1

                l_x = np.max([pad_coord[0], positive[1] - params['window']])
                r_x = np.min([pad_coord[1], positive[0] + params['window']]) - params['window']

                assert r_x - l_x > 0, 'Window for type {} too small'.format(label)
                pos_x = int((r_x - l_x) * np.random.rand() + l_x)

                #create new area with this window, smaller with an offset
                new_area = Area(area.year, area.month, area.day, area.type,
                                l_x=pos_x, r_x=pos_x+params['window'], b_y=area.b_y, t_y=area.t_y)
                new_area.add_valid_annot(area.get_positives(), with_height=False, allow_partial=True)

                filtered_img = remove_artifactsC(img)
                img_windowed = np.zeros(shape=(img.shape[0], params['window']))
                img_windowed[:, int(np.max([area.l_x - pos_x, 0])):int(np.min([area.r_x - pos_x, params['window']]))] = \
                    filtered_img[:, int(np.max([pos_x,area.l_x])):int(np.min([pos_x+params['window'], area.r_x]))]
                new_area.set_offset(-pos_x)

                img_windowed_cpy = img_windowed.copy()
                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                labels = []
                labels_txt = []
                for positive in new_area.get_positives():
                    xmins.append(positive[0] / params['window'])
                    xmaxs.append(positive[1] / params['window'])
                    if params['full_height']:
                        ymins.append(0.0)
                        ymaxs.append(1.0)
                    else:
                        ymins.append((positive[2] - 10.0) / 70.0)
                        ymaxs.append((positive[3] - 10.0) / 70.0)
                    labels.append(1)
                    labels_txt.append('Type'.encode('utf8'))

                if label == 4:
                    current_pos = 1800
                    while current_pos < area.r_x:
                        xmin_bis = []
                        xmax_bis = []
                        ymin_bis = []
                        ymax_bis = []
                        label_bis = []
                        labels_txt_bis = []
                        img_windowed_bis = img_windowed.copy()
                        img_windowed_bis[:,current_pos:] = 0
                        current_pos += 1800
                        current_pos_norm = current_pos/28800.0
                        i = 0
                        while i < len(xmins):
                            if xmins[i] <= current_pos_norm:
                                xmin_bis.append(xmins[i])
                                xmax_bis.append(np.min([current_pos_norm, xmaxs[i]]))
                                ymin_bis.append(ymins[i])
                                ymax_bis.append(ymaxs[i])
                                label_bis.append(labels[i])
                                labels_txt_bis.append(labels_txt[i])
                            i += 1
                        img_windowed_bis = cv2.resize(img_windowed_bis, dsize=(params['width_px'], params['height_px']))
                        cv2.imwrite('img/{}.png'.format(sample_stored), img_windowed_bis)
                        img_windowed_bis = (img_windowed_bis/ np.max(img_windowed_bis) * 255.0).astype(dtype=np.uint8)
                        cv2.imwrite('tmp.png', img_windowed_bis)
                        if show:
                            plot_bursts(img_windowed_bis,  # mettre img_windowed
                                    new_area.get_positives(),
                                    '{:02d}/{:02d}/{}'.format(area.day, area.month, area.year))
                        with tf.io.gfile.GFile('tmp.png', 'rb') as fid:
                            img_windowed_bis = fid.read()
                        tf_example = tf.train.Example(features=tf.train.Features(feature={
                            'image/height': int64_feature(params['height_px']),
                            'image/width': int64_feature(params['width_px']),
                            'image/encoded': bytes_feature(img_windowed_bis),
                            'image/format': bytes_feature('png'.encode('utf8')),
                            'image/object/bbox/xmin': float_list_feature(xmin_bis),
                            'image/object/bbox/xmax': float_list_feature(xmax_bis),
                            'image/object/bbox/ymin': float_list_feature(ymins),
                            'image/object/bbox/ymax': float_list_feature(ymaxs),
                            'image/object/class/label': int64_list_feature(label_bis),
                            'image/object/class/text': bytes_list_feature(labels_txt_bis),
                        }))
                        if store_in_test:
                            tests_pos.append(tf_example.SerializeToString())
                        else:
                            trains_pos.append(tf_example.SerializeToString())

                else:
                    img_windowed = cv2.resize(img_windowed, dsize=(params['width_px'], params['height_px']))  # image_windowed = image complète
                    cv2.imwrite('img/{}.png'.format(sample_stored), img_windowed)
                    img_windowed = (img_windowed / np.max(img_windowed) * 255.0).astype(dtype=np.uint8)
                    cv2.imwrite('tmp.png', img_windowed)
                    if show:
                        plot_bursts(img_windowed_cpy, #mettre img_windowed
                                new_area.get_positives(),
                                '{:02d}/{:02d}/{}'.format(area.day, area.month, area.year))
                    with tf.io.gfile.GFile('tmp.png', 'rb') as fid:
                        img_windowed = fid.read()
                #encoded_jpg_io = io.BytesIO(img_windowed)
                #image = Image.open(encoded_jpg_io)
                #width, height = image.size

                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        'image/height': int64_feature(params['height_px']),
                        'image/width': int64_feature(params['width_px']),
                        'image/encoded': bytes_feature(img_windowed),
                        'image/format': bytes_feature('png'.encode('utf8')),
                        'image/object/bbox/xmin': float_list_feature(xmins),
                        'image/object/bbox/xmax': float_list_feature(xmaxs), #à changer pour les types IV
                        'image/object/bbox/ymin': float_list_feature(ymins),
                        'image/object/bbox/ymax': float_list_feature(ymaxs),
                        'image/object/class/label': int64_list_feature(labels),
                        'image/object/class/text': bytes_list_feature(labels_txt),
                    }))

                if store_in_test:
                    tests_pos.append(tf_example.SerializeToString())
                    tests_areas.append(area)
                else:
                    trains_pos.append(tf_example.SerializeToString())
                    trains_areas.append(area)

            sample_stored += 1
            print('{} stored for type {}'.format(sample_stored, label))

            # negatives in positive area
            for _ in range(neg_per_pos_area):
                if neg_in_pos_areas > len(trains_neg):
                    filename = os.path.join('../Solar_Interface/Image_Filters/Read_Data/Data', str(area.year) + '%02d' % area.month)
                    extension = "S" + str(area.year)[2:4] + '%02d' % area.month + '%02d' % area.day + ".RT1"
                    filename = os.path.join(filename, extension)
                    if not os.path.exists(filename):
                        continue
                    img, _, _, _, _, _ = read_data(filename)
                    filtered_img = remove_artifactsC(img)
                    for _ in range(1000):
                        to_pad = np.max([params['window'] - (area.r_x - area.l_x), 0])
                        pos_x = np.random.rand() * np.max([(area.r_x - area.l_x - params['window']), 0]) + area.l_x
                        odd = 0
                        if int(to_pad / 2) * 2 != to_pad:
                            odd = 1
                        if to_pad == 0:
                            new_area = Area(area.year, area.month, area.day, area.type,
                                            l_x=int(np.max([pos_x, area.l_x])),
                                            r_x=int(np.min([pos_x + params['window'], area.r_x])),
                                            b_y=area.b_y,
                                            t_y=area.t_y)
                        else:
                            new_area = Area(area.year, area.month, area.day, area.type,
                                            l_x=int(np.max([pos_x, area.l_x])),
                                            r_x=int(np.min([pos_x + params['window'], area.r_x])),
                                            b_y=area.b_y,
                                            t_y=area.t_y)
                        new_area.add_valid_annot(area.get_positives(), with_height=False, allow_partial=True)

                        if new_area.get_positives().shape[0] == 0:
                            if to_pad == 0:
                                img_windowed = filtered_img[:, int(np.max([pos_x, area.l_x])):int(
                                    np.min([pos_x + params['window'], area.r_x]))]
                            else:
                                img_windowed[:, int(to_pad / 2):-int(to_pad / 2) - odd] = \
                                    filtered_img[:,
                                    int(np.max([pos_x, area.l_x])):int(np.min([pos_x + params['window'], area.r_x]))]

                            if show:
                                plot_bursts(img_windowed,
                                            [],
                                            '{:02d}/{:02d}/{}'.format(area.day, area.month, area.year))
                            img_windowed = cv2.resize(img_windowed, dsize=(params['width_px'], params['height_px']))

                            img_windowed = (img_windowed / np.max(img_windowed) * 255.0).astype(dtype=np.uint8)
                            cv2.imwrite('tmp.png', img_windowed)
                            with tf.io.gfile.GFile('tmp.png', 'rb') as fid:
                                img_windowed = fid.read()
                            tf_example = tf.train.Example(features=tf.train.Features(feature={
                                'image/height': int64_feature(params['height_px']),
                                'image/width': int64_feature(params['width_px']),
                                'image/encoded': bytes_feature(img_windowed),
                                'image/format': bytes_feature('png'.encode('utf8')),
                                'image/object/bbox/xmin': float_list_feature([]),
                                'image/object/bbox/xmax': float_list_feature([]),
                                'image/object/bbox/ymin': float_list_feature([]),
                                'image/object/bbox/ymax': float_list_feature([]),
                                'image/object/class/label': int64_list_feature([]),
                                'image/object/class/text': bytes_list_feature([]),
                            }))
                            if store_in_test:
                                tests_neg.append(tf_example.SerializeToString())
                                tests_areas.append(area)
                            else:
                                trains_neg.append(tf_example.SerializeToString())
                                trains_areas.append(area)
                            break

        if total_pos * ratio_train_test < sample_stored:
            store_in_test = True
        if len(area.get_positives()) == 0:
            filename = os.path.join('../Solar_Interface/Image_Filters/Read_Data/Data',
                                    str(area.year) + '%02d' % area.month)
            extension = "S" + str(area.year)[2:4] + '%02d' % area.month + '%02d' % area.day + ".RT1"
            filename = os.path.join(filename, extension)
            if os.path.exists(filename):
                neg_areas += 1

    # negatives = images sans événements
    neg_stored = 0
    store_in_test = False
    total_neg = ratio_of_neg * total_pos
    neg_per_neg_area = np.max([1, int(total_neg / neg_areas)])
    for area in areas:
        if neg_stored >= total_neg:
            break
        if len(area.get_positives()) == 0:
            for _ in range(neg_per_neg_area):
                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                labels = []
                labels_txt = []
                filename = os.path.join('../Solar_Interface/Image_Filters/Read_Data/Data',
                                        str(area.year) + '%02d' % area.month)
                extension = "S" + str(area.year)[2:4] + '%02d' % area.month + '%02d' % area.day + ".RT1"
                filename = os.path.join(filename, extension)
                if not os.path.exists(filename):
                    continue

                img, _, _, _, _, _ = read_data(filename)
                if img.shape[1] == 0:
                    print('Warning ! skip {}, empty file'.format(filename))
                    continue
                if label != 3 and img.shape[1] != area.r_x:
                    area.r_x = img.shape[1] - 1

                filtered_img = remove_artifactsC(img)
                img_windowed_bis = img_windowed

                to_pad = np.max([params['window'] - (area.r_x - area.l_x), 0])
                pos_x = np.random.rand() * np.max([(area.r_x - area.l_x - params['window']), 0]) + area.l_x
                odd = 0
                if int(to_pad / 2) * 2 != to_pad:
                    odd = 1
                if to_pad == 0:
                    img_windowed = filtered_img[:, int(np.max([pos_x, area.l_x])):int(np.min([pos_x + params['window'], area.r_x]))]
                else:
                    img_windowed[:, int(to_pad / 2):-int(to_pad / 2) - odd] = filtered_img[:, int(np.max([pos_x, area.l_x])):int(np.min([pos_x + params['window'], area.r_x]))]

                if show:
                    plot_bursts(img_windowed,
                                [],
                                '{:02d}/{:02d}/{}'.format(area.day, area.month, area.year))
                #crop ce qui a à droite de la ligne verticale et mettre des 0 à la place dans le cas d'un type 4 loop + 10 mins
                if label == 4:
                    current_pos = 1800
                    while current_pos < area.r_x:
                        xmin_bis = []
                        xmax_bis = []
                        ymin_bis = []
                        ymax_bis = []
                        label_bis = []
                        labels_txt_bis = []
                        img_windowed_bis = img_windowed.copy()
                        img_windowed_bis[:, current_pos:] = 0
                        current_pos += 1800
                        current_pos_norm = current_pos / 28800.0
                        i = 0
                        while i < len(xmins):
                            if xmins[i] <= current_pos_norm:
                                xmin_bis.append(xmins[i])
                                xmax_bis.append(np.min([current_pos_norm, xmaxs[i]]))
                                ymin_bis.append(ymins[i])
                                ymax_bis.append(ymaxs[i])
                                label_bis.append(labels[i])
                                labels_txt_bis.append(labels_txt[i])
                            i += 1
                        img_windowed_bis = cv2.resize(img_windowed_bis, dsize=(params['width_px'], params['height_px']))
                        cv2.imwrite('img/{}.png'.format(sample_stored), img_windowed_bis)
                        img_windowed_bis = (img_windowed_bis / np.max(img_windowed_bis) * 255.0).astype(dtype=np.uint8)
                        cv2.imwrite('tmp.png', img_windowed_bis)
                        if show:
                            plot_bursts(img_windowed_bis,  # mettre img_windowed
                                        new_area.get_positives(),
                                        '{:02d}/{:02d}/{}'.format(area.day, area.month, area.year))
                        with tf.io.gfile.GFile('tmp.png', 'rb') as fid:
                            img_windowed_bis = fid.read()
                        tf_example = tf.train.Example(features=tf.train.Features(feature={
                            'image/height': int64_feature(params['height_px']),
                            'image/width': int64_feature(params['width_px']),
                            'image/encoded': bytes_feature(img_windowed_bis),
                            'image/format': bytes_feature('png'.encode('utf8')),
                            'image/object/bbox/xmin': float_list_feature(xmin_bis),
                            'image/object/bbox/xmax': float_list_feature(xmax_bis),
                            'image/object/bbox/ymin': float_list_feature(ymins),
                            'image/object/bbox/ymax': float_list_feature(ymaxs),
                            'image/object/class/label': int64_list_feature(label_bis),
                            'image/object/class/text': bytes_list_feature(labels_txt_bis),
                        }))
                        if store_in_test:
                            tests_pos.append(tf_example.SerializeToString())
                        else:
                            trains_pos.append(tf_example.SerializeToString())
                else:
                    img_windowed = cv2.resize(img_windowed, dsize=(params['width_px'], params['height_px']))
                    img_windowed = (img_windowed / np.max(img_windowed) * 255.0).astype(dtype=np.uint8)
                    cv2.imwrite('tmp.png', img_windowed)
                    with tf.io.gfile.GFile('tmp.png', 'rb') as fid:
                        img_windowed = fid.read()
                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        'image/height': int64_feature(params['height_px']),
                        'image/width': int64_feature(params['width_px']),
                        'image/encoded': bytes_feature(img_windowed),
                        'image/format': bytes_feature('png'.encode('utf8')),
                        'image/object/bbox/xmin': float_list_feature(xmins),
                        'image/object/bbox/xmax': float_list_feature(xmaxs),
                        'image/object/bbox/ymin': float_list_feature(ymins),
                        'image/object/bbox/ymax': float_list_feature(ymaxs),
                        'image/object/class/label': int64_list_feature(labels),
                        'image/object/class/text': bytes_list_feature(labels_txt),
                    }))
                if store_in_test:
                    tests_neg.append(tf_example.SerializeToString())
                    tests_areas.append(area)
                else:
                    trains_neg.append(tf_example.SerializeToString())
                    trains_areas.append(area)

                neg_stored += 1
                print('{} neg stored for type {}'.format(neg_stored, label))
            if total_neg * ratio_train_test < neg_stored:
                store_in_test = True

    print('Writing...')
    #balancing pos and neg if neg > pos in training
    idx = 0
    while len(trains_pos) < len(trains_neg):
        trains_pos.append(trains_pos[idx])
        idx += 1
    trains = trains_pos + trains_neg
    tests = tests_pos + tests_neg

    #shuffling pos/neg
    random.shuffle(trains)
    random.shuffle(tests)

    for train in trains:
        writer_train.write(train)
    for test in tests:
        writer_test.write(test)
    print('-----------------')

    writer_train.close()
    writer_test.close()
    prev_area = None
    with open(prefix_file + "_train_areas.txt", "w") as f:
        for area in trains_areas:
            if area != prev_area:
                f.write('{},{},{},{},{}\n'.format(area.year, area.month, area.day, area.l_x, area.r_x))
                prev_area = area

    prev_area = None
    with open(prefix_file + "_test_areas.txt", "w") as f:
        for area in tests_areas:
            if area != prev_area:
                f.write('{},{},{},{},{}\n'.format(area.year, area.month, area.day, area.l_x, area.r_x))
                prev_area = area

def main():

    all_data = read_annotations(path='../Solar_Interface/Image_Filters/Read_Data/Data/Annotation.txt')
    years_available = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    sdate = date(2011, 1, 1)
    edate = date(2020, 12, 31)
    delta = edate - sdate

    # keep filters
    include_hard_type_4 = False
    if include_hard_type_4:
        samples_to_exclude_4 = np.empty(shape=(0, 8))
        all_data[all_data[:, 4] == 6, 4] = 4 # comment to not include hard type IV
    else:
        samples_to_exclude_4 = all_data[all_data[:, 4] == 6, :]

    #all_data = all_data[np.bitwise_or(np.bitwise_and(all_data[:, 4] == 2, all_data[:, 1] - all_data[:, 0] < 2000.0),
    #                                  all_data[:, 4] != 2)]
    print(all_data[np.bitwise_and(all_data[:, 4] == 2, all_data[:, 1] - all_data[:, 0] > 2000.0)])
    samples_to_exclude_3 = all_data[np.bitwise_and(all_data[:, 4] == 3, all_data[:, 1] - all_data[:, 0] > 80.0)]
    all_data = all_data[np.bitwise_or(np.bitwise_and(all_data[:, 4] == 3, all_data[:, 1] - all_data[:, 0] <= 80.0),
                                      all_data[:, 4] != 3)]

    areas_type_2, areas_type_3, areas_type_4, \
    total_pos_2, total_pos_3, total_pos_4 = get_data_in_areas(all_data, sdate, edate,
                                                              samples_to_exclude_3,
                                                              samples_to_exclude_4)

    random.shuffle(areas_type_2)
    random.shuffle(areas_type_3)
    random.shuffle(areas_type_4)
    generate_tfrecord(areas_type_2, 2, params[2], 'dataset_type_2', 0.75, 18.0, 5.0, total_pos_2, variant_pos=3, show=False)
    #generate_tfrecord(areas_type_3, 3, params[3], 'dataset_type_3', 0.75, 0.1, 3.0, total_pos_3, variant_pos=2, show=False)
    #generate_tfrecord(areas_type_4, 4, params[4], 'dataset_type_4', 0.75, 12.0, 0.0, total_pos_4, variant_pos=1, show=False)

if __name__ == '__main__':
    main()
