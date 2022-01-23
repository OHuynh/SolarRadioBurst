import cv2
import os
import tree

from data_proc.preprocess import remove_artifactsC
from data_proc.read_utils import *
from data_proc.plot import *

path = '../data'
path_dest = '../filtered_data'
assert os.path.exists(path), 'Create a data folder in project root and download RT1 inside YYYYMM folder'


def list_files_with_ext(path_rt1, ext):
    month_dir = [f for f in os.listdir(path_rt1) if os.path.isdir(os.path.join(path_rt1, f))]
    rt1_list = tree.flatten(
        tree.map_structure(lambda dir: [os.path.join(path_rt1, dir, f)
                                        for f in os.listdir(os.path.join(path_rt1, dir))
                                        if os.path.isfile(os.path.join(path_rt1, dir, f))
                                        and f.endswith(ext)], month_dir))
    return rt1_list


rt1_files = list_files_with_ext(path, '.RT1')

if not os.path.exists(path_dest):
    os.mkdir(path_dest)

for f in os.listdir(path):
    if os.path.isdir(os.path.join(path, f)) and not os.path.exists(os.path.join(path_dest, f)):
        os.mkdir(os.path.join(path_dest, f))

preprocessed_files = list_files_with_ext(path_dest, '.png')

for file in rt1_files:
    print(f'{len(preprocessed_files)}/{len(rt1_files)}')
    preprocessed_file = os.path.join(path_dest,
                                     os.path.split(os.path.split(file)[-2])[1],
                                     os.path.splitext(os.path.basename(file))[0] + '.png')
    if preprocessed_file in preprocessed_files:
        continue
    img, _, _, _, _ = read_data(file)
    #plot_bursts(img, [])
    filtered_img = remove_artifactsC(img).astype(dtype='uint8')
    #plot_bursts(filtered_img, [])
    cv2.imwrite(preprocessed_file, filtered_img)
    preprocessed_files.append(preprocessed_file)