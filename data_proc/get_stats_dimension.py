
import numpy as np
import matplotlib.pyplot as plt
from data_proc.read_utils import read_annotations

all_data = read_annotations(path='../data/Annotation.txt',
                            dim=9)

params = {2: {'window': 2700},
          3: {'window': 500},
          4: {'window': 28800},
          5: {'window': 1200},
          }

fig, axs = plt.subplots(4, 4, tight_layout=True)
all_data[all_data[:, 4] == 6, 4] = 4
all_data = all_data[np.bitwise_or(np.bitwise_and(all_data[:, 4] == 3, all_data[:, 1] - all_data[:, 0] <= 80.0),
                                  all_data[:, 4] != 3)]
for type in range(2, 6):
    indices = all_data[:, 4] == type
    samples = all_data[indices, :]
    if type == 3: #should be in safe area
        safe_areas = all_data[all_data[:, 4] == 5]
        for sample_idx in np.arange(samples.shape[0]):
            sample = samples[sample_idx, :]
            found = False
            safe_for_the_day = safe_areas[(safe_areas[:, 5:8] == sample[5:8]).all(axis=1), :]
            for safe_area in safe_for_the_day:
                if safe_area[0] < sample[0] and sample[1] < safe_area[1]:
                    found = True
                    break
            if found:
                samples[sample_idx, 8] = 1
        #samples = samples[samples[:, 1] - samples[:, 0] < 2000] #ignore storm

        samples = samples[samples[:, 8] == 1]
    nb_samples = np.sum(indices)
    print('-----------------------------------------------')
    print('Type : {} Samples : {}'.format(type, nb_samples))
    if nb_samples > 0:
        widths = samples[:, 1] - samples[:, 0]
        heights = samples[:, 3] - samples[:, 2]
        max_x = np.max(widths)
        min_x = np.min(widths)
        max_y = np.max(heights)
        min_y = np.min(heights)
        ratio = (widths / params[type]['window']) / (heights / 70.0)
        #ratio = (widths) / (heights - 10.0)
        max_ratio = np.max(ratio)
        min_ratio = np.min(ratio)
        mean_x = np.mean(widths)
        mean_y = np.mean(heights)
        mean_ratio = np.mean(ratio)
        median_x = np.median(widths)
        median_y = np.median(heights)
        median_ratio = np.median(ratio)

        std_deviation_x = np.std(widths)
        std_deviation_y = np.std(heights)
        std_deviation_ratio = np.std(ratio)

        print('{:.0f} < x < {:.0f} mean : {:.2f} std_deviation : {:.2f} median : {:.0f}'.format(min_x, max_x, mean_x, std_deviation_x, median_x))
        print('{:.0f} < y < {:.0f} mean : {:.2f} std_deviation : {:.2f} median : {:.0f}'.format(min_y, max_y, mean_y, std_deviation_y, median_y))
        print('{:.3f} < x/y < {:.3f} mean : {:.2f} std_deviation : {:.2f} median : {:.3f}'.format(min_ratio, max_ratio, mean_ratio,
                                                                                                std_deviation_ratio,
                                                                                                median_ratio))

        axs[type - 2, 0].hist(widths, bins=1000)
        axs[type - 2, 1].hist(heights, bins=1000)
        axs[type - 2, 2].hist(ratio, bins=1000)
        axs[type - 2, 3].scatter(widths, heights)
        axs[type - 2, 0].autoscale()
        axs[type - 2, 1].autoscale()
        axs[type - 2, 2].autoscale()
        axs[type - 2, 3].autoscale()

plt.show()