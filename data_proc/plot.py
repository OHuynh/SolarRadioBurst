import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
mpl.use('TkAgg')

def plot_bursts(data, annots, title=''):
    fig, ax = plt.subplots(1)
    plt.imshow(data, aspect='auto')
    plt.xlabel('time step')

    for annot in annots:
        by = (annot[2] - 10.0) / 70.0 * 400.0
        ty = (annot[3] - 10.0) / 70.0 * 400.0
        rect = patches.Rectangle((annot[0], by),
                                 annot[1] - annot[0],
                                 ty - by,
                                 linewidth=1,
                                 edgecolor='black' if annot[4] == 0 else 'r',
                                 facecolor='none')
        print(annot)
        ax.add_patch(rect)

    plt.ylabel('frequency')
    plt.title(title)
    plt.show()