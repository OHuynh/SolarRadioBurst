import numpy as np
import skimage
import ctypes


libFastMedianRFI = ctypes.CDLL("../data_proc/fastMedianRFI.so")

from data_proc.plot import plot_bursts
rad90 = np.pi / 2.0

def rfi_denoise(spec, T=60, k=1):
    """
    Args:
        spec: signal
        f: frequency range
        t: period (short-term/long term)
    Returns:
        spec : signal filtered
    """
    tau = 0.05
    nbf = spec.shape[0]
    nbT = spec.shape[1] / T

    mean_per_f = np.mean(spec, axis=1)
    e0_per_f = np.std(spec, axis=1)
    max_per_f = np.max(spec, axis=1)
    min_per_f = np.min(spec, axis=1)
    BF0_min = np.max([mean_per_f - 3 * e0_per_f, min_per_f], axis=0)
    BF0_max = np.min([mean_per_f + 3 * e0_per_f, max_per_f], axis=0)

    val0 = BF0_min + tau * (BF0_max - BF0_min)
    indices = np.arange(0, spec.shape[1] - T, T).reshape([-1, 1]) + np.arange(T).reshape([1, -1])
    spec_window = spec[:, indices]
    window_min = np.min(spec_window, axis=2)
    window_max = np.max(spec_window, axis=2)
    window_mean = np.mean(spec_window, axis=2)
    window_std = np.std(spec_window, axis=2)
    # window_BF0_min = np.max([window_mean - 3 * window_std, window_min], axis=0)
    # window_BF0_max = np.min([window_mean + 3 * window_std, window_max], axis=0)

    cond = window_mean < (mean_per_f.reshape([-1, 1]) + e0_per_f.reshape([-1, 1]))
    # Lol ? val = window_BF0_min + tau * (window_BF0_max - window_BF0_min)
    # val = moy
    spec[:, :indices.shape[0] * T].reshape([spec.shape[0], -1, T])[cond, :] = \
        np.abs(spec[:, :indices.shape[0] * T].reshape([spec.shape[0], -1, T])[cond, :]
               - window_mean[cond].reshape([-1, 1]))

    BF0_min_mean = []
    for i in range(400):
        BF0_min_mean.append(np.mean(BF0_min[:i + 1]))
    BF0_min_mean = np.array(BF0_min_mean)
    BF0_min_mean = BF0_min_mean.repeat(cond.shape[1]).reshape(BF0_min_mean.shape[0], -1)
    not_cond = np.bitwise_not(cond)
    spec[:, :indices.shape[0] * T].reshape([spec.shape[0], -1, T])[not_cond, :] = \
        np.abs(spec[:, :indices.shape[0] * T].reshape([spec.shape[0], -1, T])[not_cond, :]
               - k * BF0_min_mean[not_cond].reshape([-1, 1]))

    return spec, BF0_min_mean


def gradient_median2(spec, distance=15.0):
    """
    median filter along gradient direction
    Args:
        spec: rt1
        distance: distance along the gradient vector on which we take the median
    Returns: spec filtered
    """

    gx, gy = np.gradient(spec)
    GDirInv = np.mod(-np.arctan2(gy, gx), 2 * np.pi)
    Gdir = np.mod(rad90 + GDirInv, 2 * np.pi)
    Gmag = np.sqrt(gx** 2 + gy**2)

    AInv=-np.tan(GDirInv);
    pixels_along_grad = int(distance) * 2

    AInv[np.bitwise_or(GDirInv == rad90, GDirInv == 3 * rad90)] = \
        -np.tan(GDirInv[np.bitwise_or(GDirInv == rad90, GDirInv == 3 * rad90)])
    GDirInv[np.bitwise_or(GDirInv == rad90, GDirInv == 3 * rad90)] -= (2 * np.pi / 360.0)

    AInv[Gmag == 0] = -np.tan(GDirInv[Gmag == 0])
    GDirInv[Gmag == 0] = -(2*np.pi * 89.0 / 360.0)

    # find the segment to discretize along the gradient direction centered on each pixel
    pt1_ax = distance * np.cos(GDirInv)
    pt1_ay = distance * np.sin(GDirInv)
    pt2_ax = distance * np.cos(GDirInv + np.pi)
    pt2_ay = distance * np.sin(GDirInv + np.pi)

    xi = np.tile(np.arange(pt1_ax.shape[1]), pt1_ax.shape[0]).reshape(pt1_ax.shape[0], pt1_ax.shape[1])
    yi = np.repeat(np.arange(pt1_ax.shape[0]), pt1_ax.shape[1]).reshape(pt1_ax.shape[0], pt1_ax.shape[1])
    pt1_ax = xi + pt1_ax
    pt2_ax = xi + pt2_ax
    pt1_ay = yi + pt1_ay
    pt2_ay = yi + pt2_ay

    # very slow
    # pts_ax = np.round(np.linspace(pt1_ax, pt2_ax, 30))
    # pts_ay = np.round(np.linspace(pt1_ay, pt2_ay, 30))

    pts_ax = np.minimum(np.maximum(np.linspace(pt1_ax, pt2_ax, pixels_along_grad).astype(dtype=np.int32), 0), pt1_ax.shape[1] - 1)
    pts_ay = np.minimum(np.maximum(np.linspace(pt1_ay, pt2_ay, pixels_along_grad).astype(dtype=np.int32), 0), pt1_ax.shape[0] - 1)

    # appended value should be different from last one
    dup_x = np.diff(pts_ax, axis=0, append=(-50)).astype(dtype=np.bool)
    dup_y = np.diff(pts_ay, axis=0, append=(-50)).astype(dtype=np.bool)

    to_keep = np.bitwise_not(np.bitwise_or(dup_x, dup_y)) # use as mask
    del dup_x
    del dup_y
    medians = spec[pts_ay, pts_ax].reshape(pixels_along_grad, pt1_ax.shape[0], pt1_ax.shape[1])
    filtered_spec = np.ma.median(np.ma.array(medians, mask=to_keep), axis=0)
    del to_keep
    return filtered_spec.data


def gradient_main(spec):
    gx, gy = np.gradient(spec)
    GDirInv = np.mod(-np.arctan2(gy, gx), 2 * np.pi)
    Gdir = np.mod(rad90 + GDirInv, 2 * np.pi)
    Gmag = np.sqrt(gx** 2 + gy**2).astype(dtype=np.float32)
    width = 60
    img_height = spec.shape[0]
    img_width = spec.shape[1]

    Gmag_padded = np.pad(Gmag, pad_width=((0,0), (0, width * 2)))
    Gdir_padded = np.pad(Gdir, pad_width=((0,0), (0, width * 2)))
    Gmag_mean = np.zeros_like(Gmag)
    Gdir_mean = np.zeros_like(Gdir)
    for i in range(0, 2*width):
         Gmag_mean += Gmag_padded[:,i:img_width + i]
         Gdir_mean += Gdir_padded[:,i:img_width + i] * Gmag_padded[:,i:img_width + i]
    epsilon = 1e-7
    Gdir_mean /= (Gmag_mean + epsilon)
    alpha = 10.0
    Gmag_mean *= np.exp(-alpha * np.square((np.abs(np.abs(Gdir_mean) - rad90) / rad90)))

    dangle = 25.0 * np.pi / 180.0
    indices_90 = np.bitwise_or(np.bitwise_and(Gdir_mean > -(rad90 + dangle), Gdir_mean < (rad90 - dangle)),
                              np.bitwise_and(Gdir_mean > (rad90 - dangle), Gdir_mean < (rad90 + dangle)))
    sgmag = 150;
    indices_low_mag = Gmag_mean < sgmag
    Gdir_mean_n = Gdir_mean.copy()
    Gmag_mean_n = Gmag_mean.copy()
    Gdir_mean_n[indices_low_mag] = 0
    Gmag_mean_n[indices_low_mag] = 0
    Gdir_mean_n[indices_90] = Gdir_mean[indices_90]
    Gmag_mean_n[indices_90] = Gmag_mean[indices_90]

    flagp = np.zeros(Gdir_mean_n.shape, dtype=bool)
    flagp[Gdir_mean_n > 0] = True
    flagp = np.roll(flagp, 1, axis=0)
    flagp[-1, :] = False

    flagn = np.zeros(Gdir_mean_n.shape, dtype=bool)
    flagn[Gdir_mean_n < 0] = True
    flagn = np.roll(flagp, -1, axis=0)
    flagn[0, :] = False

    flag = np.bitwise_or(flagn, flagp)
    return flag


def remove_artifactsC(spec, T = 3600):
    # spec is structured variable among others:
    # spec.data : power data, linear
    guard = 2  # guard
    cal_d = 40
    nbp = np.ceil(spec.shape[1] / T).astype(dtype=np.int)
    simtf = spec.sum(axis=0).astype(dtype=np.int64) #np.sum(spec)
    simtf = np.diff(simtf)

    for k in range(nbp):
        debut = k * T
        fin = np.min([(k + 1) * T, spec.shape[1] - 1])
        F = abs(simtf[debut:fin])
        indice = np.argmax(F)
        pos = debut + indice + 1
        debut = max(pos - cal_d - guard, 1)
        fin = min(pos + guard, spec.shape[1]-1)

        # Remplacer calibres par bruit:
        # parametrage avant calibre
        debut_bd = max(debut - T, 0)
        fin_bd = max(debut - 1, 0)
        md = np.mean(spec[:, debut_bd:fin_bd+1],axis=1,keepdims=True)
        ed = np.std(spec[:, debut_bd:fin_bd+1],axis=1,keepdims=True)

        # parametrage apres calibre
        debut_bf = fin
        fin_bf = min(fin + T, spec.shape[1]-1)
        mf = np.mean(spec[:, debut_bf:fin_bf+1],axis=1,keepdims=True)
        ef = np.std(spec[:, debut_bf:fin_bf+1],axis=1,keepdims=True)

        # effet de bord
        if fin_bd == 1:
            ed = ef
            md = mf

        if debut_bf == spec.shape[1]:
            ef = ed
            mf = md

        # elimination des pulses de calibration
        ii = np.arange(debut, fin+1)
        e = (ef - ed) / (cal_d + 2 * guard) * (np.reshape(ii, [1, -1]) - debut) + ed
        moy = (mf - md) / (cal_d + 2 * guard) * (np.reshape(ii, [1, -1]) - debut) + md
        spec[:, debut: fin+1] = np.random.randn(spec.shape[0], fin - debut + 1) * e + moy

    k = 1
    #import time
    #import matplotlib.pyplot as plt
    #plot_bursts(spec, [])
    #start_time = time.time()
    spec, BF0_min_mean = rfi_denoise(spec, k=1)
    #plot_bursts(spec, [])
    spec = gradient_median2(spec)
    spec = spec + k * np.mean(BF0_min_mean)
    #plot_bursts(spec, [])
    flag = gradient_main(spec)
    #plot_bursts(flag, [])
    filtered_spec = np.zeros_like(spec)
    spec_t = np.ravel(spec.T)
    flag_t = np.ravel(flag.T)
    libFastMedianRFI.fastmedianRFI(ctypes.c_void_p(spec_t.ctypes.data),
                                   ctypes.c_int(spec.shape[0]),
                                   ctypes.c_int(spec.shape[1]),
                                   ctypes.c_int(15), #size_filter
                                   ctypes.c_void_p(flag_t.ctypes.data),
                                   ctypes.c_void_p(filtered_spec.ctypes.data)
                                   )
    filtered_spec = filtered_spec.reshape(-1, spec.shape[0]).T
    #spec = skimage.filters.rank.median(spec.astype(np.uint8), footprint=np.ones([1, 15]), mask=np.bitwise_not(flag))
    #plot_bursts(spec, [])
    #print(f"{time.time() - start_time} ms")
    return filtered_spec