import numpy as np

def rfi_denoise(spec, T=60):
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
    k = 1

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

    return spec



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

    spec = rfi_denoise(spec)


    return spec