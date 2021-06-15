import numpy as np


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

    return spec