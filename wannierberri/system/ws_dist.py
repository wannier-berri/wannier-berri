import numpy as np
from ..__utility import iterate3dpm


def wigner_seitz(real_lattice, mp_grid):
    ws_search_size = np.array([1] * 3)
    dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
    origin = divmod((dist_dim + 1), 2)[0] - 1
    real_metric = real_lattice.dot(real_lattice.T)
    mp_grid = np.array(mp_grid)
    irvec = []
    ndegen = []
    for n in iterate3dpm(mp_grid * ws_search_size):
        dist = []
        for i in iterate3dpm((1, 1, 1) + ws_search_size):
            ndiff = n - i * mp_grid
            dist.append(ndiff.dot(real_metric.dot(ndiff)))
        dist = np.array(dist)
        dist_min = np.min(dist)
        if abs(dist[origin] - dist_min) < 1.e-7:
            irvec.append(n)
            ndegen.append(np.sum(abs(dist - dist_min) < 1.e-7))

    return np.array(irvec), np.array(ndegen)
