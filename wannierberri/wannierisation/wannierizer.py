import copy
import numpy as np
from .kpoint_and_neighbours import Kpoint_and_neighbours
from ..parallel import check_ray_initialized
from ..symmetry.sawf import VoidSymmetrizer


class Wannierizer:

    def __init__(self,
                 real_lattice,
                 bk_cart,
                 wcc_red,
                 parallel=True, symmetrizer=None,
                 ):
        self.real_lattice = real_lattice
        self.kpoints = []
        if parallel:
            parallel = check_ray_initialized()
        self.parallel = parallel
        if self.parallel:
            import ray
            self.ray_get = ray.get
            self.kpoints_and_neighbours_class = ray.remote(Kpoint_and_neighbours).remote
        else:
            self.ray_get = lambda x: x
            self.kpoints_and_neighbours_class = Kpoint_and_neighbours
        self.bk_cart = bk_cart
        if symmetrizer is None:
            symmetrizer = VoidSymmetrizer()
        self.symmetrizer = symmetrizer
        self.wcc_red = wcc_red
        self.wcc_cart = wcc_red.dot(real_lattice)
        self.spreads = None
        self.wcc_bk_phase = np.exp(1j * self.wcc_cart.dot(self.bk_cart.T))

    def add_kpoint(self, **kwargs):
        """
        add a k-point to the list of k-points

        Parameters
        ----------
        same as :class:`Kpoint_and_neighbours`
        """
        kpoint = self.kpoints_and_neighbours_class(**{k: copy.deepcopy(v) for k, v in kwargs.items()})
        self.kpoints.append(kpoint)

    def get_U_opt_full(self):
        if self.parallel:
            return np.array(self.ray_get([kpoint.get_U_opt_full.remote() for kpoint in self.kpoints]))
        else:
            return np.array([kpoint.get_U_opt_full() for kpoint in self.kpoints])

    def update_all(self, U_neigh, **kwargs):
        kwargs_loc = {**kwargs, "wcc_bk_phase": self.wcc_bk_phase}
        if self.parallel:
            remotes = [kpoint.update.remote(U, **kwargs_loc) for kpoint, U in zip(self.kpoints, U_neigh)]
            list = self.ray_get(remotes)
        else:
            list = [kpoint.update(U, **kwargs_loc) for kpoint, U in zip(self.kpoints, U_neigh)]
        U_k = [x[0] for x in list]
        self.update_wcc([x[1] for x in list], [x[2] for x in list])
        return U_k

    def update_Unb_all(self, U_neigh):
        if self.parallel:
            remotes = [kpoint.update_Unb.remote(U, wcc_bk_phase=self.wcc_bk_phase) for kpoint, U in zip(self.kpoints, U_neigh)]
            list = self.ray_get(remotes)
        else:
            list = [kpoint.update_Unb(U, wcc_bk_phase=self.wcc_bk_phase) for kpoint, U in zip(self.kpoints, U_neigh)]
        self.update_wcc([x[0] for x in list], [x[1] for x in list])


    def update_wcc(self, wcc_k, r2_k):
        wcc = self.symmetrizer.symmetrize_WCC(self.wcc_cart + sum(wcc_k))
        d_wcc = wcc - self.wcc_cart
        self.spreads = self.symmetrizer.symmetrize_spreads(sum(r2_k) - np.linalg.norm(d_wcc, axis=1)**2)
        self.wcc_cart = wcc
        self.wcc_red = wcc.dot(np.linalg.inv(self.real_lattice))
        self.wcc_bk_phase = np.exp(1j * self.wcc_cart.dot(self.bk_cart.T))
        return self.wcc_cart, self.spreads
