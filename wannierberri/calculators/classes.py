import numpy as np
import abc, functools
from collections import defaultdict
from math import ceil
from wannierberri.__result_tab import KBandResult,TABresult
from . import Calculator





#######################################
#                                     #
#      integration (Efermi-omega)     #
#                                     #
#######################################




#######################################
#                                     #
#      tabulating                     #
#                                     #
#######################################


class Tabulator(Calculator):

    def __init__(self, Formula, kwargs_formula={}, **kwargs):
        self.Formula = Formula
        self.ibands = None
        self.kwargs_formula = kwargs_formula
        super().__init__(**kwargs)

    def __call__(self, data_K):
        formula = self.Formula(data_K, **self.kwargs_formula)
        nk = data_K.nk
        NB = data_K.num_wann
        ibands = self.ibands
        if ibands is None:
            ibands = np.arange(NB)
        band_groups = data_K.get_bands_in_range_groups(
            -np.Inf, np.Inf, degen_thresh=self.degen_thresh, degen_Kramers=self.degen_Kramers, sea=False)
        # bands_groups  is a digtionary (ib1,ib2):E
        # now select only the needed groups
        band_groups = [
            [n for n in groups.keys() if np.any((ibands >= n[0]) * (ibands < n[1]))] for groups in band_groups
        ]  # select only the needed groups
        group = [[] for ik in range(nk)]
        for ik in range(nk):
            for ib in ibands:
                for n in band_groups[ik]:
                    if ib < n[1] and ib >= n[0]:
                        group[ik].append(n)
                        break

        rslt = np.zeros((nk, len(ibands)) + (3, ) * formula.ndim)
        for ik in range(nk):
            values = {}
            for n in band_groups[ik]:
                inn = np.arange(n[0], n[1])
                out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
                values[n] = formula.trace(ik, inn, out) / (n[1] - n[0])
            for ib, b in enumerate(ibands):
                rslt[ik, ib] = values[group[ik][ib]]
        return KBandResult(rslt, TRodd=formula.TRodd, Iodd=formula.Iodd)


class TabulatorAll(Calculator):

    def __init__(self, tabulators, ibands=None, mode="grid"):
        """ tabulators - dict 'key':tabulator
        one of them should be "Energy" """
        self.tabulators = tabulators
        mode = mode.lower()
        assert mode in ("grid","path")
        self.mode = mode
        if "Energy" not in self.tabulators.keys():
            raise ValueError("Energy is not included in tabulators")
        if ibands is not None:
            ibands = np.array(ibands)
        for k, v in self.tabulators.items():
            if v.ibands is None:
                v.ibands = ibands
            else:
                assert v.ibands == ibands

    def __call__(self, data_K):
        return TABresult(
            kpoints=data_K.kpoints_all.copy(),
            mode=self.mode,
            recip_lattice=data_K.system.recip_lattice,
            results={k: v(data_K)
                     for k, v in self.tabulators.items()} )


    @property
    def allow_path(self):
        return self.mode == "path"
