#######################################
#                                     #
#         integration (Efermi-only)   #
#                                     #
#######################################


import numpy as np
from collections import defaultdict
from math import ceil
from wannierberri import covariant_formulak as frml
from wannierberri import __factors as factors
from wannierberri.__result import EnergyResult
from . import Calculator


# The base class for Static Calculators
# particular calculators are below

class StaticCalculator(Calculator):

    def __init__(self, Efermi, tetra=False, smoother=None, kwargs_formula={}, **kwargs):
        self.Efermi = Efermi
        self.tetra = tetra
        self.kwargs_formula = kwargs_formula
        self.smoother = smoother
        assert hasattr(self, 'factor')
        assert hasattr(
            self, 'fder'), "fder not set -  derivative of fermi distribution . 0: fermi-sea, 1: fermi-surface 2: f''  "
        assert hasattr(self, 'Formula'), "Formula not set - it  should be class with a trace(ik,inn,out) method "

        if not self.tetra:
            self.extraEf = 0 if self.fder == 0 else 1 if self.fder in (1, 2) else 2 if self.fder == 3 else None
            self.dEF = Efermi[1] - Efermi[0]
            self.EFmin = Efermi[0] - self.extraEf * self.dEF
            self.EFmax = Efermi[-1] + self.extraEf * self.dEF
            self.nEF_extra = Efermi.shape[0] + 2 * self.extraEf

        super().__init__(**kwargs)

    def __call__(self, data_K):

        nk = data_K.nk
        NB = data_K.num_wann
        formula = self.Formula(data_K, **self.kwargs_formula)
        ndim = formula.ndim

        # get a list [{(ib1,ib2):W} for ik in op:ed]
        if self.tetra:
            weights = data_K.tetraWeights.weights_all_band_groups(
                self.Efermi, der=self.fder, degen_thresh=self.degen_thresh,
                degen_Kramers=self.degen_Kramers)  # here W is array of shape Efermi
        else:
            weights = data_K.get_bands_in_range_groups(
                self.EFmin,
                self.EFmax,
                degen_thresh=self.degen_thresh,
                degen_Kramers=self.degen_Kramers,
                sea=(self.fder == 0))  # here W is energy

#        """formula  - TraceFormula to evaluate
#           bands = a list of lists of k-points for every
        shape = (3, ) * ndim
        lambdadic = lambda: np.zeros(((3, ) * ndim), dtype=float)
        values = [defaultdict(lambdadic) for ik in range(nk)]
        for ik, bnd in enumerate(weights):
            if formula.additive:
                for n in bnd:
                    inn = np.arange(n[0], n[1])
                    out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
                    values[ik][n] = formula.trace(ik, inn, out)
            else:
                nnall = set([_ for n in bnd for _ in n])
                _values = {}
                for n in nnall:
                    inn = np.arange(0, n)
                    out = np.arange(n, self.NB)
                    _values[n] = formula.trace(ik, inn, out)
                for n in bnd:
                    values[ik][n] = _values[n[1]] - _values[n[0]]

        if self.tetra:
            # tetrahedron method
            restot = np.zeros(self.Efermi.shape + shape)
            for ik, weights in enumerate(weights):
                valuesik = values[ik]
                for n, w in weights.items():
                    restot += np.einsum("e,...->e...", w, valuesik[n])
        else:
            # no tetrahedron
            restot = np.zeros((self.nEF_extra, ) + shape)
            for ik, weights in enumerate(weights):
                valuesik = values[ik]
                for n, E in sorted(weights.items()):
                    if E < self.EFmin:
                        restot += valuesik[n][None]
                    elif E <= self.EFmax:
                        iEf = ceil((E - self.EFmin) / self.dEF)
                        restot[iEf:] += valuesik[n]
            if self.fder == 0:
                pass
            elif self.fder == 1:
                restot = (restot[2:] - restot[:-2]) / (2 * self.dEF)
            elif self.fder == 2:
                restot = (restot[2:] + restot[:-2] - 2 * restot[1:-1]) / (self.dEF**2)
            elif self.fder == 3:
                restot = (restot[4:] - restot[:-4] - 2 * (restot[3:-1] - restot[1:-3])) / (2 * self.dEF**3)
            else:
                raise NotImplementedError(f"Derivatives  d^{self.fder}f/dE^{self.fder} is not implemented")

        restot *= self.factor / (data_K.nk * data_K.cell_volume)

        res = EnergyResult(self.Efermi, restot, TRodd=formula.TRodd, Iodd=formula.Iodd, smoothers=[self.smoother])
        res.set_save_mode(self.save_mode)
        return res



###############################################
###############################################
###############################################
###############################################
####                                     ######
####        Implemented calculators      ######
####                                     ######
###############################################
###############################################
###############################################
###############################################


#  TODO: Ideally, a docstring of every calculator should contain the equation that it implements
#        and references (with urls) to the relevant papers



class _DOS(StaticCalculator):

    def __init__(self, fder,**kwargs):
        self.Formula = frml.Identity
        self.factor = 1
        self.fder = fder
        super().__init__(**kwargs)

    def __call__(self, data_K):
        return super().__call__(data_K) * data_K.cell_volume

class DOS(_DOS):

    def __init__(self, **kwargs):
        super().__init__(fder=1, **kwargs)

class CumDOS(_DOS):

    def __init__(self, **kwargs):
        super().__init__(fder=0, **kwargs)


class AHC(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Omega
        self.factor = factors.fac_ahc
        self.fder = 0
        super().__init__(**kwargs)


class Ohmic(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.InvMass
        self.factor = factors.factor_ohmic
        self.fder = 0
        super().__init__(**kwargs)


class BerryDipole_FermiSurf(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelOmega
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)


class BerryDipole_FermiSea(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.DerOmega
        self.factor = 1
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res
