import numpy as np
from . import Calculator
from wannierberri.formula import covariant as frml
from wannierberri.formula import covariant_basic as frml_basic
from wannierberri.result import KBandResult


# The base class for Tabulating
# particular calculators are below


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





class Energy(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Eavln, **kwargs)

class Velocity(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Velocity, **kwargs)


class BerryCurvature(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.Omega, **kwargs)


class Spin(Tabulator):
    r" Spin expectation :math:` \langle u | \mathbf{\sigma} | u \rangle`"
    def __init__(self, **kwargs):
        super().__init__(frml.Spin, **kwargs)

class DerBerryCurvature(Tabulator):
    r"Derivative of Berry curvature :math:`X_{ab}\partial_b\Omega_a`"
    def __init__(self, **kwargs):
        super().__init__(frml.DerOmega, **kwargs)

class OrbitalMoment(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.morb, **kwargs)


class DerOrbitalMoment(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml_basic.Der_morb, **kwargs)


class SpinBerry(Tabulator):

    def __init__(self, **kwargs):
        super().__init__(frml.SpinOmega, **kwargs)

