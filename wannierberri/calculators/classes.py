from wannierberri import __result as result
from wannierberri.__tabulate import TABresult
import numpy as np
import abc, functools
from wannierberri.__kubo import Gaussian, Lorentzian
from collections import defaultdict
from math import ceil
# from numba import njit


class Calculator():

    def __init__(self, degen_thresh=1e-4, degen_Kramers=False, save_mode="bin+txt"):
        self.degen_thresh = degen_thresh
        self.degen_Kramers = degen_Kramers
        self.save_mode = save_mode


#######################################
#                                     #
#         integration (Efermi-only)   #
#                                     #
#######################################


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

        res = result.EnergyResult(self.Efermi, restot, TRodd=formula.TRodd, Iodd=formula.Iodd, smoothers=[self.smoother])
        res.set_save_mode(self.save_mode)
        return res


#######################################
#                                     #
#      integration (Efermi-omega)     #
#                                     #
#######################################


def FermiDirac(E, mu, kBT):
    "here E is a number, mu is an array"
    if kBT == 0:
        return 1.0 * (E <= mu)
    else:
        res = np.zeros_like(mu)
        res[mu > E + 30 * kBT] = 1.0
        res[mu < E - 30 * kBT] = 0.0
        sel = abs(mu - E) <= 30 * kBT
        res[sel] = 1.0 / (np.exp((E - mu[sel]) / kBT) + 1)
        return res


class DynamicCalculator(Calculator, abc.ABC):

    def __init__(self, Efermi=None, omega=None, kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', **kwargs):

        for k, v in locals().items():  # is it safe to do so?
            if k not in ['self', 'kwargs']:
                vars(self)[k] = v
        super().__init__(**kwargs)

        self.formula_kwargs = {}
        self.Formula = None
        self.final_factor = 1.
        self.dtype = complex
        self.EFmin = self.Efermi.min()
        self.EFmax = self.Efermi.max()
        self.omegamin = self.omega.min()
        self.omegamax = self.omega.max()
        self.eocc1max = self.EFmin - 30 * self.kBT
        self.eocc0min = self.EFmax + 30 * self.kBT

        if self.smr_type == 'Lorentzian':
            self.smear = functools.partial(Lorentzian, width=self.smr_fixed_width)
        elif self.smr_type == 'Gaussian':
            self.smear = functools.partial(Gaussian, width=self.smr_fixed_width, adpt_smr=False)
        else:
            raise ValueError("Invalid smearing type {self.smr_type}")
        self.FermiDirac = functools.partial(FermiDirac, mu=self.Efermi, kBT=self.kBT)

    @abc.abstractmethod
    def factor_omega(self, E1, E2):
        pass

    def factor_Efermi(self, E1, E2):
        return self.FermiDirac(E2) - self.FermiDirac(E1)

    def nonzero(self, E1, E2):
        if (E1 < self.eocc1max and E2 < self.eocc1max) or (E1 > self.eocc0min and E2 > self.eocc0min):
            return False
        else:
            return True

    def __call__(self, data_K):
        formula = self.Formula(data_K, **self.formula_kwargs)
        restot_shape = (len(self.omega), len(self.Efermi)) + (3, ) * formula.ndim
        restot_shape_tmp = (
            len(self.omega), len(self.Efermi) * 3**formula.ndim)  # we will first get it in this shape, then transpose

        restot = np.zeros(restot_shape_tmp, self.dtype)

        for ik in range(data_K.nk):
            degen_groups = data_K.get_bands_in_range_groups_ik(
                ik, -np.Inf, np.Inf, degen_thresh=self.degen_thresh, degen_Kramers=self.degen_Kramers)
            #now find needed pairs:
            # as a dictionary {((ibm1,ibm2),(ibn1,ibn2)):(Em,En)}
            degen_group_pairs = [
                (ibm, ibn, Em, En) for ibm, Em in degen_groups.items() for ibn, En in degen_groups.items()
                if self.nonzero(Em, En)
            ]
            npair = len(degen_group_pairs)
            matrix_elements = np.array(
                [formula.trace_ln(ik, np.arange(*pair[0]), np.arange(*pair[1])) for pair in degen_group_pairs])
            factor_Efermi = np.array([self.factor_Efermi(pair[2], pair[3]) for pair in degen_group_pairs])
            factor_omega = np.array([self.factor_omega(pair[2], pair[3]) for pair in degen_group_pairs]).T
            restot += factor_omega @ (factor_Efermi[:, :, None]
                                      * matrix_elements.reshape(npair, -1)[:, None, :]).reshape(npair, -1)
        restot = restot.reshape(restot_shape).swapaxes(0, 1)  # swap the axes to get EF,omega,a,b,...
        restot[:] *= self.final_factor / (data_K.nk * data_K.cell_volume)
        return result.EnergyResult(
            [self.Efermi, self.omega], restot, TRodd=formula.TRodd, Iodd=formula.Iodd, TRtrans=formula.TRtrans)


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
        return result.KBandResult(rslt, TRodd=formula.TRodd, Iodd=formula.Iodd)


class TabulatorAll(Calculator):

    def __init__(self, tabulators, ibands=None):
        """ tabulators - dict 'key':tabulator
        one of them should be "Energy" """
        self.tabulators = tabulators
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
            kpoints=data_K.kpoints_all,
            recip_lattice=data_K.system.recip_lattice,
            results={k: v(data_K)
                     for k, v in self.tabulators.items()})
