import numpy as np
import abc
import functools
from ..utility import Gaussian, Lorentzian, FermiDirac, cached_einsum
from ..result import EnergyResult
from .calculator import Calculator
from ..formula.covariant import SpinVelocity
from ..formula import Formula
from copy import copy
from ..symmetry.point_symmetry import transform_ident, transform_trans, transform_odd, transform_odd_trans_021
from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom
from .. import factors as factors

bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]


#######################################
#                                     #
#      integration (Efermi-omega)     #
#                                     #
#######################################


class DynamicCalculator(Calculator, abc.ABC):
    """
    Base class for calculators, which depend on Fermi level and frequency (omega). 

    Parameters
    ----------
    Efermi : array_like
        Fermi level(s) in eV.
    omega : array_like
        Frequencies in eV.
    kBT : float
        Temperature in eV.
    smr_fixed_width : float
        Fixed smearing width in eV.
    smr_type : str
        Type of smearing function. Can be 'Lorentzian' or 'Gaussian'.
    max_smear_arg : float
        For efficiency, when the argument of the smearing function is larger than max_smear_arg * smr_fixed_width, 
        the corresponding pairs of bands are ignored. If None, it is set to 100 for Lorentzian and 5 for Gaussian.
    tetra : bool
        Evaluate the delta functions delta(E_n - E_m -/+ omega) by the linear tetrahedron
        method instead of the smearing, for the calculators that allow it (`JDOS`, `OpticalConductivity`, `SHC`,
        `ShiftCurrent`, `InjectionCurrent`). The band energies are interpolated linearly between the corners of
        the parallelepiped (or tetrahedron) around each k-point, while the matrix elements and the occupation
        factors are taken at the k-point itself. The result at each frequency is the average of the delta function
        over the interval between the midpoints to the neighbouring frequencies (``omega`` must be finite and
        increasing, with at least two values), so that the integral of the delta function over these intervals is
        kept, also for features narrower than the intervals. The principal-value (real) part
        of `OpticalConductivity` and `SHC` still uses the Lorentzian of width ``smr_fixed_width``, as with
        ``smr_type='Gaussian'``. Requires a grid (not a path).
    kwargs_formula : dict
        Additional keyword arguments for the formula class.
    """

    # if True, only the pair of each band group with itself is taken (terms with a single band index)
    self_pairs_only = False
    # if True, the delta functions of the frequency can be evaluated by the tetrahedron method (tetra=True)
    allow_tetra = False
    # with tetra=True, which of delta(E1 - E2 - omega) and delta(E2 - E1 - omega) factor_omega uses
    tetra_deltas = (True, True)

    def __init__(self, Efermi=None, omega=None, kBT=0, smr_fixed_width=0.1,
                 max_smear_arg=None, smr_type='Lorentzian',
                 kwargs_formula=None, dtype=complex, tetra=False, **kwargs):


        if kwargs_formula is None:
            kwargs_formula = {}
        super().__init__(**kwargs)
        self.Efermi = np.array(Efermi).reshape(-1)
        self.omega = omega
        self.kBT = kBT
        self.smr_fixed_width = smr_fixed_width
        self.smr_type = smr_type
        if tetra and not self.allow_tetra:
            raise ValueError(f"the tetrahedron method (tetra=True) is not implemented for {type(self).__name__}")
        self.tetra = tetra
        if tetra:
            w = np.asarray(omega, dtype=float).reshape(-1)
            if len(w) < 2 or not np.all(np.isfinite(w)) or np.any(np.diff(w) <= 0):
                raise ValueError("with the tetrahedron method (tetra=True), omega must be finite and increasing, with "
                                 f"at least two values; got {omega}")
            mid = (w[1:] + w[:-1]) / 2
            self.omega_edges = np.concatenate(([2 * w[0] - mid[0]], mid, [2 * w[-1] - mid[-1]]))
        self.kwargs_formula = copy(kwargs_formula)
        self.Formula = None
        self.constant_factor = 1.
        self.dtype = dtype
        self.EFmin = self.Efermi.min()
        self.EFmax = self.Efermi.max()
        self.omegamin = self.omega.min()
        self.omegamax = self.omega.max()
        self.eocc1max = self.EFmin - 30 * self.kBT
        self.eocc0min = self.EFmax + 30 * self.kBT

        if self.smr_type == 'Lorentzian':
            self.smear = functools.partial(Lorentzian, width=self.smr_fixed_width)
            if max_smear_arg is None:
                max_smear_arg = 100
        elif self.smr_type == 'Gaussian':
            self.smear = functools.partial(Gaussian, width=self.smr_fixed_width, adpt_smr=False)
            if max_smear_arg is None:
                max_smear_arg = 5
        else:
            raise ValueError("Invalid smearing type {self.smr_type}")
        self.max_smear_width = max_smear_arg * self.smr_fixed_width
        self.FermiDirac = functools.partial(FermiDirac, mu=self.Efermi, kBT=self.kBT)

    @abc.abstractmethod
    def factor_omega(self, E1, E2):
        """determines a frequency-dependent factor for bands with energies E1 and E2

        Calculators with ``allow_tetra = True`` take an additional argument ``tetra_delta``: with tetra=True, the pair
        of arrays (delta(E1 - E2 - omega), delta(E2 - E1 - omega)) from the tetrahedron method (None for the one
        not selected by ``tetra_deltas``), otherwise None."""

    def factor_Efermi(self, E1, E2):
        return self.FermiDirac(E2) - self.FermiDirac(E1)

    def nonzero(self, E1, E2):
        if (E1 < self.eocc1max and E2 < self.eocc1max) or (E1 > self.eocc0min and E2 > self.eocc0min):
            return False
        else:
            return True

    def __call__(self, data_K):
        formula = self.Formula(data_K=data_K, **self.kwargs_formula)
        restot_shape = (len(self.omega), len(self.Efermi)) + (3,) * formula.ndim
        restot_shape_tmp = (
            len(self.omega), len(self.Efermi) * 3 ** formula.ndim)  # we will first get it in this shape, then transpose

        restot = np.zeros(restot_shape_tmp, self.dtype)

        for ik in range(data_K.nk):
            degen_groups = data_K.get_bands_in_range_groups_ik(
                ik, -np.inf, np.inf, degen_thresh=self.degen_thresh, degen_Kramers=self.degen_Kramers)
            # now find needed pairs:
            # as a dictionary {((ibm1,ibm2),(ibn1,ibn2)):(Em,En)}
            if self.self_pairs_only:
                degen_group_pairs = [
                    (ibm, ibm, Em, Em) for ibm, Em in degen_groups.items()
                    if self.nonzero(Em, Em)]
            else:
                degen_group_pairs = [
                    (ibm, ibn, Em, En) for ibm, Em in degen_groups.items() for ibn, En in degen_groups.items()
                    if self.nonzero(Em, En)]
            npair = len(degen_group_pairs)
            if npair == 0:
                continue

            matrix_elements = np.array(
                [formula.trace_ln(ik, np.arange(*pair[0]), np.arange(*pair[1])) for pair in degen_group_pairs])
            factor_Efermi = np.array([self.factor_Efermi(pair[2], pair[3]) for pair in degen_group_pairs])
            if self.tetra:
                # pairs with no occupation factor (e.g. a group with itself) do not contribute
                group_pairs = [(pair[0], pair[1]) if np.any(fE != 0) else ((0, 0), (0, 0))
                               for pair, fE in zip(degen_group_pairs, factor_Efermi)]
                weights = data_K.tetraWeights.delta_weights
                delta12 = weights(ik, group_pairs, self.omega_edges) if self.tetra_deltas[0] else [None] * npair
                delta21 = (weights(ik, [(b, a) for a, b in group_pairs], self.omega_edges) if self.tetra_deltas[1]
                           else [None] * npair)
                factor_omega = np.array([self.factor_omega(pair[2], pair[3], tetra_delta=(d12, d21))
                                         for pair, d12, d21 in zip(degen_group_pairs, delta12, delta21)]).T
            else:
                factor_omega = np.array([self.factor_omega(pair[2], pair[3]) for pair in degen_group_pairs]).T
            # raise ValueError(f"{factor_omega.shape=}, {factor_Efermi.shape=}, {matrix_elements.shape=}")
            restot += factor_omega @ (factor_Efermi[:, :, None] *
                                      matrix_elements.reshape(npair, -1)[:, None, :]).reshape(npair, -1)
        restot = restot.reshape(restot_shape).swapaxes(0, 1)  # swap the axes to get EF,omega,a,b,...
        restot *= self.constant_factor / (data_K.nk * data_K.cell_volume)
        try:
            transformTR = self.transformTR
        except AttributeError:
            transformTR = formula.transformTR
        try:
            transformInv = self.transformInv
        except AttributeError:
            transformInv = formula.transformInv

        return EnergyResult(
            [self.Efermi, self.omega], restot,
            transformTR=transformTR, transformInv=transformInv,
            save_mode=self.save_mode
        )


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


###############################
#              JDOS           #
###############################
class Formula_dyn_ident(Formula):

    def __init__(self, data_K):
        super().__init__(data_K)
        self.transformTR = transform_ident
        self.transformInv = transform_ident
        self.ndim = 0

    def trace_ln(self, ik, inn1, inn2):
        return len(inn1) * len(inn2)


class JDOS(DynamicCalculator):
    r"""Joint Density of States

    :math:`\rho(\omega) = \sum_{\mathbf{k}} \sum_{m,n} \delta(E_{m\mathbf{k}} - E_{n\mathbf{k}} - \omega) \times \left(f(E_{n\mathbf{k}}) - f(E_{m\mathbf{k}})\right)` 	
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = self.smr_fixed_width
        self.Formula = Formula_dyn_ident
        self.dtype = float

    allow_tetra = True
    tetra_deltas = (True, False)

    def factor_omega(self, E1, E2, tetra_delta=None):
        if tetra_delta is not None:
            return tetra_delta[0]
        return self.smear(E1 - E2 - self.omega)

    def nonzero(self, E1, E2):
        if not super().nonzero(E1, E2):
            return False
        if self.tetra:  # the energies at the corners may differ from E1, E2
            return True
        return self.omega.min() - self.max_smear_width < E1 - E2 < self.omega.max() + self.max_smear_width


################################
#    Optical Conductivity      #
################################

class Formula_OptCond(Formula):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        A = data_K.get_A_H(external_terms=self.external_terms)
        self.AA = 1j * A[:, :, :, :, None] * A.swapaxes(1, 2)[:, :, :, None, :]
        self.ndim = 2
        self.transformTR = transform_trans
        self.transformInv = transform_ident

    def trace_ln(self, ik, inn1, inn2):
        return self.AA[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class OpticalConductivity(DynamicCalculator):

    allow_tetra = True
    tetra_deltas = (False, True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Formula = Formula_OptCond
        self.constant_factor = factors.factor_opt

    def factor_omega(self, E1, E2, tetra_delta=None):
        delta_arg_12 = E2 - E1 - self.omega  # argument of delta function [iw, n, m]
        cfac = 1. / (delta_arg_12 - 1j * self.smr_fixed_width)
        if tetra_delta is not None:
            cfac.imag = np.pi * tetra_delta[1]
        elif self.smr_type != 'Lorentzian':
            cfac.imag = np.pi * self.smear(delta_arg_12)
        return (E2 - E1) * cfac


###############################
#          SHC                #
###############################


class Formula_SHC(Formula):

    def __init__(self, data_K, SHC_type='ryoo', shc_abc=None, **parameters):
        super().__init__(data_K, **parameters)
        A = SpinVelocity(data_K, SHC_type, external_terms=self.external_terms).matrix
        B = -1j * data_K.get_A_H(external_terms=self.external_terms)
        self.imAB = np.imag(A[:, :, :, :, None, :] * B.swapaxes(1, 2)[:, :, :, None, :, None])
        self.ndim = 3
        if shc_abc is not None:
            assert len(shc_abc) == 3
            a, b, c = (x - 1 for x in shc_abc)
            self.imAB = self.imAB[:, :, :, a, b, c]
            self.ndim = 0
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def trace_ln(self, ik, inn1, inn2):
        return self.imAB[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class SHC(DynamicCalculator):

    allow_tetra = True
    tetra_deltas = (True, False)

    def __init__(self, SHC_type="ryoo", shc_abc=None, **kwargs):
        super().__init__(**kwargs)
        self.kwargs_formula.update(dict(SHC_type=SHC_type, shc_abc=shc_abc))
        self.Formula = Formula_SHC
        self.constant_factor = factors.factor_shc

    def factor_omega(self, E1, E2, tetra_delta=None):
        delta_arg_12 = E1 - E2 - self.omega  # argument of delta function [iw, n, m]
        cfac = 1. / (delta_arg_12 - 1j * self.smr_fixed_width)
        if tetra_delta is not None:
            cfac.imag = np.pi * tetra_delta[0]
        elif self.smr_type != 'Lorentzian':
            cfac.imag = np.pi * self.smear(delta_arg_12)
        return cfac / 2


# ===============
#  Shift current
# ===============

class ShiftCurrentFormula(Formula):

    def __init__(self, data_K, sc_eta, **parameters):
        super().__init__(data_K, **parameters)
        D_H = data_K.D_H
        V_H = data_K.Xbar('Ham', 1)
        del2E_H = data_K.Xbar('Ham', 2)
        dEig_inv = data_K.dEig_inv.swapaxes(2, 1)

        # define D using broadening parameter
        E_K = data_K.E_K
        dEig = E_K[:, :, None] - E_K[:, None, :]
        dEig_inv_Pval = dEig / (dEig ** 2 + sc_eta ** 2)
        D_H_Pval = -V_H * dEig_inv_Pval[:, :, :, None]

        # commutators
        # ** the spatial index of D_H_Pval corresponds to generalized derivative direction
        # ** --> stored in the fourth column of output variables
        sum_HD = (cached_einsum('knlc,klma->knmca', V_H, D_H_Pval) -
                  cached_einsum('knnc,knma->knmca', V_H, D_H_Pval) -
                  cached_einsum('knla,klmc->knmca', D_H_Pval, V_H) +
                  cached_einsum('knma,kmmc->knmca', D_H_Pval, V_H))

        # ** this one is invariant under a<-->c
        DV_bit = (cached_einsum('knmc,knna->knmca', D_H, V_H) -
                  cached_einsum('knmc,kmma->knmca', D_H, V_H) +
                  cached_einsum('knma,knnc->knmca', D_H, V_H) -
                  cached_einsum('knma,kmmc->knmca', D_H, V_H))

        # generalized derivative
        A_gen_der = (+ 1j * (del2E_H + sum_HD + DV_bit) * dEig_inv[:, :, :, np.newaxis, np.newaxis])
        if self.external_terms:
            # ** the spatial index of A_Hbar with diagonal terms corresponds to generalized derivative direction
            # ** --> stored in the fourth column of output variables
            A_Hbar = data_K.Xbar('AA')
            A_Hbar_der = data_K.Xbar('AA', 1)
            sum_AD = (cached_einsum('knlc,klma->knmca', A_Hbar, D_H_Pval) -
                      cached_einsum('knnc,knma->knmca', A_Hbar, D_H_Pval) -
                      cached_einsum('knla,klmc->knmca', D_H_Pval, A_Hbar) +
                      cached_einsum('knma,kmmc->knmca', D_H_Pval, A_Hbar))
            AD_bit = (cached_einsum('knnc,knma->knmac', A_Hbar, D_H) -
                      cached_einsum('kmmc,knma->knmac', A_Hbar, D_H) +
                      cached_einsum('knna,knmc->knmac', A_Hbar, D_H) -
                      cached_einsum('kmma,knmc->knmac', A_Hbar, D_H))
            AA_bit = (cached_einsum('knnb,knma->knmab', A_Hbar, A_Hbar) -
                      cached_einsum('kmmb,knma->knmab', A_Hbar, A_Hbar))

            A_gen_der += A_Hbar_der + AD_bit - 1j * AA_bit + sum_AD

        # generalized derivative is fourth index of A, we put it into third index of Imn
        A_H = data_K.get_A_H(external_terms=self.external_terms)
        # here we take the -real part to eliminate the 1j factor in the final factor
        Imn = - cached_einsum('knmca,kmnb->knmabc', A_gen_der, A_H).imag
        Imn += Imn.swapaxes(4, 5)  # symmetrize b and c

        self.Imn = Imn
        self.ndim = 3
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class ShiftCurrent(DynamicCalculator):

    allow_tetra = True

    def __init__(self, sc_eta, **kwargs):
        super().__init__(dtype=float, **kwargs)
        self.kwargs_formula.update(dict(sc_eta=sc_eta))
        self.Formula = ShiftCurrentFormula
        self.constant_factor = factors.factor_shift_current

    def factor_omega(self, E1, E2, tetra_delta=None):
        if tetra_delta is not None:
            return tetra_delta[0] + tetra_delta[1]
        delta_arg_12 = E1 - E2 - self.omega  # argument of delta function [iw, n, m]
        delta_arg_21 = E2 - E1 - self.omega
        return self.smear(delta_arg_12) + self.smear(delta_arg_21)


# ===================
#  Injection current
# ===================


class InjectionCurrentFormula(Formula):
    """
    Eq. (10) of Lihm and Park, PRB 105, 045201 (2022)
    Use v_mn = i * r_mn * (e_m - e_n) / hbar to replace v with r.
    """

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        A_H = data_K.get_A_H(external_terms=self.external_terms)
        Vn = data_K.delE_K
        # compute delta_V[k, m, n, a] = V_H[k, m, m, a] - V_H[k, n, n, a]
        delta_V = Vn[:, :, None, :] - Vn[:, None, :, :]  # (k, m, n, a)

        Imn = cached_einsum('kmna,kmnb,knmc->kmnabc', delta_V, A_H, A_H)

        self.Imn = Imn
        self.ndim = 3

    def trace_ln(self, ik, inn1, inn2):
        return self.Imn[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class InjectionCurrent(DynamicCalculator):

    allow_tetra = True
    tetra_deltas = (True, False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Formula = InjectionCurrentFormula
        self.transformTR = transform_odd_trans_021
        self.transformInv = transform_odd
        self.constant_factor = factors.factor_injection_current

    def factor_omega(self, E1, E2, tetra_delta=None):
        if tetra_delta is not None:
            return tetra_delta[0]
        delta_arg_12 = E1 - E2 - self.omega  # argument of delta function [iw, n, m]
        return self.smear(delta_arg_12)
