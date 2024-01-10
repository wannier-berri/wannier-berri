#######################################
#                                     #
#         integration (Efermi-only)   #
#                                     #
#######################################


import numpy as np
from collections import defaultdict
from math import ceil
from copy import copy
from ..formula import covariant as frml
from ..formula import covariant_basic as frml_basic
from .. import __factors as factors
from ..result import EnergyResult, K__Result
from . import Calculator
from ..__utility import alpha_A, beta_A

# The base class for Static Calculators
# particular calculators are below


class StaticCalculator(Calculator):

    def __init__(self, Efermi, tetra=False, smoother=None, constant_factor=1., use_factor=True, kwargs_formula={},
                 Emin=-np.Inf, Emax=np.Inf, hole_like=False, k_resolved=False, Formula=None, fder=None, **kwargs):
        self.Efermi = Efermi
        self.Emin = Emin
        self.Emax = Emax
        self.tetra = tetra
        self.kwargs_formula = copy(kwargs_formula)
        self.k_resolved = k_resolved
        self.smoother = smoother
        self.use_factor = use_factor
        self.hole_like = hole_like
        if Formula is not None:
            self.Formula = Formula
        if fder is not None:
            self.fder = fder
        assert hasattr(
            self, 'fder'), "fder not set -  derivative of fermi distribution . 0: fermi-sea, 1: fermi-surface 2: f''  "
        assert hasattr(self, 'Formula'), "Formula not set - it  should be class with a trace(ik,inn,out) method "
        self.constant_factor = constant_factor
        if self.hole_like and self.fder == 0:
            self.constant_factor *= -1
        if not self.tetra:
            self.extraEf = 0 if self.fder == 0 else 1 if self.fder in (1, 2) else 2 if self.fder == 3 else None
            self.dEF = Efermi[1] - Efermi[0] if len(Efermi) > 1 else 0.001
            self.EFmin = Efermi[0] - self.extraEf * self.dEF
            self.EFmax = Efermi[-1] + self.extraEf * self.dEF
            self.nEF_extra = Efermi.shape[0] + 2 * self.extraEf

        super().__init__(**kwargs)

    def __call__(self, data_K):

        nk = data_K.nk
        NB = data_K.num_wann
        formula = self.Formula(data_K, **self.kwargs_formula)
        ndim = formula.ndim

        # when we do not need k-resolved, we assume as it is only one k-point,m and dump everything there
        if self.k_resolved:
            nk_result = nk
            ik_to_result = lambda ik: ik
        else:
            nk_result = 1
            ik_to_result = lambda ik: 0

        # get a list [{(ib1,ib2):W} for ik in op:ed]
        if self.tetra:
            weights = data_K.tetraWeights.weights_all_band_groups(
                self.Efermi, der=-1 if self.hole_like else self.fder, degen_thresh=self.degen_thresh,
                degen_Kramers=self.degen_Kramers, Emin=self.Emin, Emax=self.Emax)  # here W is array of shape Efermi
        else:
            weights = data_K.get_bands_in_range_groups(
                self.EFmin,
                self.EFmax,
                degen_thresh=self.degen_thresh,
                degen_Kramers=self.degen_Kramers,
                sea=(self.fder == 0),
                Emin=self.Emin,
                Emax=self.Emax
            )  # here W is energy

        #        """formula  - TraceFormula to evaluate
        #           bands = a list of lists of k-points for every
        shape = (3,) * ndim

        lambdadic = lambda: np.zeros(((3,) * ndim), dtype=float)
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
                    out = np.arange(n, NB)
                    _values[n] = formula.trace(ik, inn, out)
                for n in bnd:
                    values[ik][n] = _values[n[1]] - _values[n[0]]

        if self.tetra:
            # tetrahedron method
            restot = np.zeros((nk_result,) + self.Efermi.shape + shape)
            for ik, weights in enumerate(weights):
                valuesik = values[ik]
                for n, w in weights.items():
                    restot[ik_to_result(ik)] += np.einsum("e,...->e...", w, valuesik[n])
        else:
            # no tetrahedron
            restot = np.zeros((nk_result, self.nEF_extra,) + shape)
            for ik, weights in enumerate(weights):
                valuesik = values[ik]
                for n, E in sorted(weights.items()):
                    if E < self.EFmin:
                        restot[ik_to_result(ik)] += valuesik[n][None]
                    elif E <= self.EFmax:
                        iEf = ceil((E - self.EFmin) / self.dEF)
                        restot[ik_to_result(ik), iEf:] += valuesik[n]
            if self.fder == 0:
                pass
            elif self.fder == 1:
                restot = (restot[:, 2:] - restot[:, :-2]) / (2 * self.dEF)
            elif self.fder == 2:
                restot = (restot[:, 2:] + restot[:, :-2] - 2 * restot[:, 1:-1]) / (self.dEF ** 2)
            elif self.fder == 3:
                restot = (restot[:, 4:] - restot[:, :-4] - 2 * (restot[:, 3:-1] - restot[:, 1:-3])) / (
                            2 * self.dEF ** 3)
            else:
                raise NotImplementedError(f"Derivatives  d^{self.fder}f/dE^{self.fder} is not implemented")

        restot /= data_K.cell_volume
        if not self.k_resolved:
            restot /= data_K.nk
        if self.use_factor:
            restot *= self.constant_factor
        else:
            restot *= np.sign(self.constant_factor)

        if self.k_resolved:
            return K__Result([restot], transformTR=formula.transformTR, transformInv=formula.transformInv,
                             rank=restot.ndim - 2,
                             other_properties=dict(
                                 comment=self.comment,
                                 efermi=self.Efermi
                             )
                             )
        else:
            res = EnergyResult(self.Efermi, restot[0], transformTR=formula.transformTR,
                               transformInv=formula.transformInv, smoothers=[self.smoother], comment=self.comment)
            res.set_save_mode(self.save_mode)
        return res

    @property
    def require_energy(self):
        return True


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


####################
# basic quantities #
####################

class _DOS(StaticCalculator):

    def __init__(self, fder, **kwargs):
        self.Formula = frml.Identity
        self.fder = fder
        super().__init__(**kwargs)

    def __call__(self, data_K):
        return super().__call__(data_K) * data_K.cell_volume


class DOS(_DOS):
    r"""Density of states"""

    def __init__(self, **kwargs):
        super().__init__(fder=1, **kwargs)


class CumDOS(_DOS):
    r"""Cumulative density of states"""

    def __init__(self, **kwargs):
        super().__init__(fder=0, **kwargs)


class Spin(StaticCalculator):
    r"""Spin per unit cell (dimensionless)

       | Output: :math:`\int [dk] s f`"""

    def __init__(self, **kwargs):
        self.Formula = frml.Spin
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        return super().__call__(data_K) * data_K.cell_volume


class Morb(StaticCalculator):
    r"""Orbital magnetic moment per unit cell (:math:`\mu_B`)

        | Eq(1) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.85.014435>`__
        | Output: :math:`M = -\int [dk] (H + G - 2E_f \cdot \Omega) f`"""

    def __init__(self, constant_factor=-factors.eV_au / factors.bohr ** 2, **kwargs):
        self.Formula = frml.Morb_Hpm
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)
        self.AHC = AHC(constant_factor=constant_factor, print_comment=False, **kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Omega_res = self.AHC(data_K).mul_array(self.Efermi, axes=0)
        Hplus_res.add(- 2 * Omega_res)
        return Hplus_res * data_K.cell_volume


class Morb_test(Morb):

    def __init__(self, constant_factor=-factors.eV_au / factors.bohr ** 2, **kwargs):
        self.Formula = frml_basic.tildeHGc
        self.fder = 0
        self.comment = r"""Orbital magnetic moment per unit cell for testing (\mu_B)
        Output:
        :math: `M = -\int [dk] (H + G - 2E_f \cdot \Omega) f`"""
        StaticCalculator.__init__(self, constant_factor=constant_factor, **kwargs)
        self.AHC = AHC_test(constant_factor=constant_factor, print_comment=False, **kwargs)


####################
#  conductivities  #
####################

class GME_orb_FermiSurf(StaticCalculator):
    r"""Gyrotropic tensor orbital part (:math:`A`)
        | With Fermi surface integral. Eq(9) `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.077201>`__
        | Output: :math:`K^{orb}_{\alpha :\mu} = \int [dk] v_\alpha m^{orb}_\mu f'`
        | Where :math:`m^{orb} = H + G - 2E_f \cdot \Omega`"""

    def __init__(self, constant_factor=factors.factor_gme * factors.fac_orb_Z, **kwargs):
        self.Formula = frml.VelHplus
        self.fder = 1
        super().__init__(constant_factor=constant_factor, **kwargs)
        self.BerryDipole = BerryDipole_FermiSurf(constant_factor=constant_factor, print_comment=False, **kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Omega_res = self.BerryDipole(data_K).mul_array(self.Efermi)
        return Hplus_res - 2 * Omega_res


class GME_orb_FermiSea(StaticCalculator):
    r"""Gyrotropic tensor orbital part (:math:`A`)
        | With Fermi sea integral. Eq(30) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`__
        | Output: :math:`K^{orb}_{\alpha :\mu} = -\int [dk] \partial_\alpha m_\mu f`
        | Where :math:`m = H + G - 2E_f \cdot \Omega`"""

    def __init__(self, constant_factor=factors.factor_gme * factors.fac_orb_Z, **kwargs):
        self.Formula = frml.DerMorb
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)
        self.BerryDipole = BerryDipole_FermiSea(constant_factor=constant_factor, print_comment=False, **kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Hplus_res.data = Hplus_res.data.swapaxes(1, 2)
        Omega_res = self.BerryDipole(data_K).mul_array(self.Efermi)
        return Hplus_res - 2 * Omega_res


class GME_orb_FermiSea_test(StaticCalculator):
    r"""Gyrotropic tensor orbital part for testing (:math:`A`)
        | With Fermi sea integral.
        | Output: :math:`K^{orb}_{\alpha :\mu} = -\int [dk] \partial_\alpha m_\mu f`
        | Where :math: `m = H + G - 2E_f \cdot \Omega` """

    def __init__(self, constant_factor=factors.factor_gme * factors.fac_orb_Z, **kwargs):
        self.Formula = frml_basic.tildeHGc_d
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)
        self.BerryDipole = BerryDipole_FermiSea_test(constant_factor=constant_factor, print_comment=False, **kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Hplus_res.data = Hplus_res.data.swapaxes(1, 2)
        Omega_res = self.BerryDipole(data_K).mul_array(self.Efermi)
        return Hplus_res - 2 * Omega_res


class GME_spin_FermiSea(StaticCalculator):
    r"""Gyrotropic tensor spin part (:math:`A`)

        | With Fermi sea integral. Eq(30) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`__
        | Output: :math:`K^{spin}_{\alpha :\mu} = -\int [dk] \partial_\alpha s_\mu f`"""

    def __init__(self, constant_factor=factors.factor_gme * factors.fac_spin_Z, **kwargs):
        self.Formula = frml.DerSpin
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class GME_spin_FermiSurf(StaticCalculator):
    r"""Gyrotropic tensor spin part (:math:`A`)

        | With Fermi surface integral. Eq(9) `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.077201>`__
        | Output: :math:`K^{spin}_{\alpha :\mu} = \tau \int [dk] v_\alpha s_\mu f'`"""

    def __init__(self, constant_factor=factors.factor_gme * factors.fac_spin_Z, **kwargs):
        self.Formula = frml.VelSpin
        self.fder = 1
        super().__init__(constant_factor=constant_factor, **kwargs)


# E^1 B^0
class AHC(StaticCalculator):
    r"""Anomalous Hall conductivity (:math:`s^3 \cdot A^2 / (kg \cdot m^3) = S/m`)

        | With Fermi sea integral Eq(11) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`__
        | Output: :math:`O = -e^2/\hbar \int [dk] \Omega f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_\delta E_\beta`"""

    def __init__(self, constant_factor=factors.factor_ahc, **kwargs):
        "describe input parameters here"
        self.Formula = frml.Omega
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)


class AHC_test(StaticCalculator):
    r"""Anomalous Hall conductivity for testing (:math:`s^3 \cdot A^2 / (kg \cdot m^3) = S/m`)

        | Output: :math:`O = - e^2/\hbar \int [dk] \Omega f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_\delta E_\beta`"""

    def __init__(self, constant_factor=factors.factor_ahc, **kwargs):
        self.Formula = frml_basic.tildeFc
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)


class Ohmic_FermiSea(StaticCalculator):
    __doc__ = (r"""Ohmic conductivity (:math:`S/m`)

        | With Fermi sea integral. Eq(31) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`__
        | Output: :math:`\sigma_{\alpha\beta} = e^2/\hbar \tau \int [dk] \partial_\beta v_\alpha f`"""
               fr"for \tau=1{factors.TAU_UNIT_TXT}"
               r"| Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta`")

    def __init__(self, constant_factor=factors.factor_ohmic, **kwargs):
        self.Formula = frml.InvMass
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)


class Ohmic_FermiSurf(StaticCalculator):
    r"""Ohmic conductivity (:math:`S/m`)

        | With Fermi surface integral.
        | Output: :math:`\sigma_{\alpha\beta} = -e^2/\hbar \tau \int [dk] v_\alpha v_\beta f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta`"""

    def __init__(self, constant_factor=factors.factor_ohmic, **kwargs):
        self.Formula = frml.VelVel
        self.fder = 1
        super().__init__(constant_factor=constant_factor, **kwargs)


# E^1 B^1
class Hall_classic_FermiSurf(StaticCalculator):
    r"""Classic Hall conductivity (:math:`S/m/T`)

        | With Fermi surface integral.
        | Output: :math:`\sigma_{\alpha\beta :\mu} = e^3/\hbar^2 \tau^2 \epsilon_{\gamma\mu\rho} \int [dk] v_\alpha \partial_\rho v_\beta v_\gamma f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta :\mu} E_\beta B_\mu`"""

    def __init__(self, constant_factor=factors.factor_hall_classic, **kwargs):
        self.Formula = frml.VelMassVel
        self.fder = 1
        super().__init__(constant_factor=constant_factor, **kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
        res.data = 0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
        res.rank -= 2
        return res


class Hall_classic_FermiSea(StaticCalculator):
    r"""Classic Hall conductivity (:math:`S/m/T`)

        | With Fermi sea integral.
        | Output: :math:`\sigma_{\alpha\beta :\mu} = -e^3/\hbar^2 \tau^2 \epsilon_{\gamma\mu\rho} \int [dk] \partial_\gamma v_\alpha \partial_\rho v_\beta f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta :\mu} E_\beta B_\mu`"""

    def __init__(self, constant_factor=factors.factor_hall_classic, **kwargs):
        self.Formula = frml.MassMass
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        res.data = res.data.transpose(0, 4, 1, 2, 3)
        res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
        res.data = 0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
        res.rank -= 2
        return res


# E^2 B^0
class BerryDipole_FermiSurf(StaticCalculator):
    r"""Berry curvature dipole (dimensionless)

        | With Fermi surface integral. Eq(8) in `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.216806>`__
        | Output: :math:`D_{\beta\delta} = -\int [dk] v_\beta \Omega_\delta f'`"""

    def __init__(self, **kwargs):
        self.Formula = frml.VelOmega
        self.fder = 1
        super().__init__(**kwargs)


class NLAHC_FermiSurf(BerryDipole_FermiSurf):
    r"""Nonlinear anomalous Hall conductivity (:math:`S^2/A`)

        | With Fermi surface integral. Eq(8) in `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.216806>`__
        | Output: :math:`D_{\beta\delta} = -e^3/\hbar^2 \tau \int [dk] v_\beta \Omega_\delta f'`
        | Instruction: :math:`j_\alpha = \epsilon_{\alpha\delta\gamma} D_{\beta\delta} E_\beta E\gamma`"""

    def __init__(self, constant_factor=factors.factor_nlahc, **kwargs):
        super().__init__(constant_factor=constant_factor, **kwargs)


class BerryDipole_FermiSea(StaticCalculator):
    r"""Berry curvature dipole (dimensionless)

        | With Fermi sea integral. Eq(29) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`__
        | Output: :math:`D_{\beta\delta} = \int [dk] \partial_\beta \Omega_\delta f`"""

    def __init__(self, **kwargs):
        self.Formula = frml.DerOmega
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class NLAHC_FermiSea(BerryDipole_FermiSea):
    r"""Nonlinear anomalous Hall conductivity  (:math:`S^2/A`)

        | With Fermi sea integral. Eq(29) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`__
        | Output: :math:`D_{\beta\delta} = e^3/\hbar^2 \tau \int [dk] \partial_\beta \Omega_\delta f`
        | Instruction: :math:`j_\alpha = \epsilon_{\alpha\delta\gamma} D_{\beta\delta} E_\beta E\gamma`"""

    def __init__(self, constant_factor=factors.factor_nlahc, **kwargs):
        super().__init__(constant_factor=constant_factor, **kwargs)


class BerryDipole_FermiSea_test(StaticCalculator):
    r"""Berry curvature dipole for testing (dimensionless)

        | With Fermi sea integral.
        | Output: :math:`D_{\beta\delta} = \tau \int [dk] \partial_beta \Omega_\delta f`"""

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeFc_d
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class NLDrude_FermiSea(StaticCalculator):
    r"""Drude conductivity (:math:`S^2/A`)

        | With Fermi sea integral. Eq(3) in `Ref <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043081>`__
        | Output: :math:`\sigma_{\alpha\beta\gamma} = -e^3/\hbar^2 \tau^2 \int [dk] \partial_{\beta\gamma} v_\alpha f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""

    def __init__(self, constant_factor=factors.factor_nldrude, **kwargs):
        self.Formula = frml.Der3E
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)


class NLDrude_FermiSurf(StaticCalculator):
    r"""Drude conductivity (:math:`S^2/A`)

        | With Fermi surface integral.
        | Output: :math:`\sigma_{\alpha\beta\gamma} = e^3/\hbar^2 \tau^2 \int [dk] \partial_\beta v_\alpha v_\gamma f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""

    def __init__(self, constant_factor=factors.factor_nldrude, **kwargs):
        self.Formula = frml.MassVel
        self.fder = 1
        super().__init__(constant_factor=constant_factor, **kwargs)


class NLDrude_Fermider2(StaticCalculator):
    r"""Drude conductivity (:math:`S^2/A`)

        | With second derivative of distribution function. Eq(A28) in `Ref <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043081>`__
        | Output: :math:`\sigma_{\alpha\beta\gamma} = -e^3/\hbar^2 \tau^2 \int [dk] v_\beta v_\alpha v_\gamma f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""

    def __init__(self, constant_factor=factors.factor_nldrude / 2, **kwargs):
        self.Formula = frml.VelVelVel
        self.fder = 2
        super().__init__(constant_factor=constant_factor, **kwargs)


class SHC(StaticCalculator):
    r"""Spin Hall conductivity with dc (:math:`S/m`)

        | With Fermi sea integral. Eq(1) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`_
        | Qiao type : with kwargs_formula={'spin_current_type':'qiao'}. Eq(49,50) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.214402>`__
        | Ryoo type : with kwargs_formula={'spin_current_type':'ryoo'}. Eq(17,26-29) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`__
        | Output: :math:`\sigma_{\alpha\beta} = -e^2/\hbar \int [dk] Im(j_{nm,\alpha} v_{mn,\beta})/(\epsilon_n - \epsilon_m)^2 f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta`"""

    def __init__(self, constant_factor=-factors.factor_ahc / 2, **kwargs):
        self.Formula = frml.SpinOmega
        self.fder = 0
        super().__init__(constant_factor=constant_factor, **kwargs)


# E^1 B^1
class AHC_Zeeman_spin(StaticCalculator):
    r"""AHC conductivity Zeeman correcton term spin part (:math:`S/m/T`)

        | With Fermi surface integral.
        | Output: :math:`ZAHC^{spin}_{\alpha\beta :\mu} = e^2/\hbar \int [dk] \Omega_\delta * s_\mu f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta :\mu} E_\beta B_\mu = \epsilon_{\alpha\beta\delta} ZAHC^{spin}_{\alpha\beta:\mu} E_\beta B_\mu` """

    def __init__(self, constant_factor=factors.fac_spin_Z * factors.factor_ahc, **kwargs):
        self.Formula = frml.OmegaS
        self.fder = 1
        super().__init__(constant_factor=constant_factor, **kwargs)


class OmegaOmega(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.OmegaOmega
        self.fder = 1
        super().__init__(**kwargs)


class AHC_Zeeman_orb(StaticCalculator):
    r"""AHC conductivity Zeeman correction term orbital part (:math:`S/m/T`)

        | With Fermi surface integral.
        | Output: :math:`ZAHC^{orb}_{\alpha\beta :\mu} = e^2/\hbar \int [dk] \Omega_\delta * m_\mu f'`
        | Where :math:`m = H + G - 2Ef*\Omega`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta :\mu} E_\beta B_\mu = e \epsilon_{\alpha\beta\delta} ZAHC^{orb}_{\alpha\beta:\mu} E_\beta B_\mu`"""

    def __init__(self, constant_factor=factors.fac_orb_Z * factors.factor_ahc, **kwargs):
        self.Formula = frml.OmegaHplus
        self.fder = 1
        super().__init__(constant_factor=constant_factor, **kwargs)
        self.OmegaOmega = OmegaOmega(constant_factor=constant_factor, print_comment=False, **kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Omega_res = self.OmegaOmega(data_K).mul_array(self.Efermi)
        return Hplus_res - 2 * Omega_res
