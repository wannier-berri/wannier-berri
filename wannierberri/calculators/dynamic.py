import numpy as np
import abc, functools
from wannierberri.__utility import Gaussian, Lorentzian
from wannierberri.result import EnergyResult
from . import Calculator
from wannierberri.formula.covariant import SpinVelocity


#######################################
#                                     #
#      integration (Efermi-omega)     #
#                                     #
#######################################


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
        return EnergyResult(
            [self.Efermi, self.omega], restot, TRodd=formula.TRodd, Iodd=formula.Iodd, TRtrans=formula.TRtrans)






# auxillary function"
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

from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

from wannierberri import __factors as factors


###############################
#              JDOS           #
###############################

class Formula_dyn_ident():

    def __init__(self, data_K):
        self.TRodd = False
        self.Iodd = False
        self.TRtrans = False
        self.ndim = 0

    def trace_ln(self, ik, inn1, inn2):
        return len(inn1) * len(inn2)


class JDOS(DynamicCalculator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = self.smr_fixed_width
        self.Formula = Formula_dyn_ident
        self.dtype = float

    def nonzero(self, E1, E2):
        return (E1 < self.Efermi.max()) and (E2 > self.Efermi.min()) and (
            self.omega.min() - 5 * self.smr_fixed_width < E2 - E1 < self.omega.max() + 5 * self.smr_fixed_width)

    def energy_factor(self, E1, E2):
        res = np.zeros((len(self.Efermi), len(self.omega)))
        gauss = self.smear(E2 - E1 - self.omega, self.smr_fixed_width)
        res[(E1 < self.Efermi) * (self.Efermi < E2)] = gauss[None, :]
        return res


################################
#    Optical Conductivity      #
################################


class Formula_OptCond():

    def __init__(self, data_K):
        A = data_K.A_H
        self.AA = 1j * A[:, :, :, :, None] * A.swapaxes(1, 2)[:, :, :, None, :]
        self.ndim = 2
        self.TRodd = False
        self.Iodd = False
        self.TRtrans = True

    def trace_ln(self, ik, inn1, inn2):
        return self.AA[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class OpticalConductivity(DynamicCalculator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Formula = Formula_OptCond
        self.final_factor = factors.factor_opt

    def factor_omega(self, E1, E2):
        delta_arg_12 = E2 - E1 - self.omega  # argument of delta function [iw, n, m]
        cfac = 1. / (delta_arg_12 - 1j * self.smr_fixed_width)
        if self.smr_type != 'Lorentzian':
            cfac.imag = np.pi * self.smear(delta_arg_12)
        return (E2 - E1) * cfac


###############################
#          SHC                #
###############################


class Formula_SHC():

    def __init__(self, data_K, SHC_type='ryoo', shc_abc=None):
        A = SpinVelocity(data_K, SHC_type).matrix
        B = -1j * data_K.A_H
        self.imAB = np.imag(A[:, :, :, :, None, :] * B.swapaxes(1, 2)[:, :, :, None, :, None])
        self.ndim = 3
        if shc_abc is not None:
            assert len(shc_abc) == 3
            a, b, c = (x - 1 for x in shc_abc)
            self.imAB = self.imAB[:, :, :, a, b, c]
            self.ndim = 0
        self.TRodd = False
        self.Iodd = False
        self.TRtrans = False

    def trace_ln(self, ik, inn1, inn2):
        return self.imAB[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class _SHC(DynamicCalculator):

    def __init__(self, SHC_type="ryoo", shc_abc=None, **kwargs):
        super().__init__(**kwargs)
        self.formula_kwargs = dict(SHC_type=SHC_type, shc_abc=shc_abc)
        self.Formula = Formula_SHC
        self.final_factor = factors.factor_shc

    def factor_omega(self, E1, E2):
        delta_minus = self.smear(E2 - E1 - self.omega)
        delta_plus = self.smear(E1 - E2 - self.omega)
        cfac2 = delta_plus - delta_minus  # TODO : for Lorentzian do the real and imaginary parts together
        cfac1 = np.real((E1 - E2) / ((E1 - E2)**2 - (self.omega + 1j * self.smr_fixed_width)**2))
        cfac = (2 * cfac1 + 1j * np.pi * cfac2) / 4.
        return cfac


class SHC(_SHC):
    "a more laconic implementation of the energy factor"

    def factor_omega(self, E1, E2):
        delta_arg_12 = E1 - E2 - self.omega  # argument of delta function [iw, n, m]
        cfac = 1. / (delta_arg_12 - 1j * self.smr_fixed_width)
        if self.smr_type != 'Lorentzian':
            cfac.imag = np.pi * self.smear(delta_arg_12)
        return cfac / 2
