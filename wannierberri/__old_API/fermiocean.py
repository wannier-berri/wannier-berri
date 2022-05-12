import numpy as np
from wannierberri.__utility import alpha_A, beta_A
from wannierberri import __factors  as factors
from collections import defaultdict
from wannierberri import result as result
from math import ceil
from wannierberri.formula import FormulaProduct
from wannierberri.formula import covariant as frml
from wannierberri.formula import covariant_basic as frml_basic


def cumdos(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Identity(), data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * data_K.cell_volume


def dos(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Identity(), data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * data_K.cell_volume


def Hall_classic_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    formula = FormulaProduct(
        [data_K.covariant('Ham', commader=1),
         frml.InvMass(data_K),
         data_K.covariant('Ham', commader=1)],
        name='vel-mass-vel')
    res = FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factors.factor_Hall_classic
    res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
    res.data = -0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
    res.rank -= 2
    return res


def Hall_classic(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""sigma11tau2 """
    formula1 = FormulaProduct([frml.InvMass(data_K), frml.InvMass(data_K)], name='mass-mass')
    formula2 = FormulaProduct([data_K.covariant('Ham', commader=1), frml.Der3E(data_K)], name='vel-Der3E')
    res = FermiOcean(
        formula1, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factors.factor_Hall_classic
    term2 = FermiOcean(
        formula2, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factors.factor_Hall_classic
    res.data = res.data.transpose(0, 4, 1, 2, 3) + term2.data.transpose(0, 4, 2, 3, 1)
    res.data = res.data[:, :, :, beta_A, alpha_A] - res.data[:, :, :, alpha_A, beta_A]
    res.data = -0.5 * (res.data[:, alpha_A, beta_A, :] - res.data[:, beta_A, alpha_A, :])
    res.rank -= 2
    return res


def Hall_morb_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    # first, transform to SI, not forgettint e/2hbar multilier for morb - now in A*m/J,
    # restoring the sign of spin magnetic moment
    factor = -factors.Ang_SI * factors.elementary_charge / (2 * factors.hbar)
    factor *= factors.elementary_charge**2 / factors.hbar  # multiply by a dimensional factor - now in S/(T*m)
    factor *= -1
    #factor *= 1e-2  #  finally transform to S/(T*cm)
    formula_1 = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         frml.Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='berry-morb_Hpm')
    formula_2 = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         frml.Omega(data_K, **kwargs_formula)], name='berry-berry')
    res = FermiOcean(formula_1, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    res += -2 * FermiOcean(
        formula_2, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)().mul_array(Efermi)
    return res * factor


def Hall_spin_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    # first, transform to SI - now in 1/(m*T) ,restoring the sign of spin magnetic moment
    factor = -factors.bohr_magneton / (factors.elementary_charge * factors.Ang_SI)
    factor *= -1
    factor *= factors.elementary_charge**2 / factors.hbar  # multiply by a dimensional factor - now in S/(T*m)
    #factor *= 1e-2  #  finally transform to S/(T*cm)
    formula = FormulaProduct([frml.Omega(data_K, **kwargs_formula), frml.Spin(data_K)], name='berry-spin')
    return FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factor


def AHC(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Omega(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factors.fac_ahc


def AHC_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    res = FermiOcean(
        frml_basic.tildeFc(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    return res * factors.fac_ahc


def spin(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.Spin(data_K), data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()


def berry_dipole_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    formula = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         data_K.covariant('Ham', commader=1)], name='berry-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def berry_dipole(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" sigma20tau1"""
    res = FermiOcean(
        frml.DerOmega(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def berry_dipole_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" sigma20tau1"""
    res = FermiOcean(
        frml_basic.tildeFc_d(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def Hplus_der(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    res = FermiOcean(
        frml.DerMorb(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def Hplus_der_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    res = FermiOcean(
        frml_basic.tildeHGc_d(data_K, sign=+1, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2)
    return res


def gme_orb_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    formula_1 = FormulaProduct(
        [frml.Morb_Hpm(data_K, sign=+1, **kwargs_formula),
         data_K.covariant('Ham', commader=1)], name='morb_Hpm-vel')
    formula_2 = FormulaProduct(
        [frml.Omega(data_K, **kwargs_formula),
         data_K.covariant('Ham', commader=1)], name='berry-vel')
    res = FermiOcean(formula_1, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    res += -2 * FermiOcean(
        formula_2, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)().mul_array(Efermi)
    # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factors.factor_gme * factors.fac_orb_Z#-factors.elementary_charge**2 / (2 * factors.hbar)
    return res


def gme_orb(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    Hp = Hplus_der(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    D = berry_dipole(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    #tensor_K = -factors.elementary_charge**2 / (2 * factors.hbar) * (Hp - 2 * Efermi[:, None, None] * D)
    tensor_K = factors.factor_gme * factors.fac_orb_Z * (Hp - 2 * Efermi[:, None, None] * D)
    return result.EnergyResult(Efermi, tensor_K, TRodd=False, Iodd=True)


def gme_orb_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    Hp = Hplus_der_test(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    D = berry_dipole_test(data_K, Efermi, tetra=tetra, **kwargs_formula).data
    #tensor_K = -factors.elementary_charge**2 / (2 * factors.hbar) * (Hp - 2 * Efermi[:, None, None] * D)
    tensor_K = factors.factor_gme * factors.fac_orb_Z * (Hp - 2 * Efermi[:, None, None] * D)
    return result.EnergyResult(Efermi, tensor_K, TRodd=False, Iodd=True)


def gme_spin_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    formula = FormulaProduct([frml.Spin(data_K), data_K.covariant('Ham', commader=1)], name='spin-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factors.factor_gme * factors.fac_spin_Z #-factors.bohr_magneton / factors.Ang_SI**2
    return res


def gme_spin(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    formula = FormulaProduct([frml.DerSpin(data_K)], name='derspin')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    # swap axes to be consistent with the eq. (30) of DOI:10.1038/s41524-021-00498-5
    res.data = np.swapaxes(res.data, 1, 2) * factors.factor_gme * factors.fac_spin_Z#-factors.bohr_magneton / factors.Ang_SI**2
    return res


def Morb(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    fac_morb = -factors.eV_au / factors.bohr**2
    return (
        FermiOcean(
            frml.Morb_Hpm(data_K, sign=+1, **kwargs_formula),
            data_K,
            Efermi,
            tetra,
            fder=0,
            degen_thresh=degen_thresh,
            degen_Kramers=degen_Kramers)() - 2 * FermiOcean(
                frml.Omega(data_K, **kwargs_formula),
                data_K,
                Efermi,
                tetra,
                fder=0,
                degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers)().mul_array(Efermi)) * (data_K.cell_volume * fac_morb)


def Morb_test(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    fac_morb = -factors.eV_au / factors.bohr**2
    return (
        FermiOcean(
            frml_basic.tildeHGc(data_K, sign=+1, **kwargs_formula),
            data_K,
            Efermi,
            tetra,
            fder=0,
            degen_thresh=degen_thresh,
            degen_Kramers=degen_Kramers)() - 2 * FermiOcean(
                frml_basic.tildeFc(data_K, **kwargs_formula),
                data_K,
                Efermi,
                tetra,
                fder=0,
                degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers)().mul_array(Efermi)) * (data_K.cell_volume * fac_morb)


def ohmic_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    velocity = data_K.covariant('Ham', commader=1)
    formula = FormulaProduct([velocity, velocity], name='vel-vel')
    return FermiOcean(
        formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factors.factor_ohmic


def ohmic(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r""" sigma10tau1"""
    formula = frml.InvMass(data_K)
    return FermiOcean(
        formula, data_K, Efermi, tetra, fder=0, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factors.factor_ohmic


def Der3E(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""sigma20tau2 f0 """
    res = FermiOcean(
        frml.Der3E(data_K, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factors.factor_nldrude
    return res


def Der3E_fsurf(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""sigma20tau2 first der f0 """
    formula = FormulaProduct([frml.InvMass(data_K), data_K.covariant('Ham', commader=1)], name='mass-vel')
    res = FermiOcean(formula, data_K, Efermi, tetra, fder=1, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)() * factors.factor_nldrude
    return res


def Der3E_fder2(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    r"""sigma20tau2 second der f0 """
    formula = FormulaProduct(
        [data_K.covariant('Ham', commader=1),
         data_K.covariant('Ham', commader=1),
         data_K.covariant('Ham', commader=1)],
        name='vel-vel-vel')
    res = factors.factor_nldrude * FermiOcean(
        formula, data_K, Efermi, tetra, fder=2, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers)()
    return res


def spin_hall(data_K, Efermi, spin_current_type, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return FermiOcean(
        frml.SpinOmega(data_K, spin_current_type, **kwargs_formula),
        data_K,
        Efermi,
        tetra,
        fder=0,
        degen_thresh=degen_thresh,
        degen_Kramers=degen_Kramers)() * factors.fac_spin_hall


def spin_hall_qiao(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return spin_hall(data_K, Efermi, "qiao", tetra=tetra, **kwargs_formula)


def spin_hall_ryoo(data_K, Efermi, tetra=False, degen_thresh=1e-4, degen_Kramers=False, **kwargs_formula):
    return spin_hall(data_K, Efermi, "ryoo", tetra=tetra, **kwargs_formula)


###############################
# The private part goes here  #
###############################


class FermiOcean():
    """ formula should have a trace(ik,inn,out) method
    fder derivative of fermi distribution . 0: fermi-sea, 1: fermi-surface 2: f''
    """

    def __init__(self, formula, data_K, Efermi, tetra, fder, degen_thresh=1e-4, degen_Kramers=False):

        ndim = formula.ndim
        self.Efermi = Efermi
        self.fder = fder
        self.tetra = tetra
        self.nk = data_K.nk
        self.NB = data_K.num_wann
        self.formula = formula
        self.final_factor = 1. / (data_K.nk * data_K.cell_volume)

        # get a list [{(ib1,ib2):W} for ik in op:ed]
        if self.tetra:
            self.weights = data_K.tetraWeights.weights_all_band_groups(
                Efermi, der=self.fder, degen_thresh=degen_thresh,
                degen_Kramers=degen_Kramers)  # here W is array of shape Efermi
        else:
            self.extraEf = 0 if fder == 0 else 1 if fder in (1, 2) else 2 if fder == 3 else None
            self.dEF = Efermi[1] - Efermi[0]
            self.EFmin = Efermi[0] - self.extraEf * self.dEF
            self.EFmax = Efermi[-1] + self.extraEf * self.dEF
            self.nEF_extra = Efermi.shape[0] + 2 * self.extraEf
            self.weights = data_K.get_bands_in_range_groups(
                self.EFmin, self.EFmax, degen_thresh=degen_thresh, degen_Kramers=degen_Kramers,
                sea=(self.fder == 0))  # here W is energy
        self.__evaluate_traces(formula, self.weights, ndim)

    def __evaluate_traces(self, formula, bands, ndim):
        """formula  - TraceFormula to evaluate
           bands = a list of lists of k-points for every
        """
        self.shape = (3, ) * ndim
        lambdadic = lambda: np.zeros(((3, ) * ndim), dtype=float)
        self.values = [defaultdict(lambdadic) for ik in range(self.nk)]
        for ik, bnd in enumerate(bands):
            if formula.additive:
                for n in bnd:
                    inn = np.arange(n[0], n[1])
                    out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], self.NB)))
                    self.values[ik][n] = formula.trace(ik, inn, out)
            else:
                nnall = set([_ for n in bnd for _ in n])
                _values = {}
                for n in nnall:
                    inn = np.arange(0, n)
                    out = np.arange(n, self.NB)
                    _values[n] = formula.trace(ik, inn, out)
                for n in bnd:
                    self.values[ik][n] = _values[n[1]] - _values[n[0]]

    def __call__(self):
        if self.tetra:
            res = self.__call_tetra()
        else:
            res = self.__call_notetra()
        res *= self.final_factor
        return result.EnergyResult(self.Efermi, res, TRodd=self.formula.TRodd, Iodd=self.formula.Iodd)

    def __call_tetra(self):
        restot = np.zeros(self.Efermi.shape + self.shape)
        for ik, weights in enumerate(self.weights):
            values = self.values[ik]
            for n, w in weights.items():
                restot += np.einsum("e,...->e...", w, values[n])
        return restot

    def __call_notetra(self):
        restot = np.zeros((self.nEF_extra, ) + self.shape)
        for ik, weights in enumerate(self.weights):
            values = self.values[ik]
            for n, E in sorted(weights.items()):
                if E < self.EFmin:
                    restot += values[n][None]
                elif E <= self.EFmax:
                    iEf = ceil((E - self.EFmin) / self.dEF)
                    restot[iEf:] += values[n]
        if self.fder == 0:
            return restot
        if self.fder == 1:
            return (restot[2:] - restot[:-2]) / (2 * self.dEF)
        elif self.fder == 2:
            return (restot[2:] + restot[:-2] - 2 * restot[1:-1]) / (self.dEF**2)
        elif self.fder == 3:
            return (restot[4:] - restot[:-4] - 2 * (restot[3:-1] - restot[1:-3])) / (2 * self.dEF**3)
        else:
            raise NotImplementedError(f"Derivatives  d^{self.fder}f/dE^{self.fder} is not implemented")
