from .classes import StaticCalculator
from wannierberri import covariant_formulak as frml
from wannierberri import covariant_formulak_basic as frml_basic
import numpy as np
from wannierberri import __factors as factors
from ..__utility import alpha_A, beta_A

#######################################
#                                     #
#         integration (Efermi-only)   #
#                                     #
#######################################

#  TODO: Ideally, a docstring of every calculator should contain the equation that it implements
#        and references (with urls) to the relevant papers



####################
# basic quantities #
####################

class _DOS(StaticCalculator):

    def __init__(self, fder,**kwargs):
        self.Formula = frml.Identity
        self.factor = 1
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
        self.factor = 1
        self.fder = 0
        super().__init__(**kwargs)


class Morb(StaticCalculator):
    r"""Orbital magnetic moment per unit cell (:math:`\mu_B`)

        | Eq(1) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.85.014435>`_
        | Output: :math:`M = -\int [dk] (H + G - 2E_f \cdot \Omega) f`"""

    def __init__(self, **kwargs):
        self.Formula = frml.Morb_Hpm
        self.factor = 1
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Omega_res = AHC(Efermi=self.Efermi, tetra=self.tetra, smoother=self.smoother,
                use_factor=False, print_comment=False,
                kwargs_formula=self.kwargs_formula)(data_K).mul_array(self.Efermi)
        final_factor = -factors.eV_au / factors.bohr**2
        if not self.use_factor:
            final_factor = np.sign(final_factor)
        #with use_factor=False, the factor of AHC is -1, so it is '+' below.
        return final_factor * data_K.cell_volume * (Hplus_res + 2 * Omega_res)


class Morb_test(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeHGc
        self.factor = 1
        self.fder = 0
        self.comment = r"""Orbital magnetic moment per unit cell for testing (\mu_B)
        Output:
        :math: `M = -\int [dk] (H + G - 2E_f \cdot \Omega) f`"""
        super().__init__(**kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Omega_res = AHC_test(Efermi=self.Efermi, tetra=self.tetra, smoother=self.smoother,
                use_factor=False, print_comment=False,
                kwargs_formula=self.kwargs_formula)(data_K).mul_array(self.Efermi)
        final_factor = -factors.eV_au / factors.bohr**2
        if not self.use_factor:
            final_factor = np.sign(final_factor)
        #with use_factor=False, the factor of AHC is -1, so it is '+' below.
        return final_factor * data_K.cell_volume * (Hplus_res + 2 * Omega_res)


####################
#  cunductivities  #
####################

# GME tensor
class GME_orb_FermiSurf(StaticCalculator):
    r"""Gyrotropic tensor orbital part (:math:`A`)
        | With Fermi surface integral. Eq(9) `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.077201>`_
        | Output: :math:`K^{orb}_{\alpha :\mu} = \int [dk] v_\alpha m^{orb}_\mu f'`
        | Where :math:`m^{orb} = H + G - 2E_f \cdot \Omega`"""

    def __init__(self, **kwargs):
        self.Formula = frml.VelHplus
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Omega_res = BerryDipole_FermiSurf(Efermi=self.Efermi, tetra=self.tetra,
                smoother=self.smoother,use_factor=False, print_comment=False,
                kwargs_formula=self.kwargs_formula)(data_K).mul_array(self.Efermi)
        final_factor = factors.factor_gme * factors.fac_orb_Z
        if not self.use_factor:
            final_factor = np.sign(final_factor)

        return final_factor * (Hplus_res - 2 * Omega_res)


class GME_orb_FermiSea(StaticCalculator):
    r"""Gyrotropic tensor orbital part (:math:`A`)
        | With Fermi sea integral. Eq(30) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`_
        | Output: :math:`K^{orb}_{\alpha :\mu} = -\int [dk] \partial_\alpha m_\mu f`
        | Where :math:`m = H + G - 2E_f \cdot \Omega`"""

    def __init__(self, **kwargs):
        self.Formula = frml.DerMorb
        self.factor = 1
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Hplus_res.data = Hplus_res.data.swapaxes(1, 2)
        Omega_res = BerryDipole_FermiSea(Efermi=self.Efermi, tetra=self.tetra,
                smoother=self.smoother,use_factor=False, print_comment=False,
                kwargs_formula=self.kwargs_formula)(data_K).mul_array(self.Efermi)
        final_factor = factors.factor_gme * factors.fac_orb_Z
        if not self.use_factor:
            final_factor = np.sign(final_factor)

        return final_factor * (Hplus_res - 2 * Omega_res)


class GME_orb_FermiSea_test(StaticCalculator):
    r"""Gyrotropic tensor orbital part for testing (:math:`A`)
        | With Fermi sea integral.
        | Output: :math:`K^{orb}_{\alpha :\mu} = -\int [dk] \partial_\alpha m_\mu f`
        |Where :math: `m = H + G - 2E_f \cdot \Omega`"""

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeHGc_d
        self.factor = 1
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Hplus_res.data = Hplus_res.data.swapaxes(1, 2)
        Omega_res = BerryDipole_FermiSea_test(Efermi=self.Efermi, tetra=self.tetra,
                smoother=self.smoother,use_factor=False, print_comment=False,
                kwargs_formula=self.kwargs_formula)(data_K).mul_array(self.Efermi)
        final_factor = factors.factor_gme * factors.fac_orb_Z
        if not self.use_factor:
            final_factor = np.sign(final_factor)

        return final_factor * (Hplus_res - 2 * Omega_res)








class GME_spin_FermiSea(StaticCalculator):
    r"""Gyrotropic tensor spin part (:math:`A`)

        | With Fermi sea integral. Eq(30) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`_
        | Output: :math:`K^{spin}_{\alpha :\mu} = -\int [dk] \partial_\alpha s_\mu f`"""

    def __init__(self, **kwargs):
        self.Formula = frml.DerSpin
        self.factor = factors.factor_gme * factors.fac_spin_Z
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class GME_spin_FermiSurf(StaticCalculator):
    r"""Gyrotropic tensor spin part (:math:`A`)

        | With Fermi surface integral. Eq(9) `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.077201>`_
        | Output: :math:`K^{spin}_{\alpha :\mu} = \tau \int [dk] v_\alpha s_\mu f'`"""

    def __init__(self, **kwargs):
        self.Formula = frml.VelSpin
        self.factor = factors.factor_gme * factors.fac_spin_Z
        self.fder = 1
        super().__init__(**kwargs)


# E^1 B^0
class AHC(StaticCalculator):
    r"""Anomalous Hall conductivity (:math:`s^3 \cdot A^2 / (kg \cdot m^3) = S/m`)

        | With Fermi sea integral Eq(11) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`_
        | Output: :math:`O = -e^2/\hbar \int [dk] \Omega f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_\delta E_\beta`"""

    def __init__(self, **kwargs):
        "describe input parameters here"
        self.Formula = frml.Omega
        self.factor = factors.factor_ahc
        self.fder = 0
        super().__init__(**kwargs)


class AHC_test(StaticCalculator):
    r"""Anomalous Hall conductivity for testing (:math:`s^3 \cdot A^2 / (kg \cdot m^3) = S/m`)

        | Output: :math:`O = - e^2/\hbar \int [dk] \Omega f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta = \epsilon_{\alpha\beta\delta} O_\delta E_\beta`"""

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeFc
        self.factor = factors.factor_ahc
        self.fder = 0
        super().__init__(**kwargs)


class Ohmic_FermiSea(StaticCalculator):
    __doc__ = (r"""Ohmic conductivity (:math:`S/m`)

        | With Fermi sea integral. Eq(31) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`_
        | Output: :math:`\sigma_{\alpha\beta} = e^2/\hbar \tau \int [dk] \partial_\beta v_\alpha f`"""+
        fr"for \tau=1{factors.TAU_UNIT_TXT}"+
        r"""| Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta`""")

    def __init__(self, **kwargs):
        self.Formula = frml.InvMass
        self.factor = factors.factor_ohmic
        self.fder = 0
        super().__init__(**kwargs)


class Ohmic_FermiSurf(StaticCalculator):
    r"""Ohmic conductivity (:math:`S/m`)

        | With Fermi surface integral.
        | Output: :math:`\sigma_{\alpha\beta} = -e^2/\hbar \tau \int [dk] v_\alpha v_\beta f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta`"""

    def __init__(self, **kwargs):
        self.Formula = frml.VelVel
        self.factor = factors.factor_ohmic
        self.fder = 1
        super().__init__(**kwargs)


# E^1 B^1
class Hall_classic_FermiSurf(StaticCalculator):
    r"""Classic Hall conductivity (:math:`S/m/T`)

        | With Fermi surface integral.
        | Output: :math:`\sigma_{\alpha\beta :\mu} = e^3/\hbar^2 \tau^2 \epsilon_{\gamma\mu\rho} \int [dk] v_\alpha \partial_\rho v_\beta v_\gamma f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta :\mu} E_\beta B_\mu`"""

    def __init__(self, **kwargs):
        self.Formula = frml.VelMassVel
        self.factor = factors.factor_hall_classic
        self.fder = 1
        super().__init__(**kwargs)

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

    def __init__(self, **kwargs):
        self.Formula = frml.MassMass
        self.factor = factors.factor_hall_classic
        self.fder = 0
        super().__init__(**kwargs)

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

        | With Fermi surface integral. Eq(8) in `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.216806>`_
        | Output: :math:`D_{\beta\delta} = -\int [dk] v_\beta \Omega_\delta f'`"""

    def __init__(self, **kwargs):
        self.Formula = frml.VelOmega
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)


class NLAHC_FermiSurf(BerryDipole_FermiSurf):
    r"""Nonlinear anomalous Hall conductivity (:math:`S^2/A`)

        | With Fermi surface integral. Eq(8) in `Ref <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.115.216806>`_
        | Output: :math:`D_{\beta\delta} = -e^3/\hbar^2 \tau \int [dk] v_\beta \Omega_\delta f'`
        | Instruction: :math:`j_\alpha = \epsilon_{\alpha\delta\gamma} D_{\beta\delta} E_\beta E\gamma`"""

    def __init__(self, **kwargs):
        self.factor = factors.factor_nlahc
        super().__init__(**kwargs)


class BerryDipole_FermiSea(StaticCalculator):
    r"""Berry curvature dipole (dimensionless)

        | With Fermi sea integral. Eq(29) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`_
        | Output: :math:`D_{\beta\delta} = \int [dk] \partial_\beta \Omega_\delta f`"""

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


class NLAHC_FermiSea(BerryDipole_FermiSea):
    r"""Nonlinear anomalous Hall conductivity  (:math:`S^2/A`)

        | With Fermi sea integral. Eq(29) in `Ref <https://www.nature.com/articles/s41524-021-00498-5>`_
        | Output: :math:`D_{\beta\delta} = e^3/\hbar^2 \tau \int [dk] \partial_\beta \Omega_\delta f`
        | Instruction: :math:`j_\alpha = \epsilon_{\alpha\delta\gamma} D_{\beta\delta} E_\beta E\gamma`"""

    def __init__(self,**kwargs):
        self.factor = factors.factor_nlahc
        super().__init__(**kwargs)


class BerryDipole_FermiSea_test(StaticCalculator):
    r"""Berry curvature dipole for testing (dimensionless)

        | With Fermi sea integral.
        | Output: :math:`D_{\beta\delta} = \tau \int [dk] \partial_beta \Omega_\delta f`"""

    def __init__(self, **kwargs):
        self.Formula = frml_basic.tildeFc_d
        self.factor = 1
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res


class NLDrude_FermiSea(StaticCalculator):
    r"""Drude conductivity (:math:`S^2/A`)

        | With Fermi sea integral. Eq(3) in `Ref <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043081>`_
        | Output: :math:`\sigma_{\alpha\beta\gamma} = -e^3/\hbar^2 \tau^2 \int [dk] \partial_{\beta\gamma} v_\alpha f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""

    def __init__(self, **kwargs):
        self.Formula = frml.Der3E
        self.factor = factors.factor_nldrude
        self.fder = 0
        super().__init__(**kwargs)


class NLDrude_FermiSurf(StaticCalculator):
    r"""Drude conductivity (:math:`S^2/A`)

        | With Fermi surface integral.
        | Output: :math:`\sigma_{\alpha\beta\gamma} = e^3/\hbar^2 \tau^2 \int [dk] \partial_\beta v_\alpha v_\gamma f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""

    def __init__(self, **kwargs):
        self.Formula = frml.MassVel
        self.factor = factors.factor_nldrude
        self.fder = 1
        super().__init__(**kwargs)


class NLDrude_Fermider2(StaticCalculator):
    r"""Drude conductivity (:math:`S^2/A`)

        | With second derivative of distribution function. Eq(A28) in `Ref <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043081>`_
        | Output: :math:`\sigma_{\alpha\beta\gamma} = -e^3/\hbar^2 \tau^2 \int [dk] v_\beta v_\alpha v_\gamma f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta\gamma} E_\beta E\gamma`"""

    def __init__(self, **kwargs):
        self.Formula = frml.VelVelVel
        self.factor = 0.5 * factors.factor_nldrude
        self.fder = 2
        super().__init__(**kwargs)


class SHC(StaticCalculator):
    r"""Spin Hall conductivity with dc (:math:`S/m`)

        | With Fermi sea integral. Eq(1) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`_
        | Qiao type : with kwargs_formula={'spin_current_type':'qiao'}. Eq(49,50) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.214402>`_
        | Ryoo type : with kwargs_formula={'spin_current_type':'ryoo'}. Eq(17,26-29) in `Ref <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.235113>`_
        | Output: :math:`\sigma_{\alpha\beta} = -e^2/\hbar \int [dk] Im(j_{nm,\alpha} v_{mn,\beta})/(\epsilon_n - \epsilon_m)^2 f`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta} E_\beta`"""

    def __init__(self, **kwargs):
        self.Formula = frml.SpinOmega
        self.factor = factors.factor_ahc * -0.5
        self.fder = 0
        super().__init__(**kwargs)


# E^1 B^1
class AHC_Zeeman_spin(StaticCalculator):
    r"""AHC conductivity Zeeman correcton term spin part (:math:`S/m/T`)

        | With Fermi surface integral.
        | Output: :math:`ZAHC^{spin}_{\alpha\beta :\mu} = e^2/\hbar \int [dk] \Omega_\delta * s_\mu f'`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta :\mu} E_\beta B_\mu = \epsilon_{\alpha\beta\delta} ZAHC^{spin}_{\alpha\beta:\mu} E_\beta` B_\mu"""

    def __init__(self, **kwargs):
        self.Formula = frml.OmegaS
        self.factor = factors.fac_spin_Z * factors.factor_ahc
        self.fder = 1
        super().__init__(**kwargs)


class OmegaOmega(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.OmegaOmega
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)


class AHC_Zeeman_orb(StaticCalculator):
    r"""AHC conductivity Zeeman correction term orbital part (:math:`S/m/T`)

        | With Fermi surface integral.
        | Output: :math:`ZAHC^{orb}_{\alpha\beta :\mu} = e^2/\hbar \int [dk] \Omega_\delta * m_\mu f'`
        | Where :math: `m = H + G - 2Ef*\Omega`
        | Instruction: :math:`j_\alpha = \sigma_{\alpha\beta :\mu} E_\beta B_\mu = e \epsilon_{\alpha\beta\delta} ZAHC^{orb}_{\alpha\beta:\mu} E_\beta B_\mu`"""

    def __init__(self, **kwargs):
        self.Formula = frml.OmegaHplus
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)

    def __call__(self, data_K):
        Hplus_res = super().__call__(data_K)
        Omega_res = OmegaOmega(Efermi=self.Efermi, tetra=self.tetra,
                smoother=self.smoother,use_factor=False, print_comment=False,
                kwargs_formula=self.kwargs_formula)(data_K).mul_array(self.Efermi)
        final_factor = factors.fac_orb_Z * factors.factor_ahc
        if not self.use_factor:
            final_factor = np.sign(final_factor)

        return final_factor * (Hplus_res - 2 * Omega_res)
