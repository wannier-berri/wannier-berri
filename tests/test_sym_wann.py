"""Test wberri.run function"""
import os

import numpy as np
import pytest
import pickle
from pytest import approx

import wannierberri as wberri
from wannierberri import calculators as calc
from wannierberri.smoother import FermiDiracSmoother
from wannierberri.result import EnergyResult

from common import OUTPUT_DIR, REF_DIR
from common_comparers import compare_quant
from common_systems import (
    Efermi_Fe,
    Efermi_Fe_FPLO,
    Efermi_GaAs,
    Efermi_Haldane,
    Efermi_CuMnAs_2d,
    Efermi_Chiral,
    Efermi_Te_gpaw,
    Efermi_Te_sparse,
    omega_phonon,
    Efermi_Mn3Sn,
)

from test_kubo import check_integrate_dynamical
from test_run import (
        check_run,
        calculators_GaAs,
        calculators_GaAs_internal,
        )


def test_shiftcurrent_symmetry(check_integrate_dynamical, system_GaAs_sym_tb):
    """Test shift current with and without symmetry is the same for a symmetrized system"""
    import copy

    quantities=["opt_conductivity"]
#    quantities=["ahc"]
#   quantities=["opt_shiftcurrent"]
    kwargs = dict(
        quantities=quantities,
#        Efermi=np.array([7.0]),
        Efermi=np.linspace(7.0,8.9,10),
        omega=np.arange(1.0, 5.1, 0.5),
        grid_param=dict(NK=6, NKFFT=3),
        additional_parameters=dict(smr_fixed_width=0.20, smr_type="Gaussian", sc_eta=0.1),
        comparer=None,
    )

    system = system_GaAs_sym_tb

    result_irr_k = check_integrate_dynamical(system, use_symmetry=True, fout_name="kubo_GaAs_sym_irr_k", **kwargs)
    result_full_k = check_integrate_dynamical(system, use_symmetry=False, fout_name="kubo_GaAs_sym_full_k", **kwargs)

    # FIXME: there is small but nonzero difference between the two results.
    # It seems that the finite eta correction term (PRB 103 247101 (2021)) is needed to get perfect agreement.
    for quant in quantities:
        assert result_full_k.results[quant].data == approx(
            result_irr_k.results[quant].data, abs=1e-6)



@pytest.fixture
def check_symmetry(check_run):
    def _inner(
        calculators = {},
        precision=1e-8,
        **kwargs,
            ):
        kwargs['do_not_compare']=True
        result_irr_k = check_run( use_symmetry=True, calculators=calculators,suffix="irr_k", **kwargs)
        result_full_k = check_run( use_symmetry=False, calculators=calculators,suffix="full_k", **kwargs)
        print (calculators.keys(),result_irr_k.results.keys(),result_full_k.results.keys())

        for quant in calculators.keys():
            assert result_full_k.results[quant].data == approx(
                    result_irr_k.results[quant].data,
                    rel=abs(precision) if precision<0 else None,
                    abs=precision if precision>0 else None)

    return _inner


def test_GaAs_sym_tb(check_symmetry, system_GaAs_sym_tb):
    param = {'Efermi': Efermi_GaAs}
    calculators = {}
#    calculators.update({k: v(**param) for k, v in calculators_GaAs.items()})
#    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    calculators.update({
        'berry_dipole':calc.static.BerryDipole_FermiSea(Efermi=Efermi_GaAs, kwargs_formula={"external_terms":True}),
#    calculators.update({
#        'gyrotropic_Korb':calc.static.GME_orb_FermiSea(Efermi=Efermi_GaAs, kwargs_formula={"external_terms":False}),
#        'gyrotropic_Kspin':calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
#        'gyrotropic_Kspin_fsurf':calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
#        'gyrotropic_Korb_test':calc.static.GME_orb_FermiSea_test(Efermi=Efermi_GaAs),
                        })


    check_symmetry(system=system_GaAs_sym_tb,calculators=calculators)


def test_Mn3Sn_sym_tb(check_symmetry, system_Mn3Sn_sym_tb):
    param = {'Efermi': Efermi_Mn3Sn}
    calculators = {}
#    calculators.update({k: v(**param) for k, v in calculators_GaAs.items()})
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    calculators.update({
        'ahc':calc.static.AHC(Efermi=Efermi_Mn3Sn, kwargs_formula={"external_terms":True}),
#        'gyrotropic_Korb':calc.static.GME_orb_FermiSea(Efermi=Efermi_GaAs, kwargs_formula={"external_terms":False}),
#        'gyrotropic_Kspin':calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
#        'gyrotropic_Kspin_fsurf':calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
#        'gyrotropic_Korb_test':calc.static.GME_orb_FermiSea_test(Efermi=Efermi_GaAs),
                        })

    check_symmetry(system=system_Mn3Sn_sym_tb,calculators=calculators)

