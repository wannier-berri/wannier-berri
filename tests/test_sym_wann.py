"""Test symmetrization of Wannier models"""
import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from wannierberri import calculators as calc

from .common_systems import (
    Efermi_GaAs,
    Efermi_Fe,
    Efermi_Mn3Sn,
    Efermi_Te_sparse,
)


from .test_run import (
    calculators_GaAs_internal,
    calculators_Te,
)




@pytest.fixture
def check_symmetry(check_run):
    def _inner(system,
            calculators={},
            precision=1e-7,
            extra_precision={},
            **kwargs,
            ):
        kwargs['do_not_compare'] = True
        print(f"using symmetries :{system.pointgroup}")
        result_irr_k = check_run(system, use_symmetry=True, calculators=calculators, suffix="irr_k", **kwargs)
        result_full_k = check_run(system, use_symmetry=False, calculators=calculators, suffix="full_k", **kwargs)
        print(calculators.keys(), result_irr_k.results.keys(), result_full_k.results.keys())

        for quant in calculators.keys():
            diff = abs(result_full_k.results[quant].data - result_irr_k.results[quant].data).max()
            try:
                prec = extra_precision[quant]
            except KeyError:
                prec = precision
            if prec < 0:
                req_precision = -prec * (abs(result_full_k.results[quant].data) + abs(result_irr_k.results[quant].data)).max() / 2
            else:
                req_precision = prec
            assert diff <= req_precision, (
                f"data of {quant} with and without symmetries give a maximal "
                f"absolute difference of {diff} greater than the required precision {req_precision}"
            )
    return _inner




def test_shiftcurrent_symmetry(check_symmetry, system_GaAs_sym_tb):
    """Test shift current with and without symmetry is the same for a symmetrized system"""
    param = dict(
        Efermi=Efermi_GaAs,
        omega=np.arange(1.0, 5.1, 0.5),
        smr_fixed_width=0.2,
        smr_type='Gaussian',
        kBT=0.01,
    )
    calculators = dict(
        shift_current=calc.dynamic.ShiftCurrent(sc_eta=0.1, **param),
    )

    check_symmetry(system=system_GaAs_sym_tb,
                   grid_param=dict(NK=6, NKFFT=3),
                   calculators=calculators,
                   precision=1e-6
                    )




def test_Mn3Sn_sym_tb(check_symmetry, system_Mn3Sn_sym_tb):
    param = {'Efermi': Efermi_Mn3Sn}
    calculators = {}
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    calculators.update({
        'ahc_int': calc.static.AHC(Efermi=Efermi_Mn3Sn, kwargs_formula={"external_terms": False}),
        'ahc_ext': calc.static.AHC(Efermi=Efermi_Mn3Sn, kwargs_formula={"internal_terms": False}),
        'ahc': calc.static.AHC(Efermi=Efermi_Mn3Sn, kwargs_formula={"external_terms": True}),
    })
    check_symmetry(system=system_Mn3Sn_sym_tb, calculators=calculators,
                   extra_precision={"conductivity_ohmic": -1e-6, "ahc_int": -1e-5, 'ahc_ext': -1e-6, 'ahc': -1e-5},)


@pytest.mark.parametrize("use_k_sym", [False, True])
def test_Fe_sym_W90(check_run, system_Fe_sym_W90, compare_any_result, use_k_sym):
    system = system_Fe_sym_W90
    param = {'Efermi': Efermi_Fe}
    cals = {'ahc': calc.static.AHC,
            'Morb': calc.static.Morb,
            'spin': calc.static.Spin}
    calculators = {k: v(**param) for k, v in cals.items()}

    check_run(
        system,
        calculators,
        fout_name="Fe_sym_W90",
        # suffix="-run",
        use_symmetry=use_k_sym
    )
    cals = {'gyrotropic_Korb': calc.static.GME_orb_FermiSea,
            'berry_dipole': calc.static.BerryDipole_FermiSea,
            'gyrotropic_Kspin': calc.static.GME_spin_FermiSea}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system,
        calculators,
        fout_name="Fe_sym_W90",
        precision=1e-8,
        suffix=f"-use_k_sym-{use_k_sym}",
        compare_zero=True,
        use_symmetry=use_k_sym
    )


@pytest.fixture
def checksym_Fe(check_run, compare_any_result, check_symmetry):
    def _inner(system, extra_calculators={}):
        param = {'Efermi': Efermi_Fe}
        cals = {'dos': calc.static.DOS,
                'cumdos': calc.static.CumDOS,
            'conductivity_ohmic': calc.static.Ohmic_FermiSea,
            'conductivity_ohmic_fsurf': calc.static.Ohmic_FermiSurf,
            'ahc': calc.static.AHC,
            'Morb': calc.static.Morb,
            'spin': calc.static.Spin}
        calculators = {k: v(**param) for k, v in cals.items()}
        calculators.update({
            'ahc_int': calc.static.AHC(Efermi=Efermi_Fe, kwargs_formula={"external_terms": False}),
            'ahc_ext': calc.static.AHC(Efermi=Efermi_Fe, kwargs_formula={"internal_terms": False}),
            'SHCryoo_static': calc.static.SHC(Efermi=Efermi_Fe, kwargs_formula={'spin_current_type': 'ryoo'})
        })
        calculators.update(extra_calculators)
        check_symmetry(system=system,
                       grid_param=dict(NK=6, NKFFT=3),
                   calculators=calculators,
                   precision=-1e-8
                    )
    return _inner



def test_Fe_new(system_Fe_sym_W90, checksym_Fe):
    extra_calculators = {}
    extra_calculators['SHCqiao_static'] = \
        wberri.calculators.static.SHC(Efermi=Efermi_Fe, kwargs_formula={'spin_current_type': 'qiao'})
    extra_calculators['SHCryoo_static'] = \
        wberri.calculators.static.SHC(Efermi=Efermi_Fe, kwargs_formula={'spin_current_type': 'ryoo'})
    extra_calculators['SHCryoo_simple'] = \
        wberri.calculators.static.SHC(Efermi=Efermi_Fe, kwargs_formula={'spin_current_type': 'simple'})
    checksym_Fe(system_Fe_sym_W90, extra_calculators=extra_calculators)


def test_GaAs_sym_tb_zero(check_symmetry, check_run, system_GaAs_sym_tb, compare_any_result):
    param = {'Efermi': Efermi_GaAs}
    calculators = {}
    calculators.update({
        'berry_dipole': calc.static.BerryDipole_FermiSea(**param, kwargs_formula={"external_terms": True}),
        'gyrotropic_Korb': calc.static.GME_orb_FermiSea(Efermi=Efermi_GaAs, kwargs_formula={"external_terms": True}),
        'gyrotropic_Kspin': calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
        # 'gyrotropic_Kspin_fsurf':calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
        # 'gyrotropic_Korb_test':calc.static.GME_orb_FermiSea_test(Efermi=Efermi_GaAs),
    })

    check_run(
        system_GaAs_sym_tb,
        {'ahc': calc.static.AHC(Efermi=Efermi_GaAs)},
        fout_name="GaAs_sym_tb",
        precision=1e-5,
        compare_zero=True,
        suffix="sym-zero",
    )


def test_GaAs_random_zero(check_symmetry, check_run, system_random_GaAs_load_sym, compare_any_result):
    param = {'Efermi': Efermi_GaAs}
    calculators = {}
    calculators.update({
        'berry_dipole': calc.static.BerryDipole_FermiSea(**param, kwargs_formula={"external_terms": True}),
        # 'gyrotropic_Korb': calc.static.GME_orb_FermiSea(Efermi=Efermi_GaAs, kwargs_formula={"external_terms": True}),
        'gyrotropic_Kspin': calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
        'ahc': calc.static.AHC(Efermi=Efermi_GaAs),
        # 'gyrotropic_Kspin_fsurf':calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
        # 'gyrotropic_Korb_test':calc.static.GME_orb_FermiSea_test(Efermi=Efermi_GaAs),
    })


def test_GaAs_sym_tb(check_symmetry, system_GaAs_sym_tb):
    param = {'Efermi': Efermi_GaAs}
    calculators = {}
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    check_symmetry(system=system_GaAs_sym_tb, calculators=calculators)


def test_GaAs_random(check_symmetry, system_random_GaAs_load_sym):
    system = system_random_GaAs_load_sym
    param = {'Efermi': Efermi_GaAs}
    calculators = {}
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    param = dict(
        Efermi=Efermi_GaAs,
        omega=np.arange(1.0, 5.1, 0.5),
        smr_fixed_width=0.2,
        smr_type='Gaussian',
        kBT=0.01,
    )
    calculators.update({'SHC-ryoo': calc.dynamic.SHC(SHC_type='ryoo', **param)})
    check_symmetry(system=system, calculators=calculators,
                   extra_precision={"SHC-ryoo": 2e-7})



def test_GaAs_dynamic_sym(check_run, system_GaAs_sym_tb, compare_any_result):
    "Test shift current and injection current"

    param = dict(
        Efermi=Efermi_GaAs,
        omega=np.arange(1.0, 5.1, 0.5),
        smr_fixed_width=0.2,
        smr_type='Gaussian',
        kBT=0.01,
    )
    calculators = dict(
        shift_current=calc.dynamic.ShiftCurrent(sc_eta=0.1, **param),
        injection_current=calc.dynamic.InjectionCurrent(**param),
        opt_conductivity=calc.dynamic.OpticalConductivity(**param)
    )

    result_full_k = check_run(
        system_GaAs_sym_tb,
        calculators,
        fout_name="dynamic_GaAs_sym",
        grid_param={
            'NK': [6, 6, 6],
            'NKFFT': [3, 3, 3]
        },
        use_symmetry=False,
        do_not_compare=True,
    )

    result_irr_k = check_run(
        system_GaAs_sym_tb,
        calculators,
        fout_name="dynamic_GaAs_sym",
        suffix="sym",
        suffix_ref="",
        grid_param={
            'NK': [6, 6, 6],
            'NKFFT': [3, 3, 3]
        },
        use_symmetry=True,
        do_not_compare=True,
    )


    assert result_full_k.results["shift_current"].data == approx(
        result_irr_k.results["shift_current"].data, abs=1e-6)

    assert result_full_k.results["injection_current"].data == approx(
        result_irr_k.results["injection_current"].data, abs=1e-6)

    assert result_full_k.results["opt_conductivity"].data == approx(
        result_irr_k.results["opt_conductivity"].data, abs=1e-7)



def test_Te_sparse(check_symmetry, system_Te_sparse):
    param = {'Efermi': Efermi_Te_sparse, 'Emax': 6.15, 'hole_like': True}
    calculators = {}
    for k, v in calculators_Te.items():
        par = {}
        par.update(param)
        if k not in ["dos", "cumdos"]:
            par["kwargs_formula"] = {"external_terms": False}
        calculators[k] = v(**par)


        check_symmetry(system=system_Te_sparse,
                       grid_param=dict(NK=(6, 6, 4), NKFFT=(3, 3, 2)),
                       calculators=calculators,
                       precision=-1e-8,
                extra_precision={"berry_dipole": 5e-7},
                    )


def test_Te_sparse_tetragrid(check_run, system_Te_sparse, compare_any_result):
    param = {'Efermi': Efermi_Te_sparse, "tetra": True, 'use_factor': False, 'Emax': 6.15, 'hole_like': True}
    calculators = {}
    for k, v in calculators_Te.items():
        par = {}
        par.update(param)
        if k not in ["dos", "cumdos"]:
            par["kwargs_formula"] = {"external_terms": False}
        calculators[k] = v(**par)

    grid = wberri.grid.GridTrigonal(system_Te_sparse, length=50, NKFFT=[3, 3, 2])

    check_run(
        system_Te_sparse,
        calculators,
        fout_name="Te_sparse_tetragrid",
        use_symmetry=True,
        grid=grid,
        # temporarily weakened precision here. Will restrict it later with new data
        extra_precision={"berry_dipole": 3e-7},
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Te_sparse_tetragridH(check_run, system_Te_sparse, compare_any_result):
    param = {'Efermi': Efermi_Te_sparse, "tetra": True, 'use_factor': False}
    calculators = {}
    for k, v in calculators_Te.items():
        par = {}
        par.update(param)
        if k not in ["dos", "cumdos"]:
            par["kwargs_formula"] = {"external_terms": False}
        calculators[k] = v(**par)

    grid = wberri.grid.GridTrigonalH(system_Te_sparse, length=50, NKFFT=1, x=0.6)

    check_run(
        system_Te_sparse,
        calculators,
        fout_name="Te_sparse_tetragridH",
        use_symmetry=True,
        grid=grid,
        # temporarily weakened precision here. Will restrict it later with new data
        extra_precision={"berry_dipole": 3e-7, "dos": 2e-8},
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_KaneMele_sym(check_symmetry, system_KaneMele_odd_PythTB):
    param = {'Efermi': np.linspace(-4., 4., 21)}
    calculators = {}
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    calculators.update({
        'berry_dipole': calc.static.BerryDipole_FermiSea(**param, kwargs_formula={"external_terms": False}),
        'gyrotropic_Korb': calc.static.GME_orb_FermiSea(**param, kwargs_formula={"external_terms": False}),
        'gyrotropic_Kspin': calc.static.GME_spin_FermiSea(**param),
    })

    check_symmetry(system=system_KaneMele_odd_PythTB,
                   grid_param=dict(NK=(6, 6, 1), NKFFT=(3, 3, 1)),
                   calculators=calculators)
