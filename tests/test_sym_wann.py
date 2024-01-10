"""Test symmetrization of Wannier models"""
from wannierberri.system.sym_wann import _dict_to_matrix, _matrix_to_dict, _get_H_select, _rotate_matrix
import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from wannierberri import calculators as calc

from common_systems import (
    Efermi_GaAs,
    Efermi_Fe,
    Efermi_Mn3Sn,
    Efermi_Te_sparse,
)


from test_run import (
        calculators_GaAs_internal,
        calculators_Te,
                        )




@pytest.fixture
def check_symmetry(check_run):
    def _inner(system,
        calculators={},
        precision=1e-7,
        **kwargs,
            ):
        kwargs['do_not_compare'] = True
        result_irr_k = check_run(system, use_symmetry=True, calculators=calculators, suffix="irr_k", **kwargs)
        result_full_k = check_run(system, use_symmetry=False, calculators=calculators, suffix="full_k", **kwargs)
        print(calculators.keys(), result_irr_k.results.keys(), result_full_k.results.keys())

        for quant in calculators.keys():
            diff = abs(result_full_k.results[quant].data - result_irr_k.results[quant].data).max()
            if precision < 0:
                req_precision = -precision * (abs(result_full_k.results[quant].data) + abs(result_irr_k.results[quant].data)).max() / 2
            else:
                req_precision = precision
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
        'ahc': calc.static.AHC(Efermi=Efermi_Mn3Sn, kwargs_formula={"external_terms": True}),
                        })
    check_symmetry(system=system_Mn3Sn_sym_tb, calculators=calculators)


def test_Fe_sym_W90(check_run, system_Fe_sym_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    cals = {'ahc': calc.static.AHC,
            'Morb': calc.static.Morb,
            'spin': calc.static.Spin}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        suffix="-run",
        use_symmetry=False
    )
    cals = {'gyrotropic_Korb': calc.static.GME_orb_FermiSea,
            'berry_dipole': calc.static.BerryDipole_FermiSea,
            'gyrotropic_Kspin': calc.static.GME_spin_FermiSea}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        precision=1e-8,
        suffix="-run",
        compare_zero=True,
        use_symmetry=False
    )


def test_Fe_sym_W90_sym(check_run, system_Fe_sym_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    cals = {'ahc': calc.static.AHC,
            'Morb': calc.static.Morb,
            'spin': calc.static.Spin}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        suffix="sym-run",
        use_symmetry=True
    )
    cals = {'gyrotropic_Korb': calc.static.GME_orb_FermiSea,
            'berry_dipole': calc.static.BerryDipole_FermiSea,
            'gyrotropic_Kspin': calc.static.GME_spin_FermiSea}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        suffix="sym-run",
        precision=1e-8,
        compare_zero=True,
        use_symmetry=True
    )


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
        fout_name="berry_GaAs_sym_tb",
        precision=1e-5,
        compare_zero=True,
        suffix="sym-zero",
                )


def test_GaAs_sym_tb(check_symmetry, system_GaAs_sym_tb):
    param = {'Efermi': Efermi_GaAs}
    calculators = {}
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    check_symmetry(system=system_GaAs_sym_tb, calculators=calculators)


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
        fout_name="berry_Te_sparse_tetragrid",
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

    grid = wberri.grid.GridTrigonalH(system_Te_sparse, length=50, NKFFT=[3, 3, 2], x=0.6)

    check_run(
        system_Te_sparse,
        calculators,
        fout_name="berry_Te_sparse_tetragridH",
        use_symmetry=True,
        grid=grid,
        # temporarily weakened precision here. Will restrict it later with new data
        extra_precision={"berry_dipole": 3e-7, "dos": 2e-8},
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )





class AtomInfo():
    """fake AtomInfo for test"""
    def __init__(self, orbital_index):
        self.num_wann = sum(len(oi) for oi in orbital_index)
        self.orbital_index = orbital_index


def test_matrix_to_dict():
    wann_atom_info = [AtomInfo(n) for n in ([[1, 3], [5, 6]], [[0, 2, 4]])]
    num_wann = sum((at.num_wann for at in wann_atom_info))
    num_wann_atom = len(wann_atom_info)
    nRvec = 8
    ndimv = 2
    mat = np.random.random((num_wann, num_wann, nRvec) + (3,) * ndimv)
    H_select = _get_H_select(num_wann, num_wann_atom, wann_atom_info)
    dic = _matrix_to_dict(mat, H_select, wann_atom_info)
    mat_new = _dict_to_matrix(dic, H_select, nRvec, ndimv)
    assert mat_new == approx(mat, abs=1e-8)


def test_rotate_matrix():
    num_wann = 5
    L = np.random.random((num_wann, num_wann)) + 1j * np.random.random((num_wann, num_wann))
    R = np.random.random((num_wann, num_wann)) + 1j * np.random.random((num_wann, num_wann))
    scal = np.random.random((num_wann, num_wann)) + 1j * np.random.random((num_wann, num_wann))
    vec = np.random.random((num_wann, num_wann, 3)) + 1j * np.random.random((num_wann, num_wann, 3))
    assert _rotate_matrix(scal, L, R) == approx(np.einsum("lm,mn,np->lp", L, scal, R))
    assert _rotate_matrix(vec, L, R) == approx(np.einsum("lm,mna,np->lpa", L, vec, R))
