"""Test wberri.run function"""
import os

import numpy as np
import pytest

import wannierberri as wberri
from wannierberri import calculators as calc
from wannierberri.__result import EnergyResult

from common import OUTPUT_DIR
from common_comparers import compare_quant
from common_systems import Efermi_Fe, Efermi_GaAs, Efermi_Chiral, Efermi_Te_gpaw


@pytest.fixture
def check_run(parallel_serial, compare_any_result):
    def _inner(
        system,
        calculators={},
        fout_name="berry",
        compare_zero=False,
        parallel=parallel_serial,
        grid_param={
            'NK': [6, 6, 6],
            'NKFFT': [3, 3, 3]
        },
        adpt_num_iter=0,
        parameters_K={},
        use_symmetry=False,
        suffix="",
        suffix_ref="",
        extra_precision={},
        precision=-1e-8,
        restart=False,
        file_Klist=None,
        do_not_compare=False,
        skip_compare=[],
    ):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.run(
            system,
            grid=grid,
            calculators=calculators,
            parallel=parallel,
            adpt_num_iter=adpt_num_iter,
            use_irred_kpt=use_symmetry,
            symmetrize=use_symmetry,
            parameters_K=parameters_K,
            fout_name=os.path.join(OUTPUT_DIR, fout_name),
            suffix=suffix,
            restart=restart,
            file_Klist=file_Klist,
        )

        if do_not_compare:
            return result  # compare result externally

        if len(suffix) > 0:
            suffix = "-" + suffix
        if len(suffix_ref) > 0:
            suffix_ref = "-" + suffix_ref

        for quant in calculators.keys():
            print(quant, skip_compare)
            if quant not in skip_compare:
                prec = extra_precision[quant] if quant in extra_precision else precision
                compare_any_result(
                    fout_name,
                    quant + suffix,
                    adpt_num_iter,
                    suffix_ref=compare_quant(quant) + suffix_ref,
                    compare_zero=compare_zero,
                    precision=prec,
                    result_type=resultType(quant),
                )

        return result

    return _inner


calculators_Fe = {
    'ahc': calc.static.AHC,
    'conductivity_ohmic': calc.static.Ohmic,
}
#,'ahc_test','dos','cumdos',
#           'conductivity_ohmic','conductivity_ohmic_fsurf','Morb','Morb_test']

calculators_GaAs = {
    'berry_dipole': calc.static.BerryDipole_FermiSea,
    'berry_dipole_fsurf': calc.static.BerryDipole_FermiSurf,
}

calculators_Te = {
    'dos': calc.static.DOS,
    'cumdos': calc.static.CumDOS,
    'berry_dipole': calc.static.BerryDipole_FermiSea,
}

calculators_Chiral = {
    'conductivity_ohmic': calc.static.Ohmic(Efermi=Efermi_Chiral),
    'berry_dipole': calc.static.BerryDipole_FermiSea(Efermi=Efermi_Chiral, kwargs_formula={"external_terms": False}),
    'ahc': calc.static.AHC(Efermi=Efermi_Chiral, kwargs_formula={"external_terms": False})
}


def resultType(quant):
    if quant in []:  # in future - add other options (tabulateresult)
        pass
    else:
        return EnergyResult


def test_Fe(check_run, system_Fe_W90, compare_any_result, compare_fermisurfer):
    param = {'Efermi': Efermi_Fe}
    param_tab = {'degen_thresh': 5e-2}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    calculators["tabulate"] = calc.TabulatorAll(
        {
            "Energy": calc.tabulate.Energy(),  # yes, in old implementation degen_thresh was applied to qunatities,
            # but not to energies
            "berry": calc.tabulate.BerryCurvature(**param_tab),
        },
        ibands=[5, 6, 7, 8])

    parameters_optical = dict(
        Efermi=np.array([17.0, 18.0]), omega=np.arange(0.0, 7.1, 1.0), smr_fixed_width=0.20, smr_type="Gaussian")
    calculators['opt_conductivity'] = wberri.calculators.dynamic.OpticalConductivity(**parameters_optical)
    calculators['opt_SHCqiao'] = wberri.calculators.dynamic.SHC(SHC_type="qiao", **parameters_optical)
    calculators['opt_SHCryoo'] = wberri.calculators.dynamic.SHC(SHC_type="ryoo", **parameters_optical)

    result = check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"Morb": -1e-6},
        skip_compare=['tabulate', 'opt_conductivity', 'opt_SHCqiao', 'opt_SHCryoo'])

    for quant in 'opt_conductivity', 'opt_SHCryoo', 'opt_SHCryoo':
        compare_any_result(
            "berry_Fe_W90",
            quant + "-run",
            0,
            fout_name_ref="kubo_Fe_W90",
            suffix_ref=quant,
            precision=1e-8,
            result_type=EnergyResult)

    extra_precision = {'berry': 1e-6}
    for quant in ["Energy", "berry"]:
        for comp in result.results.get("tabulate").results.get(quant).get_component_list():
            _quant = "E" if quant == "Energy" else quant
            _comp = "-" + comp if comp != "" else ""
            #            data=result.results.get(quant).data
            #            assert data.shape[0] == len(Efermi)
            #            assert np.all( np.array(data.shape[1:]) == 3)
            prec = extra_precision[quant] if quant in extra_precision else 1e-8
            #            comparer(frmsf_name, quant+_comp+suffix,  suffix_ref=compare_quant(quant)+_comp+suffix_ref ,precision=prec )
            compare_fermisurfer(
                fout_name="berry_Fe_W90-tabulate",
                suffix=_quant + _comp + "-run",
                suffix_ref=_quant + _comp,
                fout_name_ref="tabulate_Fe_W90",
                precision=prec)


def test_Fe_parallel_ray(check_run, system_Fe_W90, compare_any_result, parallel_ray):
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="paral-ray-4-run",
        parallel=parallel_ray,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )
    parallel_ray.shutdown()


def test_Fe_sym(check_run, system_Fe_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="sym-run",
        suffix_ref="sym",
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Fe_sym_refine(check_run, system_Fe_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="sym-run",
        suffix_ref="sym",
        adpt_num_iter=1,
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Fe_pickle_Klist(check_run, system_Fe_W90, compare_any_result):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    #  First, remove the
    try:
        os.remove("Klist.pickle")
    except FileNotFoundError:
        pass
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="pickle-run",
        suffix_ref="sym",
        adpt_num_iter=0,
        use_symmetry=True,
        file_Klist="Klist.pickle",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="pickle-run",
        suffix_ref="sym",
        adpt_num_iter=1,
        use_symmetry=True,
        file_Klist="Klist.pickle",
        restart=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_GaAs(check_run, system_GaAs_W90, compare_any_result, compare_fermisurfer):
    param = {'Efermi': Efermi_GaAs}
    calculators = {k: v(**param) for k, v in calculators_GaAs.items()}

    check_run(
        system_GaAs_W90,
        calculators,
        fout_name="berry_GaAs_W90",
        suffix="run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"berry_dipole_fsurf": 1e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_Chiral_left(check_run, system_Chiral_left, compare_any_result, compare_fermisurfer):
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    check_run(
        system_Chiral_left,
        calculators_Chiral,
        fout_name="berry_Chiral",
        suffix="left-run",
        grid_param=grid_param,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=True,
        extra_precision={"Morb": -1e-6},
    )


def test_Chiral_leftTR(check_run, system_Chiral_left, system_Chiral_left_TR, compare_any_result, compare_fermisurfer):
    "check that for time-reversed model the ohmic conductivity is the same, but the AHC is opposite"
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    results = [
        check_run(
            system,
            calculators_Chiral,
            fout_name="berry_Chiral",
            suffix="right-run",
            grid_param=grid_param,
            parameters_K={
                '_FF_antisym': True,
                '_CCab_antisym': True
            },
            use_symmetry=True,
            do_not_compare=True) for system in [system_Chiral_left, system_Chiral_left_TR]
    ]

    for key in ["conductivity_ohmic", "berry_dipole", "ahc"]:
        sign = -1 if key == "ahc" else 1
        data1 = results[0].results[key].dataSmooth
        data2 = results[1].results[key].dataSmooth
        precision = max(np.max(abs(data1)) / 1E12, 1E-11)
        assert data1 == pytest.approx(sign * data2, abs=precision), key


def test_Chiral_right(check_run, system_Chiral_left, system_Chiral_right, compare_any_result, compare_fermisurfer):
    "check that for flipped chirality the ohmic conductivity is the same, but the Berry dipole is opposite"
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    results = [
        check_run(
            system,
            calculators_Chiral,
            fout_name="berry_Chiral",
            suffix="right-run",
            grid_param=grid_param,
            parameters_K={
                '_FF_antisym': True,
                '_CCab_antisym': True
            },
            use_symmetry=True,
            do_not_compare=True) for system in [system_Chiral_left, system_Chiral_right]
    ]

    for key in ["conductivity_ohmic", "berry_dipole", "ahc"]:
        sign = -1 if key == "berry_dipole" else 1
        data1 = results[0].results[key].dataSmooth
        data2 = results[1].results[key].dataSmooth
        precision = max(np.max(abs(data1)) / 1E12, 1E-11)
        assert data1 == pytest.approx(sign * data2, abs=precision), key


def test_Te_ASE(check_run, system_Te_ASE, compare_any_result):
    param = {'Efermi': Efermi_Te_gpaw, "tetra": True}
    calculators = {k: v(**param) for k, v in calculators_Te.items()}
    check_run(
        system_Te_ASE,
        calculators,
        fout_name="berry_Te_ASE",
        suffix="",
        suffix_ref="",
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Te_ASE_wcc(check_run, system_Te_ASE_wcc, compare_any_result):
    param = {'Efermi': Efermi_Te_gpaw, "tetra": True}
    calculators = {}
    for k, v in calculators_Te.items():
        par = {}
        par.update(param)
        if k not in ["dos", "cumdos"]:
            par["kwargs_formula"] = {"external_terms": False}
        calculators[k] = v(**par)

    check_run(
        system_Te_ASE_wcc,
        calculators,
        fout_name="berry_Te_ASE",
        suffix="wcc",
        suffix_ref="",
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )
