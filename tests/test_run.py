"""Test wberri.run function"""
import os

import numpy as np
import pytest
import pickle

import wannierberri as wberri
from wannierberri import calculators as calc
from wannierberri.smoother import FermiDiracSmoother
from wannierberri.__result import EnergyResult

from common import OUTPUT_DIR, REF_DIR
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


smoother_Chiral = FermiDiracSmoother(Efermi_Chiral, T_Kelvin=1200, maxdE=8)
calculators_Chiral = {
    'conductivity_ohmic': calc.static.Ohmic(Efermi=Efermi_Chiral,smoother=smoother_Chiral),
    'berry_dipole': calc.static.BerryDipole_FermiSea(Efermi=Efermi_Chiral, kwargs_formula={"external_terms": False},smoother=smoother_Chiral),
    'ahc': calc.static.AHC(Efermi=Efermi_Chiral, kwargs_formula={"external_terms": False},smoother=smoother_Chiral)
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
            precision=-1e-8,
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


def test_GaAs(check_run, system_GaAs_W90, compare_any_result):
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


def test_Chiral_left(check_run, system_Chiral_left, compare_any_result, compare_energyresult):
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

    for quant in calculators_Chiral.keys():#["conductivity_ohmic", "berry_dipole", "ahc"]:
        compare_energyresult(
                fout_name="berry_Chiral",
                suffix=quant+"-left-run",
                adpt_num_iter=0,
                suffix_ref=quant,
                mode="txt",
                compare_smooth=True,
                precision=-1e-8)


def test_Chiral_leftTR(check_run, system_Chiral_left, system_Chiral_left_TR, compare_any_result):
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


def test_Chiral_right(check_run, system_Chiral_left, system_Chiral_right, compare_any_result):
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


def test_tabulate_path(system_Haldane_PythTB):

    k_nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, dk=1.0)

    calculators = {}
    quantities = {
                            "Energy":wberri.calculators.tabulate.Energy(),
                            "berry":wberri.calculators.tabulate.BerryCurvature(kwargs_formula={"external_terms":False}),
                                  }

    calculators ["tabulate"] = wberri.calculators.TabulatorAll(quantities,
                                                       ibands=[0],
                                                        mode="path")


    run_result = wberri.run(
        system=system_Haldane_PythTB,
        grid=path,
        calculators=calculators,
        use_irred_kpt=True,
        symmetrize=True,  # should have no effect, but will check the cases and give a warning
        #                parameters_K = parameters_K,
        #                frmsf_name = None,
        #                degen_thresh = degen_thresh, degen_Kramers = degen_Kramers
    )


    tab_result=run_result.results["tabulate"]
    filename = "path_tab.pickle"
    fout = open(os.path.join(OUTPUT_DIR, filename+"-run"), "wb")

    data = {}
    for quant in quantities.keys():
        result_quant = tab_result.results.get(quant)
        for comp in result_quant.get_component_list():
            data[(quant, comp)] = result_quant.get_component(comp)
    pickle.dump(data, fout)

    data_ref = pickle.load(open(os.path.join(REF_DIR, filename), "rb"))

    for quant in quantities.keys():
        for comp in tab_result.results.get(quant).get_component_list():
            _data = data[(quant, comp)]
            _data_ref = data_ref[(quant, comp)]
            assert _data == pytest.approx(_data_ref), (
                f"tabulation along path gave a wrong result for quantity {quant} component {comp} "
                + "with a maximal difference {}".format(max(abs(data - data_ref))))

    # only checks that the plot runs without errors, not checking the result of the plot
    tab_result.plot_path_fat(
        path,
        quantity='berry',
        component='z',
        save_file=os.path.join(OUTPUT_DIR, "Haldane-berry-VB.pdf"),
        Eshift=0,
        Emin=-2,
        Emax=2,
        iband=None,
        mode="fatband",
        fatfactor=20,
        cut_k=True,
        show_fig=False)
