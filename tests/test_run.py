"""Test `wberri.run function"""
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
    omega_chiral,
    omega_phonon,
    mass_kp_iso
)


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
        grid=None,
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

        if grid is None:
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
    'ahc_test': calc.static.AHC_test,
    'conductivity_ohmic': calc.static.Ohmic_FermiSea,
    'conductivity_ohmic_fsurf': calc.static.Ohmic_FermiSurf,
    'Morb': calc.static.Morb,
    'Morb_test': calc.static.Morb_test,
    'dos': calc.static.DOS,
    'cumdos': calc.static.CumDOS,
    'spin': calc.static.Spin,
}

calculators_phonons = {
    'dos': calc.static.DOS,
    'cumdos': calc.static.CumDOS,
}


calculators_GaAs = {
    'berry_dipole': calc.static.BerryDipole_FermiSea,
    'berry_dipole_fsurf': calc.static.BerryDipole_FermiSurf,
    'berry_dipole_test': calc.static.BerryDipole_FermiSea_test
}

calculators_GaAs_internal = {
    'dos': calc.static.DOS,
    'cumdos': calc.static.CumDOS,
    'conductivity_ohmic': calc.static.Ohmic_FermiSea,
}

calculators_Haldane = {
    'dos': calc.static.DOS,
    'ahc': calc.static.AHC,
    'conductivity_ohmic': calc.static.Ohmic_FermiSea,
}

calculators_Te = {
    'dos': calc.static.DOS,
    'cumdos': calc.static.CumDOS,
    'berry_dipole': calc.static.NLAHC_FermiSea,
}

calculators_CuMnAs_2d = {
    'dos': calc.static.DOS,
    'cumdos': calc.static.CumDOS,
    'conductivity_ohmic': calc.static.Ohmic_FermiSea,
    'Hall_morb_fsurf': calc.static.AHC_Zeeman_orb,
    'Hall_classic_fsurf': calc.static.Hall_classic_FermiSea,
}

smoother_Chiral = FermiDiracSmoother(Efermi_Chiral, T_Kelvin=1200, maxdE=8)

parameters_Chiral_optical = dict(
        Efermi=Efermi_Chiral, omega=omega_chiral, smr_fixed_width=0.20, smr_type="Gaussian",
        kwargs_formula={"external_terms": False}, )


parameters_Chiral_shiftcurrent = dict(sc_eta=0.1)

calculators_Chiral = {
    'conductivity_ohmic': calc.static.Ohmic_FermiSea(Efermi=Efermi_Chiral, smoother=smoother_Chiral),
    'conductivity_ohmic_fsurf': calc.static.Ohmic_FermiSurf(Efermi=Efermi_Chiral),
    'berry_dipole': calc.static.BerryDipole_FermiSea(Efermi=Efermi_Chiral, use_factor=False, kwargs_formula={"external_terms": False}, smoother=smoother_Chiral),
    'berry_dipole_fsurf': calc.static.BerryDipole_FermiSurf(Efermi=Efermi_Chiral, use_factor=False, kwargs_formula={"external_terms": False}),
    'ahc': calc.static.AHC(Efermi=Efermi_Chiral, kwargs_formula={"external_terms": False}, smoother=smoother_Chiral),
    'Der3E': calc.static.NLDrude_FermiSea(Efermi=Efermi_Chiral),
    'Hall_classic_fsurf': calc.static.Hall_classic_FermiSurf(Efermi=Efermi_Chiral),
    'Hall_classic': calc.static.Hall_classic_FermiSea(Efermi=Efermi_Chiral),
    'dos': calc.static.DOS(Efermi=Efermi_Chiral),
    'cumdos': calc.static.CumDOS(Efermi=Efermi_Chiral),
    'opt_conductivity': wberri.calculators.dynamic.OpticalConductivity(**parameters_Chiral_optical),
    'opt_shiftcurrent': wberri.calculators.dynamic.ShiftCurrent(**parameters_Chiral_shiftcurrent, **parameters_Chiral_optical),
}




calculators_Chiral_tetra = {
    'conductivity_ohmic': calc.static.Ohmic_FermiSea(Efermi=Efermi_Chiral, tetra=True),
    'conductivity_ohmic_fsurf': calc.static.Ohmic_FermiSurf(Efermi=Efermi_Chiral, tetra=True),
    'berry_dipole': calc.static.BerryDipole_FermiSea(Efermi=Efermi_Chiral, tetra=True, use_factor=False, kwargs_formula={"external_terms": False}),
    'berry_dipole_fsurf': calc.static.BerryDipole_FermiSurf(Efermi=Efermi_Chiral, tetra=True, use_factor=False, kwargs_formula={"external_terms": False}),
    'ahc': calc.static.AHC(Efermi=Efermi_Chiral, tetra=True, kwargs_formula={"external_terms": False}),
    'Der3E': calc.static.NLDrude_FermiSea(Efermi=Efermi_Chiral, tetra=True),
    'Hall_classic_fsurf': calc.static.Hall_classic_FermiSurf(Efermi=Efermi_Chiral, tetra=True),
    'Hall_classic': calc.static.Hall_classic_FermiSea(Efermi=Efermi_Chiral, tetra=True),
    'dos': calc.static.DOS(Efermi=Efermi_Chiral, tetra=True),
    'cumdos': calc.static.CumDOS(Efermi=Efermi_Chiral, tetra=True),
    'spin': wberri.calculators.static.Spin(Efermi=Efermi_Chiral, tetra=True),
}


Efermi_Chiral_half = Efermi_Chiral[:len(Efermi_Chiral) // 2]
calculators_Chiral_half = {
    'dos': calc.static.DOS(Efermi=Efermi_Chiral_half, tetra=False),
    'cumdos': calc.static.CumDOS(Efermi=Efermi_Chiral_half, tetra=False),
}

calculators_Chiral_tetra_half = {
    'dos': calc.static.DOS(Efermi=Efermi_Chiral_half, tetra=True),
    'cumdos': calc.static.CumDOS(Efermi=Efermi_Chiral_half, tetra=True),
}


def resultType(quant):
    if quant in []:  # in future - add other options (tabulateresult)
        pass
    else:
        return EnergyResult


def test_TabulatorAll_fail():
    with pytest.raises(ValueError):
        calc.TabulatorAll(
            {
                "Energy": calc.tabulate.Energy(),  # yes, in old implementation degen_thresh was applied to qunatities,
                # but not to energies
                "V": calc.tabulate.Velocity(ibands=[5, 6]),
            },
            ibands=[5, 6, 7, 8])


def test_Fe(check_run, system_Fe_W90, compare_any_result, compare_fermisurfer):
    param = {'Efermi': Efermi_Fe}
    param_tab = {'degen_thresh': 5e-2}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    calculators["tabulate"] = calc.TabulatorAll(
        {
            "Energy": calc.tabulate.Energy(),  # yes, in old implementation degen_thresh was applied to qunatities,
            # but not to energies
            "V": calc.tabulate.Velocity(**param_tab),
            "Der_berry": calc.tabulate.DerBerryCurvature(**param_tab),
            "berry": calc.tabulate.BerryCurvature(ibands=[5, 6, 7, 8], **param_tab),
            'spin': calc.tabulate.Spin(**param_tab),
            'spin_berry': calc.tabulate.SpinBerry(**param_tab),
            'morb': calc.tabulate.OrbitalMoment(**param_tab),
            'Der_morb': calc.tabulate.DerOrbitalMoment(**param_tab),
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

    extra_precision = {'Morb': 1e-6, 'Der_berry': 5e-8}
    for quant in result.results.get("tabulate").results.keys():  # ["Energy", "berry","Der_berry","spin","morb"]:
        for comp in result.results.get("tabulate").results.get(quant).get_component_list():
            _quant = "E" if quant == "Energy" else quant
            _comp = "-" + comp if comp != "" else ""
            #            data=result.results.get(quant).data
            #            assert data.shape[0] == len(Efermi)
            #            assert np.all( np.array(data.shape[1:]) == 3)
            prec = extra_precision[quant] if quant in extra_precision else 2e-8
            #            comparer(frmsf_name, quant+_comp+suffix,  suffix_ref=compare_quant(quant)+_comp+suffix_ref ,precision=prec )
            compare_fermisurfer(
                fout_name="berry_Fe_W90-tabulate",
                suffix=_quant + _comp + "-run",
                suffix_ref=_quant + _comp,
                fout_name_ref="tabulate_Fe_W90",
                precision=prec)


def test_Fe_sparse(check_run, system_Fe_W90_sparse, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}

    parameters_optical = dict(
        Efermi=np.array([17.0, 18.0]), omega=np.arange(0.0, 7.1, 1.0), smr_fixed_width=0.20, smr_type="Gaussian")

    calculators['opt_conductivity'] = wberri.calculators.dynamic.OpticalConductivity(**parameters_optical)
    calculators['opt_SHCqiao'] = wberri.calculators.dynamic.SHC(SHC_type="qiao", **parameters_optical)
    calculators['opt_SHCryoo'] = wberri.calculators.dynamic.SHC(SHC_type="ryoo", **parameters_optical)

    check_run(
        system_Fe_W90_sparse,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="run-sparse",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"Morb": -1e-6},
        skip_compare=['tabulate', 'opt_conductivity', 'opt_SHCqiao', 'opt_SHCryoo'])

    for quant in 'opt_conductivity', 'opt_SHCryoo', 'opt_SHCryoo':
        compare_any_result(
            "berry_Fe_W90",
            quant + "-run-sparse",
            0,
            fout_name_ref="kubo_Fe_W90",
            suffix_ref=quant,
            precision=-1e-8,
            result_type=EnergyResult)


def test_Fe_dynamic_noband(check_run, system_Fe_W90, compare_any_result):
    calculators = {}
    parameters_optical = dict(
        Efermi=np.array([117.0, 118.0]), omega=np.arange(0.0, 7.1, 1.0), smr_fixed_width=0.20, smr_type="Gaussian")

    calculators['opt_conductivity'] = wberri.calculators.dynamic.OpticalConductivity(**parameters_optical)
    calculators['opt_SHCqiao'] = wberri.calculators.dynamic.SHC(SHC_type="qiao", **parameters_optical)
    calculators['opt_SHCryoo'] = wberri.calculators.dynamic.SHC(SHC_type="ryoo", **parameters_optical)

    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="run_noband",
        precision=1e-15,
        compare_zero=True)


def test_Fe_wcc(check_run, system_Fe_W90_wcc, compare_any_result):
    param_kwargs = {'Efermi': Efermi_Fe, 'kwargs_formula': {'correction_wcc': True}}
    param = {'Efermi': Efermi_Fe}
    calculators = {}
    for k, v in calculators_Fe.items():
        if k in ['dos', 'cumdos', 'conductivity_ohmic', 'conductivity_ohmic_fsurf', 'spin']:
            calculators[k] = v(**param)
        else:
            calculators[k] = v(**param_kwargs)
    check_run(
        system_Fe_W90_wcc,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="wcc-run",
        use_symmetry=False,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"Morb": -1}
    )


def test_Fe_sym(check_run, system_Fe_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}

    parameters_optical = dict(
        Efermi=np.array([17.0, 18.0]), omega=np.arange(0.0, 7.1, 1.0), smr_fixed_width=0.20, smr_type="Gaussian")

    calculators['opt_conductivity'] = wberri.calculators.dynamic.OpticalConductivity(**parameters_optical)
    calculators['opt_SHCqiao'] = wberri.calculators.dynamic.SHC(SHC_type="qiao", **parameters_optical)
    calculators['opt_SHCryoo'] = wberri.calculators.dynamic.SHC(SHC_type="ryoo", **parameters_optical)

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
        skip_compare=['tabulate', 'opt_conductivity', 'opt_SHCqiao', 'opt_SHCryoo']
            )

    for quant in 'opt_conductivity', 'opt_SHCryoo', 'opt_SHCryoo':
        compare_any_result(
            "berry_Fe_W90",
            quant + "-sym-run",
            0,
            fout_name_ref="kubo_Fe_W90_sym",
            suffix_ref=quant,
            precision=-1e-8,
            result_type=EnergyResult)



def test_Fe_set_spin(check_run, system_Fe_W90_proj_set_spin, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    calculators = {"spin": wberri.calculators.static.Spin(**param),
                   "cumdos": wberri.calculators.static.CumDOS(**param),
                  }

    parameters_shc = dict(
                            Efermi=np.array([Efermi_Fe[0], Efermi_Fe[-1]]), omega=np.arange(0.0, 7.1, 1.0),
                            smr_fixed_width=0.20, smr_type="Gaussian",
                            SHC_type="simple"
                         )
    calculators['opt_SHCsimple'] = wberri.calculators.dynamic.SHC(kwargs_formula={"external_terms": True}, **parameters_shc)
    calculators['opt_SHCsimple_internal'] = wberri.calculators.dynamic.SHC(kwargs_formula={"external_terms": False}, **parameters_shc)

    check_run(
        system_Fe_W90_proj_set_spin,
        calculators,
        fout_name="Fe_set_spin",
        use_symmetry=False,
            )



def test_Fe_FPLO(check_run, system_Fe_FPLO, compare_any_result):
    param = {'Efermi': Efermi_Fe_FPLO}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    check_run(
        system_Fe_FPLO,
        calculators,
        fout_name="berry_Fe_FPLO",
        suffix="-run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Fe_FPLO_wcc(check_run, system_Fe_FPLO_wcc, compare_any_result):
    param = {'Efermi': Efermi_Fe_FPLO}
    param_kwargs = {'Efermi': Efermi_Fe_FPLO, 'kwargs_formula': {"external_terms": False}}
    for k, v in calculators_Fe.items():
        if k in ['ahc', 'ahc_test', 'Morb', 'Morb_test']:
            calculators = {k: v(**param_kwargs)}
        else:
            calculators = {k: v(**param)}
    calculators.update({'spin': calc.static.Spin(**param)})
    check_run(
        system_Fe_FPLO_wcc,
        calculators,
        fout_name="berry_Fe_FPLO",
        suffix="wcc-run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Fe_FPLO_wcc_ext(check_run, system_Fe_FPLO_wcc, compare_any_result):
    param_kwargs = {'Efermi': Efermi_Fe_FPLO, 'kwargs_formula': {
        "internal_terms": False, "external_terms": True}}
    for k, v in calculators_Fe.items():
        if k in ['ahc', 'ahc_test', 'Morb', 'Morb_test']:
            calculators = {k: v(**param_kwargs)}
    check_run(
        system_Fe_FPLO_wcc,
        calculators,
        fout_name="berry_Fe_FPLO",
        suffix="wcc_ext-run",
        precision=1e-8,
        compare_zero=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Fe_FPLO_wcc_sym(check_run, system_Fe_FPLO_wcc, compare_any_result):
    param = {'Efermi': Efermi_Fe_FPLO}
    param_kwargs = {'Efermi': Efermi_Fe_FPLO, 'kwargs_formula': {"external_terms": False}}
    for k, v in calculators_Fe.items():
        if k in ['ahc', 'ahc_test', 'Morb', 'Morb_test']:
            calculators = {k: v(**param_kwargs)}
        else:
            calculators = {k: v(**param)}
    calculators.update({'spin': calc.static.Spin(**param)})
    check_run(
        system_Fe_FPLO_wcc,
        calculators,
        fout_name="berry_Fe_FPLO",
        suffix="wcc-sym-run",
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


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



def test_Fe_sym_refine(check_run, system_Fe_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items() if k != 'spin'}
    # We do not include spin here, because it was not there in the beginning
    # adding another calculator may change the behaviour of the refinement procedure, and hence we would
    # have to replace reference files for all calculators
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="sym-run",
        suffix_ref="sym",
        adpt_num_iter=3,
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Fe_pickle_Klist_12(check_run, system_Fe_W90, compare_any_result):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    #  First, remove the
    try:
        os.remove("Klist.pickle")
    except FileNotFoundError:
        pass
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items() if k != 'spin'}
    # We do not include spin here, because it was not there in the beginning
    # adding another calculator may change the behaviour of the refinement procedure, and hence we would
    # have to replace reference files for all calculators
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="pickle-run",
        suffix_ref="sym",
        adpt_num_iter=1,
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
        adpt_num_iter=2,
        use_symmetry=True,
        file_Klist="Klist.pickle",
        restart=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )


def test_Fe_pickle_Klist_021(check_run, system_Fe_W90, compare_any_result):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    #  First, remove the
    try:
        os.remove("Klist.pickle")
    except FileNotFoundError:
        pass
    param = {'Efermi': Efermi_Fe}
    calculators = {k: v(**param) for k, v in calculators_Fe.items() if k != 'spin'}
    # We do not include spin here, because it was not there in the beginning
    # adding another calculator may change the behaviour of the refinement procedure, and hence we would
    # have to replace reference files for all calculators
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="pickle-run-fch",
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
        suffix="pickle-run-fch",
        suffix_ref="sym",
        adpt_num_iter=2,
        use_symmetry=True,
        file_Klist="Klist.pickle",
        restart=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )
    check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="pickle-run-fch",
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
#    calculators.update({k: v(**param) for k, v in calculators_GaAs.items()})
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    calculators.update({
        'gyrotropic_Korb': calc.static.GME_orb_FermiSea(Efermi=Efermi_GaAs),
        'gyrotropic_Kspin': calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
        'gyrotropic_Kspin_fsurf': calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
        'gyrotropic_Korb_test': calc.static.GME_orb_FermiSea_test(Efermi=Efermi_GaAs), }
            )

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


def test_GaAs_tb(check_run, system_GaAs_tb, compare_any_result):

    param = {'Efermi': Efermi_GaAs}
    calculators = {k: v(**param) for k, v in calculators_GaAs.items()}

    check_run(
        system_GaAs_tb,
        calculators,
        fout_name="berry_GaAs_tb",
        suffix="run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"berry_dipole_fsurf": 1e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_GaAs_wcc(check_run, system_GaAs_W90_wcc, compare_any_result):
    param = {'Efermi': Efermi_GaAs}
    calculators = {k: v(**param) for k, v in calculators_GaAs.items()}
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})

    calculators.update({
        'gyrotropic_Kspin': calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
        'gyrotropic_Kspin_fsurf': calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
            })

    check_run(
        system_GaAs_W90_wcc,
        calculators,
        fout_name="berry_GaAs_W90",
        suffix="wcc-run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"berry_dipole_fsurf": 1e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem


def test_GaAs_tb_wcc(check_run, system_GaAs_tb_wcc, compare_any_result):

    param = {'Efermi': Efermi_GaAs}
    calculators = {k: v(**param) for k, v in calculators_GaAs.items()}

    check_run(
        system_GaAs_tb_wcc,
        calculators,
        fout_name="berry_GaAs_tb",
        suffix="wcc-run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"berry_dipole_fsurf": 1e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_GaAs_tb_wcc_ws(check_run, system_GaAs_tb_wcc_ws, compare_any_result):

    param = {'Efermi': Efermi_GaAs}
    calculators = {k: v(**param) for k, v in calculators_GaAs_internal.items()}

    check_run(
        system_GaAs_tb_wcc_ws,
        calculators,
        fout_name="berry_GaAs_W90",
        suffix="tb_wcc-run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"conductivity_ohmic": -2e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing



def test_Haldane_PythTB(check_run, system_Haldane_PythTB, compare_any_result):

    param = {'Efermi': Efermi_Haldane}
    calculators = {k: v(**param) for k, v in calculators_Haldane.items()}

    check_run(
        system_Haldane_PythTB,
        calculators,
        fout_name="berry_Haldane_tbmodels",
        suffix="pythtb-run",
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_GaAs_dynamic(check_run, system_GaAs_W90, compare_any_result):
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
        injection_current=calc.dynamic.InjectionCurrent(**param)
    )

    check_run(
        system_GaAs_W90,
        calculators,
        fout_name="dynamic_GaAs_W90",
        grid_param={
            'NK': [6, 6, 6],
            'NKFFT': [3, 3, 3]
        },
        extra_precision=dict(shift_current=1e-9, injection_current=1e-7)
    )



def test_Haldane_TBmodels(check_run, system_Haldane_TBmodels, compare_any_result):

    param = {'Efermi': Efermi_Haldane}
    calculators = {k: v(**param) for k, v in calculators_Haldane.items()}

    check_run(
        system_Haldane_TBmodels,
        calculators,
        fout_name="berry_Haldane_tbmodels",
        suffix="run",
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        }
    )


def test_Haldane_TBmodels_internal(check_run, system_Haldane_TBmodels_internal, compare_any_result):

    param_kwargs = {'Efermi': Efermi_Haldane, 'kwargs_formula': {"external_terms": False}}
    param = {'Efermi': Efermi_Haldane}
    for k, v in calculators_Haldane.items():
        if k == 'ahc':
            calculators = {k: v(**param_kwargs)}
        else:
            calculators = {k: v(**param)}

    check_run(
        system_Haldane_TBmodels_internal,
        calculators,
        fout_name="berry_Haldane_tbmodels",
        suffix="internal-run",
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        }
    )


def test_Haldane_TBmodels_external(check_run, system_Haldane_TBmodels, compare_any_result):

    check_run(
        system_Haldane_TBmodels,
        {'ahc': calc.static.AHC(Efermi=Efermi_Haldane, kwargs_formula={"internal_terms": False})},
        fout_name="berry_Haldane_tbmodels",
        suffix="wcc_external-run",
        precision=1e-8,
        compare_zero=True,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        }
    )


def test_Haldane_PythTB_sym(check_run, system_Haldane_PythTB, compare_any_result):

    param = {'Efermi': Efermi_Haldane}
    calculators = {k: v(**param) for k, v in calculators_Haldane.items()}

    check_run(
        system_Haldane_PythTB,
        calculators,
        fout_name="berry_Haldane_tbmodels",
        suffix="pythtb_sym-run",
        use_symmetry=True,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        }
    )


def test_Haldane_TBmodels_sym(check_run, system_Haldane_TBmodels, compare_any_result):

    param = {'Efermi': Efermi_Haldane}
    calculators = {k: v(**param) for k, v in calculators_Haldane.items()}

    check_run(
        system_Haldane_TBmodels,
        calculators,
        fout_name="berry_Haldane_tbmodels",
        suffix="sym-run",
        use_symmetry=True,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        }
    )


def test_Haldane_TBmodels_sym_refine(check_run, system_Haldane_TBmodels, compare_any_result):

    param = {'Efermi': Efermi_Haldane}
    calculators = {k: v(**param) for k, v in calculators_Haldane.items()}

    check_run(
        system_Haldane_TBmodels,
        calculators,
        fout_name="berry_Haldane_tbmodels",
        suffix="sym-run",
        suffix_ref="sym",
        adpt_num_iter=1,
        use_symmetry=True,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        }
    )


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
        skip_compare=["opt_shiftcurrent"],
        use_symmetry=False,  # !!! temporary
        extra_precision={"Morb": -1e-6},
    )
    # for quant in calculators_Chiral.keys():#["conductivity_ohmic", "berry_dipole", "ahc"]:
    for quant in ["conductivity_ohmic", "berry_dipole", "ahc"]:
        compare_energyresult(
                fout_name="berry_Chiral",
                suffix=quant + "-left-run",
                adpt_num_iter=0,
                suffix_ref=quant,
                mode="txt",
                compare_smooth=True,
                precision=-1e-8)

#        skip_compare=['tabulate', 'opt_conductivity', 'opt_SHCqiao', 'opt_SHCryoo'])

    for quant in 'opt_conductivity', "opt_shiftcurrent":  # 'opt_SHCryoo', 'opt_SHCryoo':
        compare_any_result(
            "berry_Chiral",
            quant + "-left-run",
            0,
            fout_name_ref="kubo_Chiral",
            suffix_ref=quant,
            precision=-1e-8,
            result_type=EnergyResult)




def test_Chiral_left_tetra(check_run, system_Chiral_left, compare_any_result):
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    result_full = check_run(
        system_Chiral_left,
        calculators_Chiral_tetra,
        fout_name="berry_Chiral_tetra",
        suffix="left-run",
        grid_param=grid_param,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=True,
        extra_precision={"Morb": -1e-6},
    )

    result_half = check_run(
        system_Chiral_left,
        calculators_Chiral_tetra_half,
        fout_name="berry_Chiral_tetra",
        suffix="left-run",
        grid_param=grid_param,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=True,
        extra_precision={"Morb": -1e-6},
        do_not_compare=True,
    )

    for key, res_half in result_half.results.items():
        print(key)
        res_full = result_full.results[key]
        nef = len(res_half.Energies[0])
        data1 = res_half.data
        data2 = res_full.data[:nef]
        assert data1.shape == data2.shape
        precision = 1e-14 * np.max(abs(data1))
        assert data1 == approx(
                data2, abs=precision), (
                        f"calculated data of {key}  of full and half sets of Fermi levels give a maximal " +
                        "absolute difference of {abs_err} greater than the required precision {required_precision}. ".format(
                            abs_err=np.max(abs(data1 - data2)), required_precision=precision))



@pytest.mark.parametrize("tetra", [True, False])
@pytest.mark.parametrize("use_sym", [True, False])
def test_Chiral_left_tab_static(check_run, system_Chiral_left, use_sym, tetra):
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    param = dict(Efermi=Efermi_Chiral, tetra=tetra, kwargs_formula={"external_terms": False})
    system = system_Chiral_left

    calculators = {"AHC": calc.static.AHC(**param),
                "Morb": calc.static.Morb(**param)
                    }
    calculators["tabulate"] = calc.TabulatorAll(
        {
            "AHC": calc.static.AHC(**param, k_resolved=True),
            "Morb": calc.static.Morb(**param, k_resolved=True)
        },
        mode="grid",
        ibands=(0, 1))

    result = check_run(
        system,
        calculators,
        fout_name="berry_Chiral_static_tab",
        suffix="",
        grid_param=grid_param,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=use_sym,
        do_not_compare=True
    )

    print(result.results.keys())
    for key in "AHC", "Morb":
        print(key)
        data_int = result.results[key].data
        data_tab = result.results["tabulate"].results[key].data
        data_tab_int = data_tab.mean(axis=0)
        assert data_tab_int.shape == data_int.shape
        prec = 1e-8 * np.max(abs(data_int))
        assert abs(data_tab_int - data_int).max() < prec


@pytest.mark.parametrize("tetra", [True, False])
@pytest.mark.parametrize("use_sym", [True, False])
def test_Haldane_tab_static(check_run, system_Haldane_PythTB, use_sym, tetra):
    grid_param = {'NK': [10, 10, 1], 'NKFFT': [5, 5, 1]}
    param = dict(Efermi=Efermi_Haldane, tetra=tetra, kwargs_formula={"external_terms": False})
    system = system_Haldane_PythTB

    calculators = {"AHC": calc.static.AHC(**param),
                "Morb": calc.static.AHC(**param)
                    }
    calculators["tabulate"] = calc.TabulatorAll(
        {
            "AHC": calc.static.AHC(**param, k_resolved=True),
            "Morb": calc.static.AHC(**param, k_resolved=True),
            "berry": calc.tabulate.BerryCurvature(kwargs_formula={"external_terms": False})
        },
        mode="grid",
        ibands=(0,))


    result = check_run(
        system,
        calculators,
        fout_name="berry_Haldane_static_tab",
        suffix="",
        grid_param=grid_param,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=use_sym,
        do_not_compare=True
    )

    print(result.results.keys())
    for key in "AHC", "Morb":
        print(key)
        data_int = result.results[key].data
        data_tab = result.results["tabulate"].results[key].data
        data_tab_int = data_tab.mean(axis=0)
        assert data_tab_int.shape == data_int.shape
        prec = 1e-8 * np.max(abs(data_int))
        assert abs(data_tab_int - data_int).max() < prec

    iEF = np.argmin(abs(Efermi_Haldane))
    ahc_k = result.results["tabulate"].results["AHC"].data[:, iEF]
    berry_k = result.results["tabulate"].results["berry"].data[:, 0] * wberri.__factors.factor_ahc / system.cell_volume
    prec = 1e-8 * np.max(abs(berry_k))
    assert abs(berry_k - ahc_k).max() < prec



def test_Chiral_left_tetra_tetragrid(check_run, system_Chiral_left, compare_any_result):
    grid = wberri.grid.GridTetra(system_Chiral_left, length=8, NKFFT=[5, 5, 2])
    check_run(
        system_Chiral_left,
        calculators_Chiral_tetra,
        fout_name="berry_Chiral_tetragrid",
        suffix="",
        grid=grid,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=True,
        extra_precision={"Morb": -1e-6},
    )


def test_Chiral_left_tetra_2EF(check_run, system_Chiral_left, compare_any_result):
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    nshift = 4
    Efermi_shift = Efermi_Chiral + Efermi_Chiral[nshift] - Efermi_Chiral[0]
    calculators = {
        'dos': calc.static.DOS(Efermi=Efermi_Chiral, tetra=True),
        'dos_trig': calc.static.DOS(Efermi=Efermi_Chiral, tetra=True),
        'dos_trig_2': calc.static.DOS(Efermi=Efermi_shift, tetra=True),
    }
    result = check_run(
        system_Chiral_left,
        calculators,
        fout_name="berry_Chiral_tetra_trigonal",
        suffix="left-run-2",
        grid_param=grid_param,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=True,
        skip_compare='dos_trig_2'
    )
    data1 = result.results.get("dos_trig_2").data[:-nshift]
    data2 = result.results.get("dos_trig").data[nshift:]
    assert data1.shape == data2.shape
    assert data1 == approx(data2), "the result with the shifted set of Fermi levels is different by {}".format(
            np.max(np.abs(data1 - data2)))


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


def test_CuMnAs_PT(check_run, system_CuMnAs_2d_broken, compare_any_result):
    "check that for flipped chirality the ohmic conductivity is the same, but the Berry dipole is opposite"

    degen_param = [('degen_thresh', 0.05), ('degen_Kramers', True)]
    calculators = {}
    for tetra in True, False:
        for degen in degen_param:
            param_kwargs = {'Efermi': Efermi_CuMnAs_2d, 'tetra': tetra, degen[0]: degen[1], 'kwargs_formula': {'external_terms': False}}
            param = {'Efermi': Efermi_CuMnAs_2d, 'tetra': tetra, degen[0]: degen[1]}
            label = f"-{tetra}-{degen[0]}"
            print(param)
            print(label)
            for k, v in calculators_CuMnAs_2d.items():
                if k == 'Hall_morb_fsurf':
                    calculators.update({k + label: v(**param_kwargs)})
                else:
                    calculators.update({k + label: v(**param)})

    degen_param = [('degen_thresh', 0.05), ('degen_Kramers', True)]
    results = check_run(
            system_CuMnAs_2d_broken,
            calculators,
            fout_name="berry_CuMnAs_2d",
            grid_param={
                'NK': [10, 10, 1],
                'NKFFT': [5, 5, 1]},
            use_symmetry=True,
            do_not_compare=True,)

    for tetra in True, False:
        for k in calculators_CuMnAs_2d.keys():
            label1 = k + f"-{tetra}-{degen_param[0][0]}"
            label2 = k + f"-{tetra}-{degen_param[1][0]}"
            data1 = results.results[label1].data
            data2 = results.results[label2].data
            assert data1.shape == data2.shape
            assert np.all(np.array(data1.shape[1:]) == 3)
            assert np.all(np.array(data2.shape[1:]) == 3)
            precision = 1e-14 * np.max(abs(data1))
            assert data1 == approx(
                data2, abs=precision), (
                        f"calcuylated data of {label1}  and {label2} give a maximal " +
                        "absolute difference of {abs_err} greater than the required precision {required_precision}. ".format(
                            abs_err=np.max(abs(data1 - data2)), required_precision=precision))



def test_Te_ASE(check_run, system_Te_ASE, data_Te_ASE, compare_any_result):
    param = {'Efermi': Efermi_Te_gpaw, "tetra": True, 'use_factor': False}
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


def test_Te_ASE_wcc(check_run, system_Te_ASE_wcc, data_Te_ASE, compare_any_result):
    param = {'Efermi': Efermi_Te_gpaw, "tetra": True, 'use_factor': False}
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
                            "Energy": wberri.calculators.tabulate.Energy(),
                            "berry": wberri.calculators.tabulate.BerryCurvature(kwargs_formula={"external_terms": False}),
                                  }

    calculators["tabulate"] = wberri.calculators.TabulatorAll(quantities,
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


    tab_result = run_result.results["tabulate"]
    filename = "path_tab.pickle"
    fout = open(os.path.join(OUTPUT_DIR, filename + "-run"), "wb")

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
                f"tabulation along path gave a wrong result for quantity {quant} component {comp} " +
                "with a maximal difference {}".format(max(abs(data - data_ref))))

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


def test_tabulate_fail(system_Haldane_PythTB):

    k_nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, dk=1.0)

    quantities = {
                            "Energy": wberri.calculators.tabulate.Energy(),
                            "berry": wberri.calculators.tabulate.BerryCurvature(kwargs_formula={"external_terms": False}),
                                  }

    key = "tabulate_grid"
    calculators_fail = {"tabulate_grid": wberri.calculators.TabulatorAll(quantities,
                                                       ibands=[0],
                                                        mode="grid"),
                        "ahc": wberri.calculators.static.AHC(Efermi=Efermi_Haldane)}
    for key, val in calculators_fail.items():
        with pytest.raises(ValueError, match=f"Calculation along a Path is running, but calculator `{key}` is not compatible with a Path"):
            wberri.run(system=system_Haldane_PythTB, grid=path, calculators={key: val})


def test_shc_static(check_run, system_Fe_W90):
    "Test whether SHC static and dynamic calculators are the same at omega=0"

    parameters_optical = dict(Efermi=np.array([17.0, 18.0]),
            omega=np.arange(0.0, 7.1, 1.0), smr_fixed_width=1e-10, smr_type="Gaussian", kBT=0)
    parameters_static = dict(Efermi=np.array([17.0, 18.0]))

    calculators = {}

    calculators['SHCqiao_dynamic'] = wberri.calculators.dynamic.SHC(SHC_type="qiao", **parameters_optical)
    calculators['SHCryoo_dynamic'] = wberri.calculators.dynamic.SHC(SHC_type="ryoo", **parameters_optical)
    calculators['SHCqiao_static'] = wberri.calculators.static.SHC(kwargs_formula={'spin_current_type': 'qiao'}, **parameters_static)
    calculators['SHCryoo_static'] = wberri.calculators.static.SHC(kwargs_formula={'spin_current_type': 'ryoo'}, **parameters_static)


    result = check_run(
        system_Fe_W90,
        calculators,
        fout_name="shc_Fe_W90",
        suffix="run",
        do_not_compare=True)


    for mode in ["qiao", "ryoo"]:
        data_static = result.results[f"SHC{mode}_static"].data
        data_dynamic = result.results[f"SHC{mode}_dynamic"].data[:, 0, ...].real
        precision = max(np.average(abs(data_static) / 1E10), 1E-8)
        assert data_static == approx(
            data_dynamic, abs=precision), (
                f"data of"
                f"SHC {mode} from static.SHC and dynamic.SHC give a maximal absolute"
                f"difference of {np.max(np.abs(data_static - data_dynamic))}.")



def test_phonons_GaAs_tetra(check_run, system_Phonons_GaAs):
    """test  dos, cumdos for phonons"""

    calculators = {k: cal(tetra=True, Efermi=omega_phonon) for k, cal in calculators_phonons.items()}
    check_run(
        system_Phonons_GaAs,
        calculators,
        fout_name="phonons_GaAs_tetra",
        use_symmetry=True,
    )


def test_factor_nlahc(check_run, system_GaAs_W90):
    "Test whether constant_factor for NLAHC works as expected"

    from wannierberri.__factors import factor_nlahc

    calculators = dict(
        bcd=calc.static.BerryDipole_FermiSurf(Efermi=Efermi_GaAs),
        nlahc=calc.static.NLAHC_FermiSurf(Efermi=Efermi_GaAs),
        nlahc_no_factor=calc.static.NLAHC_FermiSurf(Efermi=Efermi_GaAs, use_factor=False),
    )

    result = check_run(
        system_GaAs_W90,
        calculators,
        fout_name="berry_GaAs_W90_factor",
        do_not_compare=True,
    )

    data_bcd = result.results["bcd"].data
    data_nlahc = result.results["nlahc"].data
    data_nlahc_no_factor = result.results["nlahc_no_factor"].data
    precision = max(np.average(abs(data_bcd) / 1E10), 1E-8)

    assert data_nlahc_no_factor == approx(
        data_bcd, abs=precision), (
            f"data of"
            f"BerryDipole and NLAHC with no units give a maximal absolute"
            f"difference of {np.max(np.abs(data_nlahc_no_factor - data_bcd))}.")

    assert data_nlahc == approx(
        data_bcd * factor_nlahc, abs=precision), (
            f"data of"
            f"BerryDipole times factor_nlahc and NLAHC give a maximal absolute"
            f"difference of {np.max(np.abs(data_nlahc - data_bcd * factor_nlahc))}.")


@pytest.fixture
def check_kp_mass_isotropic(check_run):
    def _inner(system, name, suffix, check_anal=False):

        Efermi = np.linspace(-0.1, 0.5, 101)
        tetra = True
        calculators = {
            'cumdos': calc.static.CumDOS(Efermi=Efermi, tetra=tetra),
            'dos': calc.static.DOS(Efermi=Efermi, tetra=tetra),
            'ohmic_sea': calc.static.Ohmic_FermiSea(Efermi=Efermi, tetra=tetra),
            'ohmic_surf': calc.static.Ohmic_FermiSurf(Efermi=Efermi, tetra=tetra),
                    }

        result = check_run(
                    system,
                    calculators,
                    grid_param={
                        'length': 20,
                        'NKFFT': 5
                                },
                    fout_name="berry_kp_mass_" + name,
                    suffix=suffix,
                            )

        if check_anal:
            cumdos = result.results["cumdos"].data / system.cell_volume
            dos = result.results["dos"].data / system.cell_volume
            ohmic = {}
            ohmic["sea"] = result.results["ohmic_sea"].data / wberri.__factors.factor_ohmic
            ohmic["surf"] = result.results["ohmic_surf"].data / wberri.__factors.factor_ohmic

            precision = 1e-8
            try:
                for ss in "sea", "surf":
                    ohm = ohmic[ss]
                    for i in range(3):
                        for j in range(3):
                            if i == j:
                                assert ohm[:, i, j] == approx(ohm[:, 0, 0], abs=precision)
                            else:
                                assert ohm[:, i, j] == approx(0, abs=precision)
            except AssertionError:
                raise RuntimeError(f"ohmic({ss}) conductivity is not isotropic, componenets {i,j}")
            ohmic["sea"] = ohmic["sea"][:, 0, 0]
            ohmic["surf"] = ohmic["surf"][:, 0, 0]


            select_plus = Efermi > 0.2
            select_minus = Efermi < 0

            assert dos[select_minus] == approx(0, abs=precision)
            assert cumdos[select_minus] == approx(0, abs=precision)
            assert ohmic["sea"][select_minus] == approx(0, abs=precision)
            assert ohmic["surf"][select_minus] == approx(0, abs=precision)

            # compare with results evaluated analytically
            mass = mass_kp_iso
            Efp = Efermi[select_plus]
            dos_anal = mass * np.sqrt(2 * Efp * mass) / (2 * np.pi**2)
            cumdos_anal = np.sqrt(2 * Efp * mass)**3 / (6 * np.pi**2)
            ohmic_anal = np.sqrt(2 * Efp * mass)**3 / (6 * np.pi**2) / mass
            precision = 0.05
            assert dos[select_plus] == approx(dos_anal, rel=precision)
            assert cumdos[select_plus] == approx(cumdos_anal, rel=precision)
            assert ohmic["sea"][select_plus] == approx(ohmic_anal, rel=precision)
            assert ohmic["surf"][select_plus] == approx(ohmic_anal, rel=precision)
    return _inner


def test_kp_mass_isotropic_0(check_kp_mass_isotropic, system_kp_mass_iso_0):
    check_kp_mass_isotropic(system_kp_mass_iso_0, "iso", "0", check_anal=True)


def test_kp_mass_isotropic_1(check_kp_mass_isotropic, system_kp_mass_iso_1):
    check_kp_mass_isotropic(system_kp_mass_iso_1, "iso", "1", check_anal=True)


def test_kp_mass_isotropic_2(check_kp_mass_isotropic, system_kp_mass_iso_2):
    check_kp_mass_isotropic(system_kp_mass_iso_2, "iso", "2", check_anal=True)



def test_kp_mass_anisotropic_0(check_kp_mass_isotropic, system_kp_mass_aniso_0):
    check_kp_mass_isotropic(system_kp_mass_aniso_0, "aniso", "0", check_anal=False)


def test_kp_mass_anisotropic_1(check_kp_mass_isotropic, system_kp_mass_aniso_1):
    check_kp_mass_isotropic(system_kp_mass_aniso_1, "aniso", "1", check_anal=False)


def test_kp_mass_anisotropic_2(check_kp_mass_isotropic, system_kp_mass_aniso_2):
    check_kp_mass_isotropic(system_kp_mass_aniso_2, "aniso", "2", check_anal=False)
