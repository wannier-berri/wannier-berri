"""Test the anomalous Hall conductivity."""
import os

import pytest
import wannierberri as wberri
from wannierberri.formula import covariant as frml
from wannierberri import formula

from common import OUTPUT_DIR


def get_component_list(quantity):
    if quantity in ["E", "Energy"]:
        return [None]
    elif quantity in ["berry", "V", "morb","spin"]:
        return [a for a in "xyz"]
    elif quantity in ["Der_berry", "Der_morb"]:
        return [a + b for a in "xyz" for b in "xyz"]
    elif quantity in ["spin_berry"]:
        return [a + b + c for a in "xyz" for b in "xyz" for c in "xyz"]
    elif quantity == "omega2":
        return ["zz"]
    else:
        raise ValueError(f"unknown quantity {quantity}")


@pytest.fixture
def check_tabulate(parallel_serial, compare_fermisurfer):

    def _inner(
            system,
            quantities=[],
            user_quantities={},
            frmsf_name="tabulate",
            comparer=compare_fermisurfer,
            parallel=parallel_serial,
            numproc=0,
            grid_param={
                'NK': [6, 6, 6],
                'NKFFT': [3, 3, 3]
            },
            degen_thresh=5e-2,
            degen_Kramers=False,
            use_symmetry=False,
            parameters={},
            parameters_K={},
            specific_parameters={},
            suffix="",
            suffix_ref="",
            extra_precision={},
            ibands=None):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.tabulate(
            system,
            grid=grid,
            quantities=quantities,
            user_quantities=user_quantities,
            parallel=parallel,
            parameters=parameters,
            specific_parameters=specific_parameters,
            ibands=ibands,
            use_irred_kpt=use_symmetry,
            symmetrize=use_symmetry,
            parameters_K=parameters_K,
            frmsf_name=os.path.join(OUTPUT_DIR, frmsf_name),
            suffix=suffix,
            degen_thresh=degen_thresh,
            degen_Kramers=degen_Kramers)

        if len(suffix) > 0:
            suffix = "-" + suffix
        if len(suffix_ref) > 0:
            suffix_ref = "-" + suffix_ref

        for quant in ["E"] + quantities + list(user_quantities.keys()):
            for comp in get_component_list(quant):
                _comp = "-" + comp if comp is not None else ""
                #            data=result.results.get(quant).data
                #            assert data.shape[0] == len(Efermi)
                #            assert np.all( np.array(data.shape[1:]) == 3)
                prec = extra_precision[quant] if quant in extra_precision else None
                comparer(
                    frmsf_name,
                    quant + _comp + suffix,
                    suffix_ref=compare_quant(quant) + _comp + suffix_ref,
                    precision=prec)
        return result

    return _inner


quantities_tab = ['V', 'berry', 'Der_berry', 'morb', 'Der_morb']


def compare_quant(quant):
    #    compare= {'ahc_ocean':'ahc','ahc3_ocean':'ahc',"cumdos3_ocean":"cumdos","dos3_ocean":"dos","berry_dipole_ocean":"berry_dipole","berry_dipole3_ocean":"berry_dipole",
    #            'conductivity_ohmic3_ocean':'conductivity_ohmic','conductivity_ohmic_fsurf3_ocean':'conductivity_ohmic_fsurf'}
    compare = {}
    if quant in compare:
        return compare[quant]
    else:
        return quant


def test_Fe(check_tabulate, system_Fe_W90, compare_fermisurfer):
    """Test Energies, Velocities, berry curvature, its derivative"""
    result = check_tabulate(
        system_Fe_W90,
        quantities_tab + ["spin_berry","spin"],
        frmsf_name="tabulate_Fe_W90",
        suffix="",
        comparer=compare_fermisurfer,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        ibands=[5, 6, 7, 8],
        extra_precision={
            'berry': 1e-4,
            "Der_berry": 1e-4,
            'morb': 1e-4,
            "Der_morb": 1e-4,
            "spin_berry": 1e-4
        })

    xyz = ["x", "y", "z"]
    print(result.results.keys())
    assert result.results["Energy"].get_component_list() == [""]
    assert result.results["berry"].get_component_list() == xyz
    assert result.results["Der_berry"].get_component_list() == [a + b for a in xyz for b in xyz]
    assert result.results["spin_berry"].get_component_list() == [a + b + c for a in xyz for b in xyz for c in xyz]


def test_Fe_user(check_tabulate, system_Fe_W90, compare_fermisurfer):
    """Test Energies, Velocities, berry curvature, its derivative"""
    calculators = {
        'V': frml.Velocity,
        'berry': frml.Omega,  #berry.calcImf_band_kn ,
        'Der_berry': frml.DerOmega,  #berry.calcImf_band_kn ,
    }

    check_tabulate(
        system_Fe_W90,
        user_quantities=calculators,
        frmsf_name="tabulate_Fe_W90",
        suffix="user",
        comparer=compare_fermisurfer,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        ibands=[5, 6, 7, 8],
        extra_precision={
            'berry': 1e-4,
            "Der_berry": 1e-4
        })


def test_Chiral(check_tabulate, system_Chiral_left, compare_fermisurfer):
    """Test Energies, Velocities, berry curvature, its derivative"""
    check_tabulate(
        system_Chiral_left,
        quantities_tab,
        frmsf_name="tabulate_Chiral",
        suffix="",
        comparer=compare_fermisurfer,
        parameters={'external_terms': False},
        ibands=[0, 1])


def test_Chiral_sym(check_tabulate, system_Chiral_left, compare_fermisurfer):
    """Test Energies, Velocities, berry curvature, its derivative"""
    check_tabulate(
        system_Chiral_left,
        quantities_tab,
        frmsf_name="tabulate_Chiral",
        suffix="sym",
        comparer=compare_fermisurfer,
        use_symmetry=True,
        parameters={'external_terms': False},
        ibands=[0, 1])


def test_CuMnAs_PT(check_tabulate, system_CuMnAs_2d_broken, compare_fermisurfer):
    """Test tabulation of user-defined quantities
    Also test Kramers degen, by comparing  degen_thresh=0.05 and degen_Kramers=True (they should give the same result)"""

    class Omega2(formula.FormulaProduct):

        def __init__(self, data_K, **parameters):
            print("parameters of omega2", parameters)
            omega = frml.Omega(data_K, **parameters)
            omega2 = formula.FormulaProduct([omega, omega])
            self.__dict__.update(omega2.__dict__)

    check_tabulate(
        system_CuMnAs_2d_broken,
        user_quantities={"omega2": Omega2},
        frmsf_name="tabulate_CuMnAs",
        suffix="thresh",
        comparer=compare_fermisurfer,
        specific_parameters={"omega2": {
            "external_terms": False
        }},
        degen_thresh=0.05,
        degen_Kramers=False,
        ibands=[0, 1, 2, 3],
        extra_precision={'omega2': 1e-8})

    check_tabulate(
        system_CuMnAs_2d_broken,
        user_quantities={"omega2": Omega2},
        frmsf_name="tabulate_CuMnAs",
        suffix="Kramers",
        comparer=compare_fermisurfer,
        #               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
        specific_parameters={"omega2": {
            "external_terms": False
        }},
        degen_thresh=-1,
        degen_Kramers=True,
        ibands=[0, 1, 2, 3],
        extra_precision={'omega2': 1e-8})
