"""Test the Kubo module."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri

from common import OUTPUT_DIR
from common_comparers import compare_quant
from common_systems import symmetries_GaAs, Efermi_Chiral,omega_chiral


@pytest.fixture
def check_integrate_dynamical():
    """
    This function is similar to check_integrate, but the difference is 1) the shape of the
    data are different for dynamical quantities (has omega index), and 2) opt_conductivity
    requires a special treatment because of sym and asym data.
    """

    def _inner(
            system,
            quantities,
            fout_name,
            Efermi,
            omega,
            grid_param,
            comparer,
            additional_parameters={},
            specific_parameters={},
            parameters_K={},
            adpt_num_iter=0,
            use_symmetry=False,
            suffix="",
            suffix_ref="",
            mode="bin",
            extra_precision={}):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.integrate(
            system,
            grid=grid,
            Efermi=Efermi,
            omega=omega,
            quantities=quantities,
            use_irred_kpt=use_symmetry,
            symmetrize=use_symmetry,
            adpt_num_iter=adpt_num_iter,
            parameters=additional_parameters,
            specific_parameters=specific_parameters,
            parameters_K=parameters_K,
            fout_name=os.path.join(OUTPUT_DIR, fout_name),
            write_txt=(mode == "txt"),
            write_bin=(mode == "bin"),
            suffix=suffix,
            restart=False,
        )
        if len(suffix) > 0:
            suffix = "-" + suffix
        if len(suffix_ref) > 0:
            suffix_ref = "-" + suffix_ref

        # Test results output
        for quant in quantities:
            if quant == "opt_conductivity^sep":
                data_list = [result.results[quant].results[s].data for s in ["sym", "asym"]]
            else:
                data_list = [result.results[quant].data]
            for data in data_list:
                assert data.shape[0] == len(Efermi)
                assert data.shape[1] == len(omega)
                assert all(i == 3 for i in data.shape[2:])

        # Test file output
        quantities_compare = quantities.copy()
        if "opt_conductivity^sep" in quantities:
            quantities_compare += ["opt_conductivity^sep-sym", "opt_conductivity^sep-asym"]
            quantities_compare.remove("opt_conductivity^sep")
        if comparer:
            for quant in quantities_compare:
                prec = extra_precision[quant] if quant in extra_precision else None
                comparer(
                    fout_name,
                    quant + suffix,
                    adpt_num_iter,
                    suffix_ref=compare_quant(quant) + suffix_ref,
                    precision=prec,
                    mode=mode)

        return result

    return _inner


def test_optical(check_integrate_dynamical, system_Fe_W90, compare_energyresult, compare_sym_asym):
    """Test optical properties: optical conductivity and spin Hall conductivity
    without use of symmetries and withou adaptive refinement"""
    quantities = ["opt_conductivity", "opt_conductivity^sep", "opt_SHCqiao", "opt_SHCryoo"]

    Efermi = np.array([17.0, 18.0])
    omega = np.arange(0.0, 7.1, 1.0)
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian")
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 0
    specific_parameters = {"opt_conductivity^sep": {"sep_sym_asym": True}}

    check_integrate_dynamical(
        system_Fe_W90,
        quantities,
        fout_name="kubo_Fe_W90",
        Efermi=Efermi,
        omega=omega,
        grid_param=grid_param,
        adpt_num_iter=adpt_num_iter,
        comparer=compare_energyresult,
        additional_parameters=kubo_params,
        specific_parameters=specific_parameters)

    compare_sym_asym("kubo_Fe_W90")


def test_optical_Chiral(check_integrate_dynamical, system_Chiral_left, compare_energyresult):
    """Test optical properties: optical conductivity and spin Hall conductivity
    without use of symmetries and withou adaptive refinement
    without external terms, for the Chiral model
    """
    quantities = ["opt_conductivity","opt_shiftcurrent"]

    Efermi = Efermi_Chiral
    omega = omega_chiral
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian", external_terms=False, sc_eta=0.10)
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    adpt_num_iter = 0

    check_integrate_dynamical(
        system_Chiral_left,
        quantities,
        fout_name="kubo_Chiral",
        Efermi=Efermi,
        omega=omega,
        grid_param=grid_param,
        adpt_num_iter=adpt_num_iter,
        comparer=compare_energyresult,
        additional_parameters=kubo_params,
            )


def test_optical_Chiral_sym(check_integrate_dynamical, system_Chiral_left, compare_energyresult):
    """Test optical properties: optical conductivity and spin Hall conductivity
    without use of symmetries and withou adaptive refinement
    without external terms, for the Chiral model
    """
    quantities = ["opt_conductivity","opt_shiftcurrent"]

    #parameters_Chiral_optical = dict(
    #        Efermi=Efermi_Chiral, omega=omega_chiral, smr_fixed_width=0.20, smr_type="Gaussian" ,
    #kwargs_formula={"external_terms": False }, )

    Efermi = Efermi_Chiral
    omega = omega_chiral
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian", external_terms=False, sc_eta=0.10)
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    adpt_num_iter = 0

    check_integrate_dynamical(
        system_Chiral_left,
        quantities,
        fout_name="kubo_Chiral",
        Efermi=Efermi,
        omega=omega,
        suffix="sym",
        suffix_ref="",
        use_symmetry=True,
        grid_param=grid_param,
        adpt_num_iter=adpt_num_iter,
        comparer=compare_energyresult,
        additional_parameters=kubo_params,
            )



def test_optical_sym(check_integrate_dynamical, system_Fe_W90, compare_energyresult, compare_sym_asym):
    """Test optical properties: optical conductivity and spin Hall conductivity
    using symmetry (irreducible Kpoints) and 1 adaptive refinement"""
    quantities = ["opt_conductivity", "opt_conductivity^sep", "opt_SHCqiao", "opt_SHCryoo"]

    Efermi = np.array([17.0, 18.0])
    omega = np.arange(0.0, 7.1, 1.0)
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian")
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 1
    specific_parameters = {"opt_conductivity^sep": {"sep_sym_asym": True}}

    check_integrate_dynamical(
        system_Fe_W90,
        quantities,
        fout_name="kubo_Fe_W90_sym",
        Efermi=Efermi,
        omega=omega,
        grid_param=grid_param,
        adpt_num_iter=adpt_num_iter,
        comparer=compare_energyresult,
        additional_parameters=kubo_params,
        use_symmetry=True,
        specific_parameters=specific_parameters)

    compare_sym_asym("kubo_Fe_W90_sym", 1)

    # TODO: Add wcc test


def test_shiftcurrent(check_integrate_dynamical, system_GaAs_W90, compare_energyresult):
    """Test shift current"""
    quantities = ["opt_shiftcurrent"]

    Efermi = np.linspace(7., 9., 5)
    omega = np.arange(1.0, 5.1, 0.5)
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian", sc_eta=0.10)
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 1

    check_integrate_dynamical(
        system_GaAs_W90,
        quantities,
        fout_name="kubo_GaAs_W90",
        Efermi=Efermi,
        omega=omega,
        grid_param=grid_param,
        adpt_num_iter=adpt_num_iter,
        comparer=compare_energyresult,
        additional_parameters=kubo_params,
        extra_precision={"opt_shiftcurrent": 1e-9})

    # TODO: Add wcc test


def test_shc(system_Fe_W90):
    "Test whether SHC from kubo.py and FermiOcean3 are the same"
    quantities = ["opt_SHCqiao", "opt_SHCryoo", "shc_static_qiao", "shc_static_ryoo"]

    Efermi = np.linspace(16.0, 18.0, 21)
    omega = np.array([0.0])
    kubo_params = dict(smr_fixed_width=1e-10, smr_type="Gaussian", kBT=0)
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 0

    system = system_Fe_W90

    grid = wberri.Grid(system, **grid_param)
    additional_parameters = kubo_params
    fout_name = "shc_Fe_W90"

    result = wberri.integrate(
        system,
        grid=grid,
        Efermi=Efermi,
        omega=omega,
        quantities=quantities,
        adpt_num_iter=adpt_num_iter,
        parameters=additional_parameters,
        fout_name=os.path.join(OUTPUT_DIR, fout_name),
        restart=False,
    )

    for mode in ["qiao", "ryoo"]:
        data_fermiocean = result.results[f"shc_static_{mode}"].data
        data_kubo = result.results[f"opt_SHC{mode}"].data[:, 0, ...].real
        precision = max(np.average(abs(data_fermiocean) / 1E10), 1E-8)
        assert data_fermiocean == approx(
            data_kubo, abs=precision), (
                f"data of"
                f"SHC {mode} from FermiOcean and kubo give a maximal absolute"
                f"difference of {np.max(np.abs(data_kubo - data_fermiocean))}.")

    # TODO: Add wcc test


def test_shiftcurrent_symmetry(check_integrate_dynamical, system_GaAs_sym_tb):
    """Test shift current with and without symmetry is the same for a symmetrized system"""
    import copy

    kwargs = dict(
        quantities=["opt_shiftcurrent"],
        Efermi=np.array([7.0]),
        omega=np.arange(1.0, 5.1, 0.5),
        grid_param=dict(NK=6, NKFFT=3),
        additional_parameters=dict(smr_fixed_width=0.20, smr_type="Gaussian", sc_eta=0.1),
        comparer=None,
    )

    system = copy.deepcopy(system_GaAs_sym_tb)
    system.set_symmetry(symmetries_GaAs)

    result_irr_k = check_integrate_dynamical(system, use_symmetry=True, fout_name="kubo_GaAs_sym_irr_k", **kwargs)
    result_full_k = check_integrate_dynamical(system, use_symmetry=False, fout_name="kubo_GaAs_sym_full_k", **kwargs)

    # FIXME: there is small but nonzero difference between the two results.
    # It seems that the finite eta correction term (PRB 103 247101 (2021)) is needed to get perfect agreement.
    assert result_full_k.results["opt_shiftcurrent"].data == approx(
        result_irr_k.results["opt_shiftcurrent"].data, abs=1e-6)


def test_shiftcurrent_run(check_integrate_dynamical, system_GaAs_W90, compare_energyresult):
    "Test shift current implementation in integrate and run give the same results"

    from wannierberri import calculators as calc
    from common_systems import Efermi_GaAs

    smr_type = "Gaussian"
    omega = np.arange(1.0, 5.1, 0.5)
    kubo_params = dict(smr_fixed_width=0.20, smr_type=smr_type, sc_eta=0.10)
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    grid = wberri.Grid(system_GaAs_W90, **grid_param)

    fout_name = "kubo_GaAs_W90_run"

    result_integrate = wberri.integrate(
        system_GaAs_W90,
        grid=grid,
        Efermi=Efermi_GaAs,
        omega=omega,
        quantities=["opt_shiftcurrent"],
        parameters=kubo_params,
        fout_name=os.path.join(OUTPUT_DIR, fout_name),
        restart=False,
    )

    calculators = dict(
        opt_shiftcurrent=calc.dynamic.ShiftCurrent(
            Efermi=Efermi_GaAs,
            omega=omega,
            smr_fixed_width=0.2,
            kBT=0.,
            smr_type=smr_type,
            degen_thresh=0,
            sc_eta=0.10,
        ), )

    result_run = wberri.run(
        system_GaAs_W90,
        grid=grid,
        calculators=calculators,
        fout_name=os.path.join(OUTPUT_DIR, fout_name),
    )

    for key in result_integrate.results.keys():
        data_integrate = result_integrate.results[key].data
        data_run = result_run.results[key].data
        precision = max(np.average(abs(data_integrate) / 1E10), 1E-8)
        assert data_run == approx(
            data_integrate, abs=precision), (
                f"data of"
                f"{key} from integrate and run give a maximal absolute"
                f"difference of {np.max(np.abs(data_run - data_integrate))}.")


def test_kubo_sum_elements():
    from wannierberri.__old_API.__kubo import kubo_sum_elements
    # Compute np.einsum('nmab(c),wnm->wab(c)', x, y).
    # This implementation is much faster than calling np.einsum.
    for nw in 1,5,10:
        for nomega in 1,10,20:
            for i in range(5):
                x=np.random.random( (nw,nw,3,3) )
                y=np.random.random( (nomega,nw,nw) )
                assert kubo_sum_elements(x,y,nw)==approx( np.einsum('nmab,wnm->wab', x, y), abs=1e-8)
                x=np.random.random( (nw,nw,3,3,3) )
                assert kubo_sum_elements(x,y,nw)==approx( np.einsum('nmabc,wnm->wabc', x, y), abs=1e-8)
