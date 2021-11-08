"""Test the Kubo module."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri

from conftest import OUTPUT_DIR
from create_system import create_files_Fe_W90, system_Fe_W90, system_Fe_W90_wcc
from create_system import create_files_GaAs_W90, system_GaAs_W90, system_GaAs_W90_wcc
from create_system import symmetries_Fe
from compare_result import compare_energyresult
from test_integrate import compare_quant

@pytest.fixture
def check_integrate_dynamical():
    """
    This function is similar to check_integrate, but the difference is 1) the shape of the
    data are different for dynamical quantities (has omega index), and 2) opt_conductivity
    requires a special treatment because of sym and asym data.
    """
    def _inner(system, quantities, fout_name, Efermi, omega, grid_param, comparer,
               additional_parameters={}, 
               parameters_K={},
               adpt_num_iter=0,use_symmetry = False,
               suffix="", suffix_ref="", extra_precision={} ):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.integrate(system,
                grid = grid,
                Efermi = Efermi,
                omega = omega,
                quantities = quantities,
                use_irred_kpt = use_symmetry, symmetrize = use_symmetry,
                adpt_num_iter = adpt_num_iter,
                parameters = additional_parameters,
                parameters_K = parameters_K,
                fout_name = os.path.join(OUTPUT_DIR, fout_name),
                suffix = suffix,
                restart = False,
                )
        if len(suffix) > 0:
            suffix = "-" + suffix
        if len(suffix_ref) > 0:
            suffix_ref = "-" + suffix_ref

        # Test results output
        for quant in quantities:
            if quant == "opt_conductivity":
                data_list = [result.results.get(quant).results.get(s).data for s in ["sym", "asym"]]
            else:
                data_list = [result.results.get(quant).data]

            for data in data_list:
                assert data.shape[0] == len(Efermi)
                assert data.shape[1] == len(omega)
                assert all(i == 3 for i in data.shape[2:])

        # Test file output
        if comparer:
            quantities_compare = quantities.copy()
            if "opt_conductivity" in quantities:
                quantities_compare += ["opt_conductivity-sym", "opt_conductivity-asym"]
                quantities_compare.remove("opt_conductivity")

            for quant in quantities_compare:
                prec = extra_precision[quant] if quant in extra_precision else None
                comparer(fout_name, quant+suffix, adpt_num_iter,
                    suffix_ref=compare_quant(quant)+suffix_ref, precision=prec)

    return _inner


def test_optical(check_integrate_dynamical, system_Fe_W90, compare_energyresult):
    """Test optical properties: optical conductivity and spin Hall conductivity"""
    quantities = ["opt_conductivity", "opt_SHCqiao", "opt_SHCryoo"]

    Efermi = np.array([17.0, 18.0])
    omega = np.arange(0.0, 7.1, 1.0)
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian")
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 1

    check_integrate_dynamical(system_Fe_W90, quantities, fout_name="kubo_Fe_W90",
        Efermi=Efermi, omega=omega, grid_param=grid_param,
        adpt_num_iter=adpt_num_iter, comparer=compare_energyresult,
        additional_parameters=kubo_params)

    # TODO: Add wcc test

def test_shiftcurrent(check_integrate_dynamical, system_GaAs_W90, compare_energyresult):
    """Test shift current"""
    quantities = ["opt_shiftcurrent"]

    Efermi = np.linspace(7., 9., 5)
    omega = np.arange(1.0, 5.1, 0.5)
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian", sc_eta=0.10)
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 1

    check_integrate_dynamical(system_GaAs_W90, quantities, fout_name="kubo_GaAs_W90",
        Efermi=Efermi, omega=omega, grid_param=grid_param,
        adpt_num_iter=adpt_num_iter, comparer=compare_energyresult,
        additional_parameters=kubo_params,
        extra_precision = {"opt_shiftcurrent":1e-9})

    # TODO: Add wcc test

def test_shc(system_Fe_W90):
    "Test whether SHC from kubo.py and FermiOcean3 are the same"
    quantities = ["opt_SHCqiao", "opt_SHCryoo", "shc_qiao", "shc_ryoo"]

    Efermi = np.linspace(16.0, 18.0, 21)
    omega = np.array([0.0])
    kubo_params = dict(smr_fixed_width=1e-10, smr_type="Gaussian", kBT=0)
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 0

    system = system_Fe_W90

    grid = wberri.Grid(system, **grid_param)
    additional_parameters = kubo_params
    fout_name = "shc_Fe_W90"

    result = wberri.integrate(system,
            grid = grid,
            Efermi = Efermi,
            omega = omega,
            quantities = quantities,
            adpt_num_iter = adpt_num_iter,
            parameters = additional_parameters,
            fout_name = os.path.join(OUTPUT_DIR, fout_name),
            restart = False,
    )

    for mode in ["qiao", "ryoo"]:
        data_fermiocean = result.results[f"shc_{mode}"].data
        data_kubo = result.results[f"opt_SHC{mode}"].data[:, 0, ...].real
        precision = max(np.average(abs(data_fermiocean) / 1E10), 1E-8)
        assert data_fermiocean == approx(data_kubo, abs=precision), (f"data of"
            f"SHC {mode} from FermiOcean and kubo give a maximal absolute"
            f"difference of {np.max(np.abs(data_kubo - data_fermiocean))}.")

    # TODO: Add wcc test