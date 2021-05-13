"""Test the Kubo module."""

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri

from create_system import create_files_Fe_W90, system_Fe_W90
from compare_result import compare_energyresult

@pytest.fixture(scope="module")
def result_kubo_Fe_W90(system_Fe_W90):
    system = system_Fe_W90

    num_proc = 0

    # Set symmetry
    SYM = wberri.symmetry
    # generators = [SYM.Inversion, SYM.C4z, SYM.TimeReversal*SYM.C2x]
    generators = []
    system.set_symmetry(generators)

    # Set grid
    grid = wberri.Grid(system, NK=[6, 6, 6], NKFFT=[3, 3, 3])

    # Set parameters
    Efermi = np.array([17.0, 18.0])
    omega = np.arange(0.0, 7.1, 1.0)
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian")
    adpt_num_iter = 1
    fout_name = "kubo_Fe_W90"

    # output folder

    result = wberri.integrate(system,
            grid = grid,
            Efermi = Efermi,
            omega = omega,
            quantities = ["opt_conductivity", "opt_SHCqiao", "opt_SHCryoo"],
            numproc = num_proc,
            adpt_num_iter = adpt_num_iter,
            parameters = kubo_params,
            fout_name = fout_name,
            restart = False,
    )

    return Efermi, omega, adpt_num_iter, fout_name, result


def test_opt_conductivity(result_kubo_Fe_W90, compare_energyresult):
    """Test optical conductivity"""
    Efermi, omega, adpt_num_iter, fout_name, result = result_kubo_Fe_W90

    data_sym = result.results.get("opt_conductivity").results["sym"].data
    data_asym = result.results.get("opt_conductivity").results["asym"].data

    assert data_sym.shape == (len(Efermi), len(omega), 3, 3)
    assert data_asym.shape == (len(Efermi), len(omega), 3, 3)

    assert data_sym == approx(np.swapaxes(data_sym, 2, 3), abs=1E-6)
    assert data_asym == approx(-np.swapaxes(data_asym, 2, 3), abs=1E-6)

    compare_energyresult(fout_name, "opt_conductivity-sym", adpt_num_iter,cmplx=True)
    compare_energyresult(fout_name, "opt_conductivity-asym", adpt_num_iter,cmplx=True)


def test_SHCqiao(result_kubo_Fe_W90, compare_energyresult):
    """Test spin Hall conductivity using Qiao's method"""
    Efermi, omega, adpt_num_iter, fout_name, result = result_kubo_Fe_W90

    data = result.results.get("opt_SHCqiao").data

    assert data.shape == (len(Efermi), len(omega), 3, 3, 3)

    compare_energyresult(fout_name, "opt_SHCqiao", adpt_num_iter,cmplx=True)


def test_SHCryoo(result_kubo_Fe_W90, compare_energyresult):
    """Test spin Hall conductivity using Ryoo's method"""
    Efermi, omega, adpt_num_iter, fout_name, result = result_kubo_Fe_W90

    data = result.results.get("opt_SHCryoo").data

    assert data.shape == (len(Efermi), len(omega), 3, 3, 3)

    compare_energyresult(fout_name, "opt_SHCryoo", adpt_num_iter,cmplx=True)
