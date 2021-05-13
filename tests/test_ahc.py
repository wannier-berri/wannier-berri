"""Test the anomalous Hall conductivity."""

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri

from create_system import create_files_Fe_W90, system_Fe_W90, system_Fe_W90_wcc
from compare_result import compare_energyresult

@pytest.fixture(scope="module")
def result_ahc_Fe_W90(system_Fe_W90,system_Fe_W90_wcc):
    system = system_Fe_W90
    system_wcc = system_Fe_W90_wcc #using wannier centres in FFT

    num_proc = 0

    # Set symmetry
    SYM = wberri.symmetry
    # generators = [SYM.Inversion, SYM.C4z, SYM.TimeReversal*SYM.C2x]
    generators = []
    system.set_symmetry(generators)

    # Set grid
    grid = wberri.Grid(system, NK=[6, 6, 6], NKFFT=[3, 3, 3])

    # Set parameters
    Efermi = np.linspace(17,18,11)
    ahc_params = dict()
    adpt_num_iter = 1
    fout_name = "ahc_Fe_W90"
    fout_name_wcc = "ahc_Fe_W90_wcc"

    # output folder

    result = wberri.integrate(system,
            grid = grid,
            Efermi = Efermi,
#            omega = omega,
            quantities = ["ahc","ahc_ocean"],
            numproc = num_proc,
            adpt_num_iter = adpt_num_iter,
            parameters = ahc_params,
            fout_name = fout_name,
            restart = False,
    )

    result_wcc = wberri.integrate(system_wcc,
            grid = grid,
            Efermi = Efermi,
#            omega = omega,
            quantities = ["ahc","ahc_ocean"],
            numproc = num_proc,
            adpt_num_iter = adpt_num_iter,
            parameters = ahc_params,
            fout_name = fout_name_wcc,
            restart = False,
    )
    return Efermi, adpt_num_iter, fout_name, fout_name_wcc, result, result_wcc


def test_ahc(result_ahc_Fe_W90, compare_energyresult):
    """Test anomalous Hall conductivity"""
    Efermi,  adpt_num_iter, fout_name, fout_name_wcc ,result , result_wcc = result_ahc_Fe_W90

    data = result.results.get("ahc").data
    data_wcc = result_wcc.results.get("ahc").data

    assert data.shape == (len(Efermi), 3 )
    assert data_wcc.shape == (len(Efermi), 3 )

    compare_energyresult(fout_name, "ahc",       adpt_num_iter)
    compare_energyresult(fout_name, "ahc_ocean", adpt_num_iter,suffix_ref="ahc")
    compare_energyresult(fout_name_wcc, "ahc",       adpt_num_iter)
    compare_energyresult(fout_name_wcc, "ahc_ocean", adpt_num_iter,suffix_ref="ahc")
