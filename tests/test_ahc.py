"""Test the Kubo module."""

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri

from create_system import create_files_Fe_W90, system_Fe_W90
from compare_result import compare_energyresult

@pytest.fixture(scope="module")
def result_ahc_Fe_W90(system_Fe_W90):
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
    Efermi = np.linspace(17,18,11)
    ahc_params = dict()
    adpt_num_iter = 1
    fout_name = "ahc_Fe_W90"

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

    return Efermi, adpt_num_iter, fout_name, result


def test_ahc(result_ahc_Fe_W90, compare_energyresult):
    """Test optical conductivity"""
    Efermi,  adpt_num_iter, fout_name, result = result_ahc_Fe_W90

    data = result.results.get("ahc").data

    assert data.shape == (len(Efermi), 3 )

    compare_energyresult(fout_name, "ahc",       adpt_num_iter,cmplx=False)
    compare_energyresult(fout_name, "ahc_ocean", adpt_num_iter,cmplx=False,suffix_ref="ahc")
#    compare_energyresult(fout_name, "opt_conductivity-asym", adpt_num_iter)
