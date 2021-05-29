"""Test the Kubo module."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri

from create_system import create_files_Fe_W90, system_Fe_W90
from compare_result import compare_energyresult
from test_integrate import compare_quant

@pytest.fixture
def check_integrate_dynamical(output_dir):
    def _inner(system, quantities, fout_name, Efermi, omega, grid_param, comparer,
               numproc=0, additional_parameters={}, adpt_num_iter=0,
               suffix="", suffix_ref="", extra_precision={} ):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.integrate(system,
                grid = grid,
                Efermi = Efermi,
                omega = omega,
                quantities = quantities,
                numproc = numproc,
                adpt_num_iter = adpt_num_iter,
                parameters = additional_parameters,
                fout_name = os.path.join(output_dir, fout_name),
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
        quantities_compare = quantities.copy()
        if "opt_conductivity" in quantities:
            quantities_compare += ["opt_conductivity-sym", "opt_conductivity-asym"]
            quantities_compare.remove("opt_conductivity")

        for quant in quantities_compare:
            prec = extra_precision[quant] if quant in extra_precision else None
            comparer(fout_name, quant+suffix, adpt_num_iter,
                suffix_ref=compare_quant(quant)+suffix_ref, precision=prec)

    return _inner

Efermi_Fe = np.array([17.0, 18.0])
omega_Fe = np.arange(0.0, 7.1, 1.0)

def test_optical(check_integrate_dynamical, system_Fe_W90, compare_energyresult):
    """Test optical properties: optical conductivity and spin Hall conductivity"""
    quantities = ["opt_conductivity", "opt_SHCqiao", "opt_SHCryoo"]
    kubo_params = dict(smr_fixed_width=0.20, smr_type="Gaussian")
    grid = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])
    adpt_num_iter = 1

    check_integrate_dynamical(system_Fe_W90, quantities, fout_name="kubo_Fe_W90",
        Efermi=Efermi_Fe, omega=omega_Fe, grid_param=grid,
        adpt_num_iter=adpt_num_iter, comparer=compare_energyresult,
        additional_parameters=kubo_params)
