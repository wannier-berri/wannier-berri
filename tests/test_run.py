"""Test the anomalous Hall conductivity."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from wannierberri import fermiocean
from wannierberri.__result import EnergyResult
from conftest import parallel_serial, parallel_ray 
from conftest import OUTPUT_DIR
from create_system import create_files_Fe_W90,create_files_GaAs_W90,pythtb_Haldane,tbmodels_Haldane
from create_system import system_Fe_W90,system_Fe_W90_wcc,system_Fe_FPLO,system_Fe_FPLO_wcc
from create_system import system_GaAs_W90,system_GaAs_W90_wcc,system_GaAs_tb,system_GaAs_tb_wcc,system_GaAs_tb_wcc_ws
from create_system import system_Haldane_PythTB,system_Haldane_TBmodels,system_Haldane_TBmodels_internal
from create_system import symmetries_Fe
from create_system import system_Chiral,ChiralModel
from create_system import system_CuMnAs_2d_broken , model_CuMnAs_2d_broken
from compare_result import compare_any_result
from test_integrate import Efermi_Fe,compare_quant

@pytest.fixture
def check_run_integrate(parallel_serial,compare_any_result):
    def _inner(system,calculators = {},
                fout_name="berry",compare_zero=False,
               parallel=None,
               grid_param={'NK':[6,6,6],'NKFFT':[3,3,3]},adpt_num_iter=0,
               parameters_K={},
               use_symmetry = False,
               suffix="", suffix_ref="",
               extra_precision={},
               precision = -1e-8 ,
               restart = False,
               do_not_compare = False):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.run(system,
                grid = grid,
                calculators = calculators,
                parallel=parallel,
                adpt_num_iter = adpt_num_iter,
                use_irred_kpt = use_symmetry, symmetrize = use_symmetry,
                parameters_K = parameters_K,
                fout_name = os.path.join(OUTPUT_DIR, fout_name),
                suffix=suffix,
                restart = restart,
                )
        
        if do_not_compare:
            return result # compare result externally
                
        if len(suffix)>0:
            suffix="-"+suffix
        if len(suffix_ref)>0:
            suffix_ref="-"+suffix_ref

        for quant in calculators.keys():
            prec=extra_precision[quant] if quant in extra_precision else precision
            compare_any_result(fout_name, quant+suffix,  adpt_num_iter , suffix_ref=compare_quant(quant)+suffix_ref ,
                compare_zero=compare_zero,precision=prec, result_type = resultType(quant) )

    return _inner


@pytest.fixture(scope="session")
def calculators_Fe():
    return  {'ahc':wberri.calculators.AHC}
    #,'ahc_test','dos','cumdos',
    #           'conductivity_ohmic','conductivity_ohmic_fsurf','Morb','Morb_test']


def resultType(quant):
    if quant in ['ahc',]:
        return EnergyResult
    else:
        raise ValueError(f"Unknown which result type to expect for {quant}")

def test_Fe(check_run_integrate,system_Fe_W90, compare_any_result,calculators_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    param  = {'Efermi':Efermi_Fe}
    calculators = {k:v(**param) for k,v in calculators_Fe.items()}
    check_run_integrate(system_Fe_W90 , calculators , fout_name="berry_Fe_W90" , suffix="cal" ,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
            extra_precision = {"Morb":-1e-6})


