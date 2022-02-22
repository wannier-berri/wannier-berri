"""Test the anomalous Hall conductivity."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from wannierberri import fermiocean
from wannierberri import calculators as calc
from wannierberri.__result import EnergyResult
from conftest import parallel_serial, parallel_ray 
from conftest import OUTPUT_DIR
from create_system import create_files_Fe_W90,create_files_GaAs_W90,pythtb_Haldane,tbmodels_Haldane
from create_system import system_Fe_W90,system_Fe_W90_wcc,system_Fe_FPLO,system_Fe_FPLO_wcc
from create_system import system_GaAs_W90,system_GaAs_W90_wcc,system_GaAs_tb,system_GaAs_tb_wcc,system_GaAs_tb_wcc_ws
from create_system import system_Haldane_PythTB,system_Haldane_TBmodels,system_Haldane_TBmodels_internal
from create_system import symmetries_Fe
from create_system import system_Chiral_left,ChiralModelLeft,system_Chiral_left_TR,ChiralModelLeftTR,system_Chiral_right,ChiralModelRight
from create_system import system_CuMnAs_2d_broken , model_CuMnAs_2d_broken
from compare_result import compare_any_result
from compare_result import compare_fermisurfer
from test_integrate import Efermi_Fe,compare_quant,Efermi_GaAs, Efermi_Chiral
from test_tabulate import get_component_list

@pytest.fixture
def check_run(parallel_serial,compare_any_result):
    def _inner(system,calculators = {},
                fout_name="berry",compare_zero=False,
               parallel=None,
               grid_param={'NK':[6,6,6],'NKFFT':[3,3,3]},adpt_num_iter=0,
               parameters_K={},
               use_symmetry = False,
               suffix="", suffix_ref="",
               extra_precision={},
               precision = -1e-8 ,
               restart = False,file_Klist = None,
               do_not_compare = False,
               skip_compare=[],
                flip_sign = []
               ):

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
                restart = restart,file_Klist = file_Klist,
                )
        
        if do_not_compare:
            return result # compare result externally
                
        if len(suffix)>0:
            suffix="-"+suffix
        if len(suffix_ref)>0:
            suffix_ref="-"+suffix_ref

        for quant in calculators.keys():
            print (quant,skip_compare)
            if quant not in skip_compare:
                prec=extra_precision[quant] if quant in extra_precision else precision
                compare_any_result(fout_name, quant+suffix,  adpt_num_iter , suffix_ref=compare_quant(quant)+suffix_ref ,
                    compare_zero=compare_zero,precision=prec, result_type = resultType(quant) ,flip_sign = (quant in flip_sign) )

    return _inner


@pytest.fixture(scope="session")
def calculators_Fe():
    return  {'ahc':calc.static.AHC,
                'conductivity_ohmic':calc.static.Ohmic,
            }
    #,'ahc_test','dos','cumdos',
    #           'conductivity_ohmic','conductivity_ohmic_fsurf','Morb','Morb_test']



@pytest.fixture(scope="session")
def calculators_GaAs():
    return  {
                'berry_dipole':calc.static.BerryDipole_FermiSea,
                'berry_dipole_fsurf':calc.static.BerryDipole_FermiSurf,
            }


def resultType(quant):
    if quant in []:  # in future - add other options (tabulateresult)
        pass
    else:
        return EnergyResult

def test_Fe(check_run,system_Fe_W90, compare_any_result,calculators_Fe,Efermi_Fe,compare_fermisurfer):
    param  = {'Efermi':Efermi_Fe}
    param_tab = {'degen_thresh':5e-2}
    calculators = {k:v(**param) for k,v in calculators_Fe.items()}
    calculators["tabulate"]=calc.TabulatorAll({
                    "Energy":calc.tabulate.Energy(), # yes, in old implementation degen_thresh was applied to qunatities, 
                                    # but not to energies 
                    "berry" :calc.tabulate.BerryCurvature(**param_tab),
                                              }, 
                                               ibands = [5,6,7,8]
                                              )
    
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])

    parameters_optical = dict(Efermi = np.array([17.0, 18.0]),omega = np.arange(0.0, 7.1, 1.0),
                    smr_fixed_width=0.20, smr_type="Gaussian")
    calculators['opt_conductivity'] = wberri.calculators.dynamic.OpticalConductivity(**parameters_optical)
    calculators['opt_SHCqiao']      = wberri.calculators.dynamic.SHC(SHC_type="qiao",**parameters_optical)
    calculators['opt_SHCryoo']      = wberri.calculators.dynamic.SHC(SHC_type="ryoo",**parameters_optical)
    
    check_run(system_Fe_W90 , calculators , fout_name="berry_Fe_W90" , suffix="run" ,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
            extra_precision = {"Morb":-1e-6},
            skip_compare = ['tabulate','opt_conductivity','opt_SHCqiao','opt_SHCryoo'])

    for quant in 'opt_conductivity','opt_SHCryoo','opt_SHCryoo':
        compare_any_result("berry_Fe_W90", quant+"-run",  0 , 
            fout_name_ref = "kubo_Fe_W90",suffix_ref=quant ,
            precision=1e-8, result_type = EnergyResult )

    
    extra_precision = {'berry':1e-6}
    for quant in ["E","berry"]:
#        quant_ref = 'E' if quant == "Energy" else quant
        for comp in get_component_list(quant):
            quant_ref = quant
            _comp = "-" +comp if comp is not None else ""
#            data=result.results.get(quant).data
#            assert data.shape[0] == len(Efermi)
#            assert np.all( np.array(data.shape[1:]) == 3)
            prec=extra_precision[quant] if quant in extra_precision else 1e-8
#            comparer(frmsf_name, quant+_comp+suffix,  suffix_ref=compare_quant(quant)+_comp+suffix_ref ,precision=prec )
            compare_fermisurfer(fout_name="berry_Fe_W90-tabulate", 
                 suffix = quant+_comp+"-run",
                 suffix_ref = quant_ref+_comp,
                 fout_name_ref="tabulate_Fe_W90",precision=prec)




def test_Fe_parallel_ray(check_run, system_Fe_W90, compare_any_result,calculators_Fe,Efermi_Fe,
      parallel_ray):
    param  = {'Efermi':Efermi_Fe}
    calculators = {k:v(**param) for k,v in calculators_Fe.items()}
    check_run(system_Fe_W90 , calculators , fout_name="berry_Fe_W90" , suffix="paral-ray-4-run" ,
               parallel=parallel_ray,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
            )
    parallel_ray.shutdown()


def test_Fe_sym(check_run,system_Fe_W90, compare_any_result,calculators_Fe,Efermi_Fe):
    param  = {'Efermi':Efermi_Fe}
    calculators = {k:v(**param) for k,v in calculators_Fe.items()}
    check_run(system_Fe_W90 , calculators , fout_name="berry_Fe_W90" , suffix="sym-run" , suffix_ref= "sym",
               use_symmetry = True,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
            )

def test_Fe_sym_refine(check_run,system_Fe_W90, compare_any_result,calculators_Fe,Efermi_Fe):
    param  = {'Efermi':Efermi_Fe}
    calculators = {k:v(**param) for k,v in calculators_Fe.items()}
    check_run(system_Fe_W90 , calculators , fout_name="berry_Fe_W90" , suffix="sym-run" , suffix_ref= "sym",
                  adpt_num_iter=1,use_symmetry = True,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
            )

def test_Fe_pickle_Klist(check_run,system_Fe_W90, compare_any_result,calculators_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    #  First, remove the 
    try:
        os.remove("Klist.pickle")
    except FileNotFoundError:
        pass
    param  = {'Efermi':Efermi_Fe}
    calculators = {k:v(**param) for k,v in calculators_Fe.items()}
    check_run(system_Fe_W90 , calculators , fout_name="berry_Fe_W90" , suffix="pickle-run" , suffix_ref= "sym",
                  adpt_num_iter=0,use_symmetry = True,file_Klist = "Klist.pickle", 
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
            )
    check_run(system_Fe_W90 , calculators , fout_name="berry_Fe_W90" , suffix="pickle-run" , suffix_ref= "sym",
                  adpt_num_iter=1,use_symmetry = True,file_Klist = "Klist.pickle" ,restart = True,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
             )


def test_GaAs(check_run,system_GaAs_W90, compare_any_result,calculators_GaAs,Efermi_GaAs,compare_fermisurfer):
    param  = {'Efermi':Efermi_GaAs}
    param_tab = {'degen_thresh':5e-2}
    calculators = {k:v(**param) for k,v in calculators_GaAs.items()}
    
    grid_param = dict(NK=[6, 6, 6], NKFFT=[3, 3, 3])

    check_run(system_GaAs_W90 , calculators , fout_name="berry_GaAs_W90" , suffix="run" ,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } ,
                  extra_precision = {"berry_dipole_fsurf":1e-6} )   # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing




@pytest.fixture(scope="session")
def calculators_Chiral(Efermi_Chiral):
    calculators  = {}
    calculators['conductivity_ohmic'] = calc.static.Ohmic(Efermi=Efermi_Chiral)
    calculators['berry_dipole']       = calc.static.BerryDipole_FermiSea(Efermi=Efermi_Chiral,kwargs_formula={"external_terms":False} )
    calculators['ahc']       = calc.static.AHC(Efermi=Efermi_Chiral,kwargs_formula={"external_terms":False} )
    return calculators

def test_Chiral_left(check_run,system_Chiral_left, compare_any_result,Efermi_Chiral,compare_fermisurfer,calculators_Chiral):

    grid_param={'NK':[10,10,4], 'NKFFT':[5,5,2]} 
    check_run(system_Chiral_left , calculators_Chiral , fout_name="berry_Chiral" , suffix="left-run" ,
                grid_param = grid_param,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } , use_symmetry = True,
            extra_precision = {"Morb":-1e-6},)


def test_Chiral_leftTR(check_run,system_Chiral_left_TR, compare_any_result,Efermi_Chiral,compare_fermisurfer,calculators_Chiral):
    "check that for time-revrsed model the ohmic conductivity is the same, but the AHC is opposite"
    grid_param={'NK':[10,10,4], 'NKFFT':[5,5,2]} 
    check_run(system_Chiral_left_TR ,calculators_Chiral , fout_name="berry_Chiral" , suffix="left-run" ,
                grid_param = grid_param,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } , use_symmetry = True,
            extra_precision = {"Morb":-1e-6},
            flip_sign = ['ahc']
            )


def test_Chiral_right(check_run,system_Chiral_right, compare_any_result,Efermi_Chiral,compare_fermisurfer,calculators_Chiral):
    "check that for flipped chirality the ohmic conductivity is the same, but hte Berry dipole is opposite"
    grid_param={'NK':[10,10,4], 'NKFFT':[5,5,2]} 
    check_run(system_Chiral_right , calculators_Chiral , fout_name="berry_Chiral" , suffix="right-run" ,
                grid_param = grid_param,
               parameters_K = {'_FF_antisym':True,'_CCab_antisym':True } , use_symmetry = True,
            extra_precision = {"Morb":-1e-6},
            flip_sign = ['berry_dipole'],
        )


    
