"""Test the anomalous Hall conductivity."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from conftest import parallel_serial, parallel_ray 
from conftest import OUTPUT_DIR
from create_system import create_files_Fe_W90,create_files_GaAs_W90,pythtb_Haldane,tbmodels_Haldane
from create_system import system_Fe_W90,system_Fe_W90_wcc,system_GaAs_W90,system_GaAs_W90_wcc,system_GaAs_tb,system_GaAs_tb_wcc
from create_system import system_Haldane_PythTB,system_Haldane_TBmodels,system_Haldane_TBmodels_internal
from create_system import symmetries_Fe
from create_system import system_Chiral,ChiralModel
from compare_result import compare_energyresult


@pytest.fixture
def check_integrate(parallel_serial):
    def _inner(system,quantities,fout_name,Efermi,comparer,
               parallel=None,
               numproc=0,
               grid_param={'NK':[6,6,6],'NKFFT':[3,3,3]},adpt_num_iter=0,
               additional_parameters={}, global_parameters={},
               suffix="", suffix_ref="",
               extra_precision={},
               precision = -1e-8 ,
               compare_smooth = True,
               restart = False):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.integrate(system,
                grid = grid,
                Efermi = Efermi,
                smearEf = 600.0,
    #            omega = omega,
                quantities = quantities,
                parallel=parallel,
                adpt_num_iter = adpt_num_iter,
                parameters = additional_parameters,
                global_parameters = global_parameters,
                fout_name = os.path.join(OUTPUT_DIR, fout_name),
                suffix=suffix,
                restart = restart,
                )
        if len(suffix)>0:
            suffix="-"+suffix
        if len(suffix_ref)>0:
            suffix_ref="-"+suffix_ref

        for quant in quantities:
            data=result.results.get(quant).data
            assert data.shape[0] == len(Efermi)
            assert np.all( np.array(data.shape[1:]) == 3)
            prec=extra_precision[quant] if quant in extra_precision else precision
            comparer(fout_name, quant+suffix,  adpt_num_iter , suffix_ref=compare_quant(quant)+suffix_ref ,precision=prec, compare_smooth = compare_smooth )
    return _inner

@pytest.fixture(scope="session")
def Efermi_Fe():
    return np.linspace(17,18,11)


@pytest.fixture(scope="module")
def Efermi_GaAs():
    return np.linspace(7,9,11)

@pytest.fixture(scope="module")
def Efermi_Haldane():
    return np.linspace(-3,3,11)


@pytest.fixture(scope="session")
def Efermi_Chiral():
    return np.linspace(-5,8,27)

@pytest.fixture(scope="session")
def quantities_Fe():
    return  ['ahc','ahc_test','dos','cumdos',
               'conductivity_ohmic','conductivity_ohmic_fsurf','Morb','Morb_test']

@pytest.fixture(scope="session")
def quantities_Chiral():
    return  [
#         'spin'                     ,#: fermiocean3.spin                   ,
#         'Morb'                     ,#: fermiocean3.Morb                   ,
         'ahc'                      ,#: fermiocean3.AHC                    ,
         'cumdos'                   ,#: fermiocean3.cumdos                 ,
         'dos'                      ,#: fermiocean3.dos                    ,
         'conductivity_ohmic'       ,#: fermiocean3.ohmic                  ,
         'conductivity_ohmic_fsurf' ,#: fermiocean3.ohmic_fsurf            ,
         'berry_dipole'             ,#: fermiocean3.berry_dipole           ,
         'berry_dipole_fsurf'       ,#: fermiocean3.berry_dipole_fsurf     ,
#         'gyrotropic_Korb'          ,#: fermiocean3.gme_orb                ,
#         'gyrotropic_Korb_fsurf'    ,#: fermiocean3.gme_orb_fsurf          ,
#         'gyrotropic_Kspin'         ,#: fermiocean3.gme_spin               ,
#         'gyrotropic_Kspin_fsurf'   ,#: fermiocean3.gme_spin_fsurf         ,
         'Hall_classic'             ,#: fermiocean3.Hall_classic           , 
         'Hall_classic_fsurf'       ,#: fermiocean3.Hall_classic_fsurf     , 
#         'Hall_morb_fsurf'          ,#: fermiocean3.Hall_morb_fsurf        ,
#         'Hall_spin_fsurf'          ,#: fermiocean3.Hall_spin_fsurf        ,
         'Der3E'                    ,#: fermiocean3.Der3E                  ,
#         'Der3E_fsurf'              ,#: fermiocean3.Der3E_fsurf            ,
#         'Der3E_fder2'              ,#: fermiocean3.Der3E_fder2            ,
        ]


@pytest.fixture(scope="module")
def quantities_Haldane():
    return  ['ahc','dos','conductivity_ohmic']

@pytest.fixture(scope="module")
def quantities_GaAs():
    return  ["berry_dipole","berry_dipole_test","berry_dipole_fsurf"]


def compare_quant(quant):
#    compare= {'ahc_ocean':'ahc','ahc3_ocean':'ahc',"cumdos3_ocean":"cumdos","dos3_ocean":"dos","berry_dipole_ocean":"berry_dipole","berry_dipole3_ocean":"berry_dipole",
#            'conductivity_ohmic3_ocean':'conductivity_ohmic','conductivity_ohmic_fsurf3_ocean':'conductivity_ohmic_fsurf'}
    compare = {'ahc_test':'ahc' , 'berry_dipole_test':'berry_dipole', 'Morb_test':'Morb','gyrotropic_Korb_test':'gyrotropic_Korb'}  # it future reverse this - the test is fundamental
    if quant in compare:
        return compare[quant]
    else:
        return quant


def test_Fe(check_integrate,system_Fe_W90, compare_energyresult,quantities_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(system_Fe_W90 , quantities_Fe , fout_name="berry_Fe_W90" , suffix="" , Efermi=Efermi_Fe , comparer=compare_energyresult,compare_smooth = True ,
               global_parameters = {'use_symmetry' : False,'_FF_antisym':True,'_CCab_antisym':True } ,
            extra_precision = {"Morb":-1e-6})


def test_Fe_wcc(check_integrate,system_Fe_W90_wcc, compare_energyresult,quantities_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    # here we test against reference data obtained without wcc_phase. Low accuracy for Morb - this may be a bug
    check_integrate(system_Fe_W90_wcc , quantities_Fe , fout_name="berry_Fe_W90" , suffix="wcc" , Efermi=Efermi_Fe , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False,'_FF_antisym':True,'_CCab_antisym':True } ,
                additional_parameters = { 'correction_wcc':True} ,
            extra_precision = {"Morb":-1})  # the wcc gives quite big error, just checking that it runs
    # here we test agaist reference data obtained with wcc_phase, should matcxh with high accuracy"
#    compare_energyresult( "berry_Fe_W90", "Morb-wcc",  0 , suffix_ref="Morb-wcc" ,precision=-1e-8, compare_smooth = True )

def test_Fe_sym(check_integrate,system_Fe_W90, compare_energyresult,quantities_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(system_Fe_W90 , quantities_Fe , fout_name="berry_Fe_W90" , suffix="sym" , suffix_ref="sym", Efermi=Efermi_Fe , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : True,'_FF_antisym':True,'_CCab_antisym':True }  )


def test_GaAs(check_integrate,system_GaAs_W90, compare_energyresult,quantities_GaAs,Efermi_GaAs):
    """Test berry dipole"""
    check_integrate(system_GaAs_W90 , quantities_GaAs+['gyrotropic_Korb','gyrotropic_Korb_test'] , 
        fout_name="berry_GaAs_W90" , suffix="" , Efermi=Efermi_GaAs , comparer=compare_energyresult ,
               global_parameters = {'degen_thresh':0.005,'use_symmetry' : False ,'_FF_antisym':True,'_CCab_antisym':True},
#                additional_parameters = {"internal_terms":False },
                  extra_precision = {"berry_dipole_fsurf":1e-6} )   # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_GaAs_tb(check_integrate,system_GaAs_tb, compare_energyresult,quantities_GaAs,Efermi_GaAs):
    """Test berry dipole"""
    check_integrate(system_GaAs_tb , quantities_GaAs , fout_name="berry_GaAs_tb" , suffix="" , Efermi=Efermi_GaAs , comparer=compare_energyresult ,
               global_parameters = {'degen_thresh':0.005,'use_symmetry' : False ,'_FF_antisym':True,'_CCab_antisym':True},
                  extra_precision = {"berry_dipole_fsurf":1e-6}  )   # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing

def test_GaAs_wcc(check_integrate,system_GaAs_W90_wcc, compare_energyresult,quantities_GaAs,Efermi_GaAs):
    """Test GaAs with wcc_phase, comparing with data obtained without it"""
    check_integrate(system_GaAs_W90_wcc , quantities_GaAs,#+['gyrotropic_Korb_test'],
         fout_name="berry_GaAs_W90" , suffix="wcc" , Efermi=Efermi_GaAs , comparer=compare_energyresult ,
               global_parameters = {'degen_thresh':0.005,'use_symmetry' : False ,'_FF_antisym':True,'_CCab_antisym':True},
                additional_parameters = { 'correction_Morb_wcc':True} ,
                  extra_precision = {"berry_dipole_fsurf":1e-6} )   # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem

def test_GaAs_tb_wcc(check_integrate,system_GaAs_tb_wcc, compare_energyresult,quantities_GaAs,Efermi_GaAs):
    """Test GaAs (from tb file) with wcc_phase, comparing with data obtained without it"""
    check_integrate(system_GaAs_tb_wcc , quantities_GaAs , fout_name="berry_GaAs_tb" , suffix="wcc" , Efermi=Efermi_GaAs , comparer=compare_energyresult ,
               global_parameters = {'degen_thresh':0.005,'use_symmetry' : False ,'_FF_antisym':True,'_CCab_antisym':True},
                  extra_precision = {"berry_dipole_fsurf":1e-6} )   # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_Haldane_PythTB(check_integrate,system_Haldane_PythTB,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_PythTB , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="pythtb" , Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False} ,
            grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )

def test_Haldane_TBmodels(check_integrate,system_Haldane_TBmodels,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_TBmodels , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="" , Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False} ,
            grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )

def test_Haldane_PythTB_wcc(check_integrate,system_Haldane_PythTB,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_PythTB , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="pythtb_wcc" , Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False} ,
            grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )

def test_Haldane_TBmodels_wcc(check_integrate,system_Haldane_TBmodels,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_TBmodels , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="wcc" , Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False, } ,
               grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )


def test_Haldane_TBmodels_wcc_internal_2(check_integrate,system_Haldane_TBmodels,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_TBmodels_internal , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="wcc_internal_2" , Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False, },
                additional_parameters = { 'external_terms':False} ,
               grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )

def test_Haldane_TBmodels_wcc_internal_2(check_integrate,system_Haldane_TBmodels_internal,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_TBmodels_internal , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="wcc_internal_2" , Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False, },
                additional_parameters = { 'external_terms':False} ,
               grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )

def test_Haldane_TBmodels_wcc_external(check_integrate,system_Haldane_TBmodels,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_TBmodels , ["ahc"] , fout_name="berry_Haldane_tbmodels" , suffix="wcc_external" ,suffix_ref="wcc_external" , Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : False, },
                additional_parameters = { 'internal_terms':False} ,
               grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )


def test_Haldane_PythTB_sym(check_integrate,system_Haldane_PythTB,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_PythTB , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="pythtb_sym" , suffix_ref="",
            Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : True} ,
            grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )

def test_Haldane_TBmodels_sym(check_integrate,system_Haldane_TBmodels,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_TBmodels , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="sym" , suffix_ref="",
            Efermi=Efermi_Haldane , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : True} ,
            grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )

def test_Haldane_TBmodels_sym_refine(check_integrate,system_Haldane_TBmodels,compare_energyresult,quantities_Haldane,Efermi_Haldane):
    check_integrate(system_Haldane_TBmodels , quantities_Haldane , fout_name="berry_Haldane_tbmodels" , suffix="sym" , suffix_ref="sym",
            Efermi=Efermi_Haldane , comparer=compare_energyresult, adpt_num_iter=1,
               global_parameters = {'use_symmetry' : True} ,
            grid_param={'NK':[10,10,1], 'NKFFT':[5,5,1]} )




def test_Fe_sym_refine(check_integrate,system_Fe_W90, compare_energyresult,quantities_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(system_Fe_W90 , quantities_Fe , fout_name="berry_Fe_W90" , 
                  adpt_num_iter=1,
               global_parameters = {'use_symmetry' : True,'_FF_antisym':True,'_CCab_antisym':True } ,
                  suffix="sym" , suffix_ref="sym", Efermi=Efermi_Fe , comparer=compare_energyresult )

def test_Fe_pickle_Klist(check_integrate,system_Fe_W90, compare_energyresult,quantities_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    #  First, remove the 
    try:
        os.remove("Klist.pickle")
    except FileNotFoundError:
        pass
    check_integrate(system_Fe_W90 , quantities_Fe , fout_name="berry_Fe_W90" , 
                  adpt_num_iter=0,
               global_parameters = {'use_symmetry' : True,'_FF_antisym':True,'_CCab_antisym':True } ,
                  suffix="pickle" , suffix_ref="sym", Efermi=Efermi_Fe , comparer=compare_energyresult )
    check_integrate(system_Fe_W90 , quantities_Fe , fout_name="berry_Fe_W90" , 
                  adpt_num_iter=1,
               global_parameters = {'use_symmetry' : True,'_FF_antisym':True,'_CCab_antisym':True } ,
                  suffix="pickle" , suffix_ref="sym", Efermi=Efermi_Fe , comparer=compare_energyresult,restart=True )


def test_Fe_parallel_ray(check_integrate, system_Fe_W90, compare_energyresult,quantities_Fe,Efermi_Fe,
      parallel_ray):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos in parallel with ray"""
    check_integrate(system_Fe_W90 , quantities_Fe , fout_name="berry_Fe_W90" , suffix="paral-ray-4" , suffix_ref="",  Efermi=Efermi_Fe , comparer=compare_energyresult,parallel=parallel_ray,
               global_parameters = {'use_symmetry' : False,'_FF_antisym':True,'_CCab_antisym':True } ,
                    )

def test_Fe_parallel_old(check_integrate, system_Fe_W90, compare_energyresult,quantities_Fe,Efermi_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos in parallel with ray"""
    check_integrate(system_Fe_W90 , quantities_Fe , fout_name="berry_Fe_W90" , suffix="paral-old-4" , suffix_ref="",  Efermi=Efermi_Fe , comparer=compare_energyresult,numproc=4 ,
               global_parameters = {'use_symmetry' : False,'_FF_antisym':True,'_CCab_antisym':True } ,
                    )


def test_Chiral(check_integrate,system_Chiral,compare_energyresult,quantities_Chiral,Efermi_Chiral):
    check_integrate(system_Chiral , quantities_Chiral , fout_name="berry_Chiral" , Efermi=Efermi_Chiral , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : True },
                additional_parameters = { 'external_terms':False} ,
               grid_param={'NK':[10,10,4], 'NKFFT':[5,5,2]} )


def test_Chiral_tetra(check_integrate,system_Chiral,compare_energyresult,quantities_Chiral,Efermi_Chiral):
    check_integrate(system_Chiral , quantities_Chiral , fout_name="berry_Chiral_tetra" , Efermi=Efermi_Chiral , comparer=compare_energyresult,
               global_parameters = {'use_symmetry' : True, },
                additional_parameters = { 'external_terms':False, 'tetra':True} ,
               grid_param={'NK':[10,10,4], 'NKFFT':[5,5,2]} )
