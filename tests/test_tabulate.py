"""Test the anomalous Hall conductivity."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from conftest import parallel_serial, parallel_ray 
from conftest import OUTPUT_DIR
from create_system import create_files_Fe_W90 #,create_files_GaAs_W90,pythtb_Haldane,tbmodels_Haldane
from create_system import system_Fe_W90 #,system_GaAs_W90,system_GaAs_tb
#from create_system import system_Haldane_PythTB,system_Haldane_TBmodels,system_Haldane_TBmodels_internal
from create_system import symmetries_Fe
from compare_result import compare_fermisurfer
from create_system import system_Chiral,ChiralModel


@pytest.fixture
def get_component_list():
    def _inner(quantity):
        if quantity in ["E"]:
            return [""]
        if quantity in ["berry","V"]:
            return ["-"+a for a in "z"]
        if quantity in ["Der_berry"]:
            return  ["-"+a+b for a in "xyz" for b in "xyz"]
        raise ValueError(f"unknown quantity {quantity}")
    return _inner


@pytest.fixture
def check_tabulate(parallel_serial,get_component_list):
    def _inner(system,quantities,frmsf_name,comparer,
               parallel=None,
               numproc=0,
               grid_param={'NK':[6,6,6],'NKFFT':[3,3,3]},
               additional_parameters={}, global_parameters={},
               suffix="", suffix_ref="",
               extra_precision={},ibands = None):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.tabulate(system,
                grid = grid,
                quantities = quantities,
                parallel=parallel,
                parameters = additional_parameters,
                ibands = ibands,
                global_parameters = global_parameters,
                frmsf_name = os.path.join(OUTPUT_DIR, frmsf_name),
                suffix=suffix
                )

        if len(suffix)>0:
            suffix="-"+suffix
        if len(suffix_ref)>0:
            suffix_ref="-"+suffix_ref

        for quant in ["E"]+quantities:
          for comp in get_component_list(quant):
#            data=result.results.get(quant).data
#            assert data.shape[0] == len(Efermi)
#            assert np.all( np.array(data.shape[1:]) == 3)
            prec=extra_precision[quant] if quant in extra_precision else None
            comparer(frmsf_name, quant+comp+suffix,  suffix_ref=compare_quant(quant)+comp+suffix_ref ,precision=prec )
    return _inner


@pytest.fixture(scope="session")
def quantities_tab():
    return  ['V','berry','Der_berry']


def compare_quant(quant):
#    compare= {'ahc_ocean':'ahc','ahc3_ocean':'ahc',"cumdos3_ocean":"cumdos","dos3_ocean":"dos","berry_dipole_ocean":"berry_dipole","berry_dipole3_ocean":"berry_dipole",
#            'conductivity_ohmic3_ocean':'conductivity_ohmic','conductivity_ohmic_fsurf3_ocean':'conductivity_ohmic_fsurf'}
    compare = {}
    if quant in compare:
        return compare[quant]
    else:
        return quant


def test_Fe(check_tabulate,system_Fe_W90, compare_fermisurfer,quantities_tab):
    """Test Energies, Velocities, berry curvature, its derivative"""
    check_tabulate(system_Fe_W90 , quantities_tab , frmsf_name="tabulate_Fe_W90" , suffix="" ,  comparer=compare_fermisurfer,
               global_parameters = {'use_symmetry' : False}, ibands = [5,6,7,8] )

def test_Chiral(check_tabulate,system_Chiral, compare_fermisurfer,quantities_tab):
    """Test Energies, Velocities, berry curvature, its derivative"""
    check_tabulate(system_Chiral , quantities_tab , frmsf_name="tabulate_Chiral" , suffix="" ,  comparer=compare_fermisurfer,
               global_parameters = {'use_symmetry' : False ,"use_wcc_phase" : True } , additional_parameters = {'external_terms':False}, ibands = [0,1] )


def test_Chiral_sym(check_tabulate,system_Chiral, compare_fermisurfer,quantities_tab):
    """Test Energies, Velocities, berry curvature, its derivative"""
    check_tabulate(system_Chiral , quantities_tab , frmsf_name="tabulate_Chiral" , suffix="sym" ,  comparer=compare_fermisurfer,
               global_parameters = {'use_symmetry' : True , "use_wcc_phase" : True } , additional_parameters = {'external_terms':False}, ibands = [0,1] )

