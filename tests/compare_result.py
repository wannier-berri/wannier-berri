"""Compare results."""

import os

import numpy as np
import pytest
from pytest import approx

from conftest import REF_DIR, OUTPUT_DIR

def read_energyresult_dat(filename):
    """Read .dat file output of EnergyResult."""
    data_raw = np.loadtxt(filename)
    with open(filename, 'r') as f:
        firstline = f.readline().split()

    # energy titles: before 'x' or 'xx' or 'xxx' or ... occurs.
    E_titles = []
    for title in firstline[1:]:
        if title in ['x' * n for n in range(1, 10)]:
            break
        E_titles.append(title)
    N_energies = len(E_titles)

    data_energy = data_raw[:, :N_energies]

    n_data = (data_raw.shape[1] - N_energies) // 2
    data = data_raw[:, N_energies:N_energies+n_data]
    data_smooth = data_raw[:, N_energies+n_data:]

    return E_titles, data_energy, data, data_smooth

def error_message(fout_name, suffix, i_iter, abs_err, filename, filename_ref,required_precision):
    return (f"data of {fout_name} {suffix} at iteration {i_iter} give a maximal "
            f"absolute difference of {abs_err} greater than the required precision {required_precision}. Files {filename} and {filename_ref}")


@pytest.fixture
def compare_energyresult():
    """Compare dat file output of EnergyResult with the file in reference folder"""
    def _inner(fout_name, suffix, adpt_num_iter,suffix_ref=None,compare_zero=False,precision=None,compare_smooth = True):
        if suffix_ref is None :
            suffix_ref=suffix
        for i_iter in range(adpt_num_iter+1):
            filename     = fout_name + f"-{suffix}_iter-{i_iter:04d}.dat"
            path_filename = os.path.join(OUTPUT_DIR, filename)
            E_titles, data_energy, data, data_smooth = read_energyresult_dat(path_filename)

            if compare_zero:
                precision = 1e-11 if precision is None else abs(precision)
                data_ref    = np.zeros_like(data)
                data_smooth_ref = np.zeros_like(data)
                path_filename_ref = "ZERO"
            else:
                filename_ref = fout_name + f"-{suffix_ref}_iter-{i_iter:04d}.dat"
                path_filename_ref = os.path.join(REF_DIR, filename_ref)
                E_titles_ref, data_energy_ref, data_ref, data_smooth_ref = read_energyresult_dat(path_filename_ref)
                if precision is None:
                    precision = max(abs(np.average(data_smooth_ref) / 1E12), 1E-11)
                elif precision < 0:
                    precision = max(abs(np.average(data_smooth_ref) * abs(precision) ), 1E-11)
                assert E_titles == E_titles_ref
                assert data_energy == approx(data_energy_ref, abs=precision)
            assert data == approx(data_ref, abs=precision), error_message(
                fout_name, suffix, i_iter, np.max(np.abs(data - data_ref)), path_filename, path_filename_ref,precision)
            if compare_smooth:
                assert data_smooth == approx(data_smooth_ref, abs=precision), "smoothed " + error_message(
                    fout_name, suffix, i_iter, np.max(np.abs(data_smooth-data_smooth_ref)), path_filename, path_filename_ref,precision)
    return _inner



def read_frmsf(filename):
    """read a frmsf file"""
    f=open(filename,"r")
    grid = [int(x) for x in f.readline().split()]
    assert f.readline().strip() =='1'  , "the second line of .frmsf file should contain '1'"
    nband = int(f.readline())
    basis = np.array([f.readline().split() for i in range(3)],dtype=float)
    size_block = np.prod(grid)*nband
    data=np.loadtxt(f,dtype=float)
    size_data=len(data)
    assert size_data%size_block == 0 , f"the data contains {size_data} numbers, which is not a multiple of grid {grid} and number of bands {nbands}"
    ndata =  size_data//size_block
    assert ndata in (1,2)  , f"data may contain only energies and one block more at most, found {ndata} blocks"
    return grid,nband,basis,ndata,data[-size_block:]  # check energy nly if the quantity is missing

@pytest.fixture
def compare_fermisurfer():
    """Compare fermisurfer output with the file in reference folder"""
    def _inner(fout_name, suffix, suffix_ref=None,precision=None):
        if suffix_ref is None :
            suffix_ref=suffix

        filename     = fout_name + f"_{suffix}.frmsf"
        filename_ref = fout_name + f"_{suffix_ref}.frmsf"
        path_filename     = os.path.join(OUTPUT_DIR, filename)
        path_filename_ref = os.path.join(REF_DIR, 'frmsf', filename_ref)
        grid     , nband     , basis     , ndata     , data      = read_frmsf(path_filename)
        grid_ref , nband_ref , basis_ref , ndata_ref , data_ref  = read_frmsf(path_filename_ref)

        if precision is None:
            precision = max(abs(np.average(data_ref) / 1E12), 1E-11)
        elif precision < 0:
            precision = max(abs(np.average(data_ref) * abs(precision) ), 1E-11)

        assert grid  == grid_ref  , f"Grid {grid} != {grid_ref}"
        assert nband == nband_ref , f"nband {nband} != {nband_ref}"
        assert ndata == ndata_ref , f"ndata {ndata} != {ndata_ref}"
        assert basis == approx(basis_ref, abs = 1e-8) , f"basis  vectors differ :\n {basis} \n and \n {basis_ref}"

        assert data == approx(data_ref, abs=precision), error_message(
                fout_name, suffix, None, np.max(np.abs(data - data_ref)), path_filename, path_filename_ref,precision)
    return _inner
