"""Compare results."""

import os

import numpy as np
import pytest
from pytest import approx

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

def error_message(fout_name, suffix, i_iter, abs_err, filename, filename_ref):
    return (f"data of {fout_name} {suffix} at iteration {i_iter} give a maximal "
            f"absolute difference of {abs_err}. Files {filename} and {filename_ref}")


@pytest.fixture
def compare_energyresult(output_dir, rootdir):
    """Compare dat file output of EnergyResult with the file in reference folder"""
    def _inner(fout_name, suffix, adpt_num_iter, suffix_ref=None, precision=1E-10):
        if suffix_ref is None:
            suffix_ref = suffix
        for i_iter in range(adpt_num_iter+1):
            filename     = fout_name + f"-{suffix}_iter-{i_iter:04d}.dat"
            filename_ref = fout_name + f"-{suffix_ref}_iter-{i_iter:04d}.dat"
            path_filename = os.path.join(output_dir, filename)
            E_titles, data_energy, data, data_smooth = read_energyresult_dat(path_filename)
            path_filename_ref = os.path.join(rootdir, 'reference', filename_ref)
            E_titles_ref, data_energy_ref, data_ref, data_smooth_ref = read_energyresult_dat(path_filename_ref)

            assert E_titles == E_titles_ref
            # assert data_energy == approx(data_energy_ref, abs=precision)
            assert data == approx(data_ref, abs=precision), error_message(
                fout_name, suffix, i_iter, np.max(np.abs(data - data_ref)), path_filename, path_filename_ref)
            assert data_smooth == approx(data_smooth_ref, abs=precision), "smoothed " + error_message(
                fout_name, suffix, i_iter, np.max(np.abs(data_smooth-data_smooth_ref)), path_filename, path_filename_ref)
    return _inner
