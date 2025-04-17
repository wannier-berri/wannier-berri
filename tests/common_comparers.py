"""Compare results."""

import os

import numpy as np
import pytest
from pytest import approx
from wannierberri.result import EnergyResult

from .common import REF_DIR, ROOT_DIR, REF_DIR_INTEGRATE, OUTPUT_DIR_RUN


def compare_quant(quantity):
    """Return quantitiy name to be compared with the input quantity"""
    # it future reverse this - the test is fundamental
    compare = {
        'ahc_test': 'ahc',
        'berry_dipole_test': 'berry_dipole',
        'Morb_test': 'Morb',
        'gyrotropic_Korb_test': 'gyrotropic_Korb',
        'Energy': 'E',
        'GME_orb_FermiSea_test': 'GME_orb_FermiSea',
        'BerryDipole_FermiSea_test': 'BerryDipole_FermiSea',
    }
    if quantity in compare:
        return compare[quantity]
    else:
        return quantity


def read_energyresult_dat(filename, mode="txt"):
    """Read .dat of .npz file output of EnergyResult."""
    if mode == "bin":
        res = EnergyResult(file_npz=filename)
        #        energ = [res[f'Energies_{i}'] for i,_ in enumerate(res['E_titles'])]  # in binary mode energies are just two arrays
        # while in txt mode it is a direct product
        #        return res['E_titles']), energ , res['data'], None # we do not check smoothing in the binary mode
        return res.E_titles, res.Energies, res.data, None  # we do not check smoothing in the binary mode
    elif mode == "txt":
        # Now the txt mode
        # get the first line that does not start with "####" neither empty
        for l in open(filename, 'r'):
            l = l.strip()
            if l.startswith("####") or len(l) == 0:
                continue
            else:
                firstline = l.split()
                break
        data_raw = np.loadtxt(filename)

        # energy titles: before 'x' or 'xx' or 'xxx' or ... occurs.
        E_titles = []
        for title in firstline[1:]:
            if title in ['x' * n for n in range(1, 10)]:
                break
            E_titles.append(title)
        N_energies = len(E_titles)
        data_energy = data_raw[:, :N_energies]

        n_data = (data_raw.shape[1] - N_energies) // 2
        data = data_raw[:, N_energies:N_energies + n_data]
        data_smooth = data_raw[:, N_energies + n_data:]
        return E_titles, data_energy, data, data_smooth
    else:
        raise ValueError(f"Supported modes are `txt` and `bin`, found {mode}")


def error_message(fout_name, suffix, i_iter, abs_err, filename, filename_ref, required_precision):
    return (
        f"data of {fout_name} {suffix} at iteration {i_iter} give a maximal "
        f"absolute difference of {abs_err} greater than the required precision {required_precision}. Files {filename} and {filename_ref}"
    )


@pytest.fixture
def compare_energyresult():
    """Compare dat file output of EnergyResult with the file in reference folder"""

    def _inner(
            fout_name,
            suffix,
            adpt_num_iter,
            suffix_ref=None,
            compare_zero=False,
            precision=None,
            compare_smooth=True,
            mode="txt",
            data_reference=None):
        assert mode in ["txt", "bin", "inner"]
        if mode == "bin":
            compare_smooth = False
            ext = ".npz"
        elif mode == "txt":
            ext = ".dat"
        if suffix_ref is None:
            suffix_ref = suffix
        for i_iter in range(adpt_num_iter + 1):
            filename = fout_name + f"-{suffix}_iter-{i_iter:04d}" + ext
            path_filename = os.path.join(OUTPUT_DIR_RUN, filename)
            E_titles, data_energy, data, data_smooth = read_energyresult_dat(path_filename, mode=mode)

            if compare_zero:
                precision = 1e-11 if precision is None else abs(precision)
                data_ref = np.zeros_like(data)
                data_smooth_ref = np.zeros_like(data)
                path_filename_ref = "ZERO"
            else:
                if data_reference is None:
                    filename_ref = fout_name + f"-{suffix_ref}_iter-{i_iter:04d}" + ext
                    path_filename_ref = os.path.join(REF_DIR_INTEGRATE, filename_ref)
                    E_titles_ref, data_energy_ref, data_ref, data_smooth_ref = read_energyresult_dat(
                        path_filename_ref, mode=mode)
                else:
                    E_titles_ref, data_energy_ref, data_ref, data_smooth_ref = data_reference[i_iter]

                # just to determine precision automatically
                if compare_smooth:
                    maxval = np.max(abs(data_smooth_ref))
                else:
                    maxval = np.max(abs(data_ref))
                if precision is None:
                    precision = max(maxval / 1E12, 1E-11)
                elif precision < 0:
                    precision = max(maxval * abs(precision), 1E-11)
                print(f"E_titles : <{E_titles}> vs <{E_titles_ref}>")
                assert np.all(E_titles == E_titles_ref), f"E_titles mismatch : <{E_titles}> != <{E_titles_ref}>"
                if isinstance(data_energy, list):
                    for i, E in enumerate(zip(data_energy, data_energy_ref)):
                        assert E[0] == approx(
                            E[1]), f"energy array {i} with title {E_titles[i]} differ by {np.max(abs(E[0] - E[1]))}"
                else:
                    assert data_energy == approx(data_energy_ref, abs=precision)
            assert data == approx(
                data_ref, abs=precision), error_message(
                    fout_name, suffix, i_iter, np.max(np.abs(data - data_ref)), path_filename, path_filename_ref,
                    precision)
            if compare_smooth:
                assert data_smooth == approx(
                    data_smooth_ref, abs=precision), "smoothed " + error_message(
                        fout_name, suffix, i_iter, np.max(np.abs(data_smooth - data_smooth_ref)), path_filename,
                        path_filename_ref, precision)

    return _inner



@pytest.fixture
def compare_any_result():
    """Compare dat file output of EnergyResult with the file in reference folder"""

    def _inner(
            fout_name,
            suffix,
            adpt_num_iter,
            fout_name_ref=None,
            suffix_ref=None,
            ref_dir=None,
            compare_zero=False,
            precision=None,
            result_type=None):
        if suffix_ref is None:
            suffix_ref = suffix
        if fout_name_ref is None:
            fout_name_ref = fout_name
        if ref_dir is None:
            path_ref = REF_DIR_INTEGRATE
        else:
            path_ref = os.path.join(ROOT_DIR, ref_dir)
        ext = ".npz"
        for i_iter in range(adpt_num_iter + 1):
            filename = fout_name + f"-{suffix}_iter-{i_iter:04d}" + ext
            path_filename = os.path.join(OUTPUT_DIR_RUN, filename)
            result = result_type(file_npz=path_filename)

            if compare_zero:
                result_ref = result * 0.
                path_filename_ref = "ZERO"
                assert precision > 0, "comparing with zero is possible only with absolute precision"
            else:
                filename_ref = fout_name_ref + f"-{suffix_ref}_iter-{i_iter:04d}" + ext
                path_filename_ref = os.path.join(path_ref, filename_ref)
                result_ref = result_type(file_npz=path_filename_ref)
                maxval = result_ref._maxval_raw
                if precision is None:
                    precision = max(maxval / 1E12, 1E-11)
                elif precision < 0:
                    precision = max(maxval * abs(precision), 1E-11)
            err = (result - result_ref)._maxval_raw
            assert err < precision, error_message(
                fout_name, suffix, i_iter, err, path_filename, path_filename_ref, precision)

    return _inner


def read_frmsf(filename):
    """read a frmsf file"""
    f = open(filename, "r")
    grid = [int(x) for x in f.readline().split()]
    assert f.readline().strip() == '1', "the second line of .frmsf file should contain '1'"
    nband = int(f.readline())
    basis = np.array([f.readline().split() for i in range(3)], dtype=float)
    size_block = np.prod(grid) * nband
    data = np.loadtxt(f, dtype=float)
    size_data = len(data)
    assert size_data % size_block == 0, f"the data contains {size_data} numbers, which is not a multiple of grid {grid} and number of bands {nband}"
    ndata = size_data // size_block
    assert ndata in (1, 2), f"data may contain only energies and one block more at most, found {ndata} blocks"
    return grid, nband, basis, ndata, data[-size_block:]  # check energy only if the quantity is missing


@pytest.fixture
def compare_fermisurfer():
    """Compare fermisurfer output with the file in reference folder"""

    def _inner(fout_name, suffix="", fout_name_ref=None, suffix_ref=None, precision=None):
        if suffix_ref is None:
            suffix_ref = suffix
        if fout_name_ref is None:
            fout_name_ref = fout_name

        filename = fout_name + f"_{suffix}.frmsf"
        filename_ref = fout_name_ref + f"_{suffix_ref}.frmsf"
        path_filename = os.path.join(OUTPUT_DIR_RUN, filename)
        path_filename_ref = os.path.join(REF_DIR, 'frmsf', filename_ref)
        grid, nband, basis, ndata, data = read_frmsf(path_filename)
        grid_ref, nband_ref, basis_ref, ndata_ref, data_ref = read_frmsf(path_filename_ref)

        if precision is None:
            precision = max(abs(np.average(data_ref) / 1E12), 1E-11)
        elif precision < 0:
            precision = max(abs(np.average(data_ref) * abs(precision)), 1E-11)

        assert grid == grid_ref, f"Grid {grid} != {grid_ref}"
        assert nband == nband_ref, f"nband {nband} != {nband_ref}"
        assert ndata == ndata_ref, f"ndata {ndata} != {ndata_ref}"
        assert basis == approx(basis_ref, abs=1e-8), f"basis  vectors differ :\n {basis} \n and \n {basis_ref}"

        data_srt = np.sort(data.flatten())
        data_ref_srt = np.sort(data_ref.flatten())
        assert data_srt == approx(data_ref_srt, abs=precision)
        assert data == approx(data_ref, abs=precision), error_message(
            fout_name, suffix, None, np.max(np.abs(data - data_ref)), path_filename, path_filename_ref, precision)

    return _inner


@pytest.fixture
def compare_sym_asym():
    " to comapre the results separated by symmetric-antisymmetric part"

    def _inner(fout_name, adpt_num_iter=0, quantity="opt_conductivity"):
        mode = "bin"
        name = fout_name + "-" + quantity
        for i_iter in range(adpt_num_iter + 1):
            filename_ref = name + "^sep-sym" + f"_iter-{i_iter:04d}.npz"
            path_filename_ref = os.path.join(REF_DIR_INTEGRATE, filename_ref)
            E_titles_ref, data_energy_ref, data_ref_sym, data_smooth_ref = read_energyresult_dat(
                path_filename_ref, mode=mode)
            filename_ref = name + "^sep-asym" + f"_iter-{i_iter:04d}.npz"
            path_filename_ref = os.path.join(REF_DIR_INTEGRATE, filename_ref)
            E_titles_ref, data_energy_ref, data_ref_asym, data_smooth_ref = read_energyresult_dat(
                path_filename_ref, mode=mode)
            filename_ref = name + f"_iter-{i_iter:04d}.npz"
            path_filename_ref = os.path.join(REF_DIR_INTEGRATE, filename_ref)
            E_titles_ref, data_energy_ref, data_new, data_smooth_ref = read_energyresult_dat(
                path_filename_ref, mode=mode)
            assert data_new == approx(data_ref_sym + data_ref_asym, abs=1e-8)

    return _inner
