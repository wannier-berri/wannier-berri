import pytest

from .common import OUTPUT_DIR, DATA_DIR
import os
from wannierberri.w90files.wandata import WannierData
import numpy as np


def split_line(line):
    """Split a line into a list of floats, ignoring parentheses and whitespace."""
    numbers = []
    for l in line.split():
        for k in l.split(','):
            m = k.strip('()\n')
            if len(m) > 0:
                numbers.append(m)
    try:
        numbers = np.array(numbers, dtype=int)
    except ValueError:
        try:
            numbers = np.array(numbers, dtype=float)
        except ValueError:
            numbers = np.array(numbers, dtype=str)
    return numbers


def compare_dat_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        assert len(lines1) == len(lines2), f"Files {file1} and {file2} have different number of lines."
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            arr1 = split_line(line1)
            arr2 = split_line(line2)
            assert arr1.shape == arr2.shape, f"Files {file1} and {file2} differ at line {i + 1}: shapes {arr1.shape} vs {arr2.shape}"
            assert arr1.dtype == arr2.dtype, f"Files {file1} and {file2} differ at line {i + 1}: dtypes {arr1.dtype} vs {arr2.dtype} (line content: {line1.strip()} vs {line2.strip()})"
            if arr1.dtype == float:
                assert np.allclose(arr1, arr2, atol=1e-8), f"Files {file1} and {file2} differ at line {i + 1}: {arr1} vs {arr2}"
            else:
                assert np.all(arr1 == arr2), f"Files {file1} and {file2} differ at line {i + 1}: {arr1} vs {arr2}"



def test_write_epw(create_files_Si_W90):
    seedname = os.path.join(DATA_DIR, "Si_Wannier90", "Si")
    wandata = WannierData.from_w90_files(seedname=seedname,
                                        files=['mmn', 'eig', 'chk', 'win'])

    out_dir = os.path.join(OUTPUT_DIR, "Si_Wannier90_epw")
    alat = -wandata.chk.real_lattice[0, 0] * 2
    print(f"alat = {alat} Angstrom")
    wandata.write_epw(path=out_dir, alat_angstrom=alat, eig_name="Si.eig")

    def check_file(name):
        ref_file = os.path.join(DATA_DIR, "Si_Wannier90", name)
        out_file = os.path.join(out_dir, name)
        compare_dat_files(out_file, ref_file)

    check_file("Ukk.dat")
    check_file("mmn.dat")
    check_file("bkvec.dat")
    check_file("Si.eig")



def test_alat_espresso():
    seedname_pw = os.path.join(DATA_DIR, "diamond", "di")
    from wannierberri.utility import get_alat_espresso
    alat = get_alat_espresso(seedname_pw)
    alat_ref = 3.227980984
    assert np.isclose(alat, alat_ref, atol=1e-8), f"Expected alat ~ {alat_ref}, got {alat}"


@pytest.mark.parametrize("exclude_bands, expected_nbndskip_occ", [
    ([], 0),
    ([5], None),
    ([3, 4,], None),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10),
    ([0, 1, 2, 7, 8, 9], 3),
    ([8, 9], 0),
    ([0, 1, 5, 6, 9], None),
])
def test_nbndskip_occ(exclude_bands, expected_nbndskip_occ):
    from wannierberri.w90files.chk import get_nbndskip_occ_from_exclude_bands
    num_bands_original = 10
    # if expected value is None - expect ValueError
    if expected_nbndskip_occ is None:
        with pytest.raises(ValueError):
            nbndskip_occ = get_nbndskip_occ_from_exclude_bands(num_bands_original, exclude_bands)
    else:
        nbndskip_occ = get_nbndskip_occ_from_exclude_bands(num_bands_original, exclude_bands)
        assert nbndskip_occ == expected_nbndskip_occ, f"Expected nbndskip_occ = {expected_nbndskip_occ}, got {nbndskip_occ}"
