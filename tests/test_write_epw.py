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
                                        files=['mmn', 'eig', 'chk'])

    wandata.chk.write_epw(os.path.join(OUTPUT_DIR, "Ukk.dat"))
    wandata.mmn.write_epw(os.path.join(OUTPUT_DIR, "mmn.dat"))

    chk_ref = os.path.join(DATA_DIR, "Si_Wannier90", "Ukk.dat")
    mmn_ref = os.path.join(DATA_DIR, "Si_Wannier90", "mmn.dat")

    compare_dat_files(os.path.join(OUTPUT_DIR, "mmn.dat"), mmn_ref)
    compare_dat_files(os.path.join(OUTPUT_DIR, "Ukk.dat"), chk_ref)
