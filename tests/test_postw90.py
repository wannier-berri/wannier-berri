""" The goal of these tests is to benchmark results with postw90.xm where possible
    for that we will use the util wannierberri.utils.postw90 which mimics the behavious of postw90.x
"""

from common import ROOT_DIR, TMP_DATA_DIR
from common_systems import create_W90_files
import numpy as np
import pytest
import os
import shutil
import wannierberri as wberri


def create_W90_files_tmp(seedname, tags_needed, data_dir, tmp_dir, win_file_postw90):
    """
    create a temporary folder to run postw90 and wanierberri.utils.postw90
    """

    data_dir_full = os.path.join(ROOT_DIR, "data", data_dir)

    _ = os.path.join(ROOT_DIR, TMP_DATA_DIR)
    if not os.path.exists(_):
        os.mkdir(_)

    create_W90_files(seedname, tags_needed, data_dir=data_dir_full)
    if tmp_dir is None:
        for i in range(10**6):
            dr = f"__tmp_{data_dir}_{i:07d}"
            if not os.path.exists(os.path.join(ROOT_DIR, TMP_DATA_DIR, dr)):
                tmp_dir = dr
                break

    if tmp_dir is None:
        raise RuntimeError(f"after {i} iterations no temporary directory was created")

    data_dir_tmp = os.path.join(TMP_DATA_DIR, tmp_dir)
    if os.path.exists(data_dir_tmp):
        shutil.rmtree(data_dir_tmp)
    os.mkdir(data_dir_tmp)
    assert "win" not in tags_needed, "tags_needed should not include win, as it is concatenated with the win_file_postw90"
    for tag in tags_needed:
        fn = "{}.{}".format(seedname, tag)
        os.symlink(os.path.join(data_dir_full, fn), os.path.join(data_dir_tmp, fn))
#    os.symlink(os.path.join(ROOT_DIR,"../wannierberri"),os.path.join(data_dir_tmp,"wannierberri"))

    fn = "{}.{}".format(seedname, "win")
    win_text = open(os.path.join(data_dir_full, fn), "r").read()
    print(win_text)
    win_text = win_file_postw90 + "\n\n" + "#" * 20 + "\n\n" + win_text
    with open(os.path.join(data_dir_tmp, fn), "w") as f:
        f.write(win_text)
    return data_dir_tmp



def error_message_pw90(precision, error, parameters, tmp_dir):
    return f"""Data did not match with postw90.x. The difference = {error} which is greater
hat the required precision {precision}
    The temporary directory is : {tmp_dir}
    The parameters of the calculation are :
{parameters}"""



@pytest.fixture(scope="module")
def check_postw90(check_command_output):
    def _inner(data_dir, seedname, win_file_postw90="", precision=-1e-7, tmp_dir=None, argv=[]):

        tags_needed = ["mmn", "chk", "eig"]
        tmp_dir = create_W90_files_tmp(seedname, tags_needed, data_dir, tmp_dir, win_file_postw90)
        check_command_output(["postw90.x", seedname], cwd=tmp_dir)
        data_pw90 = np.loadtxt(os.path.join(tmp_dir, seedname + "-ahc-fermiscan.dat"))

#        out = os.path.join(tmp_dir,"stdout_wberri")
#        check_command_output(["python3","-m","wannierberri.utils.postw90",seedname], cwd=tmp_dir,stdout_filename=out)
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        wberri.utils.postw90.main([seedname] + argv)
        os.chdir(cwd)

        # so far, hardcode it for AHC, later generalize
        data_wb = np.loadtxt(os.path.join(tmp_dir, seedname + "-ahc_iter-0000.dat"))
        maxval = np.max(abs(data_wb[:, 4:7]))
        if precision is None:
            precision = max(maxval / 1E12, 1E-11)
        elif precision < 0:
            precision = max(maxval * abs(precision), 1E-11)

        assert data_pw90[:, 1:4] == pytest.approx(data_wb[:, 1:4] / 100, abs=precision), error_message_pw90(precision,
                np.max(np.abs(data_pw90[:, 1:4] - data_wb[:, 1:4] / 100)), win_file_postw90, tmp_dir)

    return _inner



@pytest.mark.parametrize("ti", [True, False])
@pytest.mark.parametrize("uws", [True, False])
def test_Fe(ti, uws, check_postw90):
    parameters = f"""
berry = true
berry_task = ahc
berry_kmesh = 6 6 6
adpt_smr = false
smr_type = gauss
smr_fixed_en_width = 0.20
fermi_energy_min = 17.0
fermi_energy_max = 18.0
fermi_energy_step = 0.1
transl_inv={ti}
use_ws_distance = {uws}
"""
    check_postw90(
                    data_dir="Fe_Wannier90",
                    seedname="Fe",
                    win_file_postw90=parameters,
                    precision=-1e-6,
                    argv=["__wb_fft_lib=numpy"]
                 )
