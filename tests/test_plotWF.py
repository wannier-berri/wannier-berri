import glob
import os
import shutil

import numpy as np
from wannierberri.w90files.unk import UNK
import wannierberri as wberri
from .common import OUTPUT_DIR, ROOT_DIR, REF_DIR


def test_UNK():
    path = os.path.join(ROOT_DIR, "data", "diamond-444")
    unk = UNK(path=path)


def test_w90data_unk():

    cwd = os.getcwd()

    tmp_dir = os.path.join(OUTPUT_DIR, "diamond-444-plotWF")

    # Check if the directory exists
    if os.path.exists(tmp_dir):
        # Remove the directory and all its contents
        # shutil.rmtree(tmp_dir)
        # print(f"Directory {tmp_dir} has been removed.")
        print(f"Directory {tmp_dir} already exists.")
    else:
        os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    data_dir = os.path.join(ROOT_DIR, "data", "diamond-444")
    prefix = "diamond"

    # for ext in ["mmn", "amn", "eig", "win"]:
    #     shutil.copy(os.path.join(data_dir, prefix + "." + ext),
    #                 os.path.join(tmp_dir, prefix + "." + ext))
    # for f in glob.glob(os.path.join(data_dir, "UNK*")):
    #     shutil.copy(f, tmp_dir)
    print("prefix = ", prefix)
    # Read the data from the Wanier90 inputs
    w90data = wberri.w90files.Wannier90data(seedname=prefix, readfiles=["amn", "mmn", "eig", "win", "unk"])
    
    w90data.wannierise(
        froz_min=-8,
        froz_max=20,
        num_iter=1000,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        print_progress_every=20,
        sitesym=False,
        localise=True
    )

    sc_max = 2
    sc_min = -1
    reduce_r_points = 1
    
    sc_origin, sc_basis, WF, rho, wcc_red, wcc_cart = w90data.plotWF(sc_min=sc_min, sc_max=sc_max, reduce_r_points=reduce_r_points)
    
    nr = np.prod(rho.shape[1:])
    print (f"wcc_cart = {wcc_cart}")
    norm = np.sum(rho,axis = (1,2,3))/nr
    print (f"norm  = {norm}") 
    xsf_str = w90data.get_xsf(data=WF, sc_origin=sc_origin, sc_basis=sc_basis)
    open(f"WF_all.xsf", "w").write(xsf_str)
    os.chdir(cwd)
