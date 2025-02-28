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
        shutil.rmtree(tmp_dir)
        print(f"Directory {tmp_dir} has been removed.")
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    data_dir = os.path.join(ROOT_DIR, "data", "diamond-444")
    prefix = "diamond"

    for ext in ["mmn", "amn", "eig", "win"]:
        shutil.copy(os.path.join(data_dir, prefix + "." + ext),
                    os.path.join(tmp_dir, prefix + "." + ext))
    for f in glob.glob(os.path.join(data_dir, "UNK*")):
        shutil.copy(f, tmp_dir)
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

    sc_max_size = 1
    grid_r = w90data.unk.grid_size
    x = np.arange(-sc_max_size,sc_max_size+1,1./grid_r[0])
    y = np.arange(-sc_max_size,sc_max_size+1,1./grid_r[1])
    z = np.arange(-sc_max_size,sc_max_size+1,1./grid_r[2])

    WF = w90data.plotWF(sc_max_size=sc_max_size)
    rho = np.sum((WF*WF.conj()).real,axis = 4)
    
    wcc_x = np.sum(rho* x[None,:,None,None,],axis = (1,2,3))
    wcc_y = np.sum(rho* y[None,None,:,None,],axis = (1,2,3))
    wcc_z = np.sum(rho* z[None,None,None,:],axis = (1,2,3))
    wcc_red = np.array([wcc_x,wcc_y,wcc_z]).T
    wcc_cart = np.dot(wcc_red,w90data.chk.real_lattice)
    print (f"wcc_cart = {wcc_cart}")
    norm = np.sum(rho,axis = (1,2,3))
    print (f"norm  = {norm}")


    os.chdir(cwd)
