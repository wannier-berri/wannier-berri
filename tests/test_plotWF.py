import glob
import os
import shutil
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
    os.chdir(cwd)
