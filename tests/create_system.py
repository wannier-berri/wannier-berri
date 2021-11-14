"""Create system objects."""

import os
import tarfile
import shutil

import pytest
import numpy as np

import wannierberri as wberri
from wannierberri import models as wb_models

from conftest import ROOT_DIR

@pytest.fixture(scope="session")
def symmetries_Fe():
    """ liust of symmetries for bcc iron"""
    sym=wberri.symmetry
    return ["C4z","C2x*TimeReversal","Inversion"]

@pytest.fixture(scope="session")
def symmetries_GaAs():
    sym=wberri.symmetry
    return ["C4z",sym.TimeReversal,sym.Rotation(3,[1,1,1])]


def create_W90_files(seedname, tags_needed, data_dir):
    """
    Extract the compressed amn and mmn data files.
    Create files listed in tags_needed using mmn2uHu.
    """

    # Extract files if is not already done
    for tag in ["mmn", "amn"]:
        if not os.path.isfile(os.path.join(data_dir, "{}.{}".format(seedname, tag))):
            tar = tarfile.open(os.path.join(data_dir, "{}.{}.tar.gz".format(seedname, tag)))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)

    # Compute tags only if the corresponding files do not exist
    tags_compute = []
    for tag in tags_needed:
        if not os.path.isfile(os.path.join(data_dir, "{}.{}".format(seedname, tag))):
            tags_compute.append(tag)

    if len(tags_compute) > 0:
        kwargs = {}
        for tag in tags_compute:
            kwargs["write" + tag.upper()] = True

        nb_out_list = wberri.mmn2uHu.run_mmn2uHu(seedname, INPUTDIR=data_dir,
            OUTDIR=str(data_dir)+"/reduced", **kwargs)
        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, "reduced_NB={0}".format(nb_out))
            os.rename(os.path.join(result_dir, "{0}_nbs={1}.{2}".format(seedname, nb_out, tag)),
                      os.path.join(data_dir, "{}.{}".format(seedname, tag)))


@pytest.fixture(scope="session")
def create_files_Fe_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "Fe"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"] # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "Fe_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def create_files_GaAs_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "GaAs"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"] # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def system_Fe_W90(create_files_Fe_W90,symmetries_Fe):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(seedname, berry=True, morb=True, 
            SHCqiao=True, SHCryoo=True,
           transl_inv=False, use_wcc_phase=False )
    system.set_symmetry(symmetries_Fe)
    return system

@pytest.fixture(scope="session")
def system_Fe_W90_wcc(create_files_Fe_W90,symmetries_Fe):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(seedname, morb=True, SHCqiao=False, SHCryoo=False,
           transl_inv=False, use_wcc_phase=True )
    system.set_symmetry(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_GaAs_W90(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data"""

    data_dir = create_files_GaAs_W90

    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.System_w90(seedname, berry=True , morb=True, transl_inv=False)

    return system



@pytest.fixture(scope="session")
def system_GaAs_W90_wcc(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data with wcc phases"""

    data_dir = create_files_GaAs_W90

    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.System_w90(seedname, morb=True, 
           transl_inv=False, use_wcc_phase=True)

    return system


@pytest.fixture(scope="session")
def system_GaAs_tb():
    """Create system for GaAs using _tb.dat data"""

    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "GaAs_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "GaAs_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)

    seedname = os.path.join(data_dir, "GaAs_tb.dat")
    system = wberri.System_tb(seedname, berry=True)

    return system

@pytest.fixture(scope="session")
def system_GaAs_tb_wcc():
    """Create system for GaAs using _tb_dat data"""

    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "GaAs_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "GaAs_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)
    # Load system
    seedname = os.path.join(data_dir, "GaAs_tb.dat")
    system = wberri.System_tb(seedname, berry=True, use_wcc_phase=True)

    return system


@pytest.fixture(scope="session")
def system_GaAs_tb_wcc_ws():
    """Create system for GaAs using _tb_dat data"""

    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "GaAs_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "GaAs_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)
    # Load system
    seedname = os.path.join(data_dir, "GaAs_tb.dat")
    system = wberri.System_tb(seedname, berry=True, use_wcc_phase=True,use_ws=True,mp_grid=(2,2,2))

    return system

@pytest.fixture(scope="session")
def tbmodels_Haldane():
    return wb_models.Haldane_tbm(delta=0.2,hop1=-1.0,hop2 =0.15)

@pytest.fixture(scope="session")
def system_Haldane_TBmodels(tbmodels_Haldane):
    
    # Load system
    system = wberri.System_TBmodels(tbmodels_Haldane, berry=True)
    system.set_symmetry(["C3z"])
    return system

@pytest.fixture(scope="session")
def system_Haldane_TBmodels_internal(tbmodels_Haldane):
    
    # Load system
    system = wberri.System_TBmodels(tbmodels_Haldane, berry=False)
    system.set_symmetry(["C3z"])
    return system



@pytest.fixture(scope="session")
def pythtb_Haldane():
    return wb_models.Haldane_ptb(delta=0.2,hop1=-1.0,hop2 =0.15)


@pytest.fixture(scope="session")
def system_Haldane_PythTB(pythtb_Haldane):
    """Create system for Haldane model using PythTB"""
    # Load system
    system = wberri.System_PythTB(pythtb_Haldane, berry=True)
    system.set_symmetry(["C3z"])
    return system




@pytest.fixture(scope="session")
def ChiralModel():
    return wb_models.Chiral(delta=2, hop1=1, hop2=1./3,  phi=np.pi/10, hopz=0.2)



@pytest.fixture(scope="session")
def system_Chiral(ChiralModel):
    """Create a chiral system that also breaks time-reversal
       can be used to test almost any quantity"""
    system = wberri.System_PythTB(ChiralModel, use_wcc_phase=True)
    system.set_symmetry(["C3z"])
    return system
