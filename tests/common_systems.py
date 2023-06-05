"""Create system objects."""

import os
import tarfile

import pytest
import numpy as np

import wannierberri as wberri
import wannierberri.symmetry as SYM
from wannierberri import models as wb_models

from common import ROOT_DIR

symmetries_Fe = [SYM.C4z, SYM.C2x * SYM.TimeReversal, SYM.Inversion]
symmetries_GaAs = [SYM.C4z * SYM.Inversion, SYM.TimeReversal, SYM.Rotation(3, [1,1,1])]

Efermi_Fe = np.linspace(17, 18, 11)
Efermi_Fe_FPLO = np.linspace(-0.5, 0.5, 11)
Efermi_GaAs = np.linspace(7, 9, 11)
Efermi_Haldane = np.linspace(-3, 3, 11)
Efermi_CuMnAs_2d = np.linspace(-2, 2, 11)
Efermi_Chiral = np.linspace(-5, 8, 27)

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
def system_Fe_W90(create_files_Fe_W90):
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
def system_Fe_W90_wcc(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(seedname, morb=True, SHCqiao=False, SHCryoo=False,
           transl_inv=False, use_wcc_phase=True )
    system.set_symmetry(symmetries_Fe)
    return system

@pytest.fixture(scope="session")
def system_Fe_sym_W90(create_files_Fe_W90):
    """Create system for Fe symmetrization using Wannier90 data"""
    
    data_dir = os.path.join(ROOT_DIR, "data", "Fe_sym_Wannier90")
    create_W90_files('Fe_sym', [], data_dir)
    
    # Load system
    seedname = os.path.join(data_dir, "Fe_sym")
    system = wberri.System_w90(seedname, berry=True, morb=False,
           spin = True, use_ws = False )
    system.set_symmetry(symmetries_Fe)
    system.symmetrize(
             proj = ['Fe:sp3d2;t2g'],
             atom_name = ['Fe'],
             positions = [[0,0,0]],
             magmom = [[0.,0.,-2.31]],soc=True,
             DFT_code = 'qe')
    
    return system

@pytest.fixture(scope="session")
def system_Fe_W90_transl_inv_diag(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(
        seedname, berry=True, transl_inv=True, use_wcc_phase=False)
    system.set_symmetry(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_transl_inv_full(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(
        seedname, berry=True, transl_inv_offdiag=True, use_wcc_phase=False)
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
def system_GaAs_sym_tb():
    """Create system for GaAs using sym_tb.dat data"""

    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "GaAs_sym_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "GaAs_sym_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)

    seedname = os.path.join(data_dir, "GaAs_sym_tb.dat")
    system = wberri.System_tb(seedname, berry=True, use_ws = False )
    system.symmetrize(positions = np.array([[0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25]]),
                atom_name = ['Ga','As'],
                proj = ['Ga:sp3','As:sp3'],
                soc=True,
                DFT_code = 'vasp')
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

# Haldane model from TBmodels
model_tbmodels_Haldane = wb_models.Haldane_tbm(delta=0.2,hop1=-1.0,hop2 =0.15)

@pytest.fixture(scope="session")
def system_Haldane_TBmodels():
    # Load system
    system = wberri.System_TBmodels(model_tbmodels_Haldane, berry=True)
    system.set_symmetry(["C3z"])
    return system

@pytest.fixture(scope="session")
def system_Haldane_TBmodels_internal():
    # Load system
    system = wberri.System_TBmodels(model_tbmodels_Haldane, berry=False)
    system.set_symmetry(["C3z"])
    return system


# Haldane model from PythTB
model_pythtb_Haldane = wb_models.Haldane_ptb(delta=0.2,hop1=-1.0,hop2 =0.15)

@pytest.fixture(scope="session")
def system_Haldane_PythTB():
    """Create system for Haldane model using PythTB"""
    # Load system
    system = wberri.System_PythTB(model_pythtb_Haldane, berry=True)
    system.set_symmetry(["C3z"])
    return system


# Chiral model
# A chiral system that also breaks time-reversal. It can be used to test almost any quantity.
model_Chiral_left = wb_models.Chiral(delta=2, hop1=1, hop2=1./3, phi=np.pi/10, hopz_left=0.2,
                                     hopz_right=0.0, hopz_vert=0)
model_Chiral_left_TR = wb_models.Chiral(delta=2, hop1=1, hop2=1./3, phi=-np.pi/10, hopz_left=0.2,
                                        hopz_right=0.0, hopz_vert=0)
model_Chiral_right = wb_models.Chiral(delta=2, hop1=1, hop2=1./3, phi=np.pi/10, hopz_left=0.0,
                                      hopz_right=0.2, hopz_vert=0)

@pytest.fixture(scope="session")
def system_Chiral_left():
    system = wberri.System_PythTB(model_Chiral_left, use_wcc_phase=True)
    system.set_symmetry(["C3z"])
    return system

@pytest.fixture(scope="session")
def system_Chiral_left_TR():
    system = wberri.System_PythTB(model_Chiral_left_TR, use_wcc_phase=True)
    system.set_symmetry(["C3z"])
    return system

@pytest.fixture(scope="session")
def system_Chiral_right():
    system = wberri.System_PythTB(model_Chiral_right, use_wcc_phase=True)
    system.set_symmetry(["C3z"])
    return system


# Systems from FPLO code interface

@pytest.fixture(scope="session")
def system_Fe_FPLO():
    """Create system for Fe using  FPLO  data"""
    path = os.path.join(ROOT_DIR, "data", "Fe_FPLO","+hamdata")
    system = wberri.System_fplo(path, use_wcc_phase=False,morb=True,spin=True )
    system.set_symmetry(symmetries_Fe)
    return system

@pytest.fixture(scope="session")
def system_Fe_FPLO_wcc():
    """Create system for Fe using  FPLO  data"""
    path = os.path.join(ROOT_DIR, "data", "Fe_FPLO","+hamdata")
    system = wberri.System_fplo(path, use_wcc_phase=True,morb=True,spin=True )
    system.set_symmetry(symmetries_Fe)
    return system


# CuMnAs 2D model
# These parameters provide ~0.4eV gap between conduction and valence bands
# and splitting into subbands is within 0.04 eV
model_CuMnAs_2d_broken = wb_models.CuMnAs_2d(nx=0,ny=1,nz=0,hop1=1,hop2=0.08,l=0.8,J=1,dt=0.01)

@pytest.fixture(scope="session")
def system_CuMnAs_2d_broken():
    system = wberri.System_PythTB(model_CuMnAs_2d_broken, use_wcc_phase=True)
    return system
