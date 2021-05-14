"""Create system objects."""

import os
import tarfile
import shutil

import pytest

import wannierberri as wberri

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
def create_files_Fe_W90(rootdir):
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "Fe"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"] # Files to calculate if they do not exist
    data_dir = os.path.join(rootdir, "data", "Fe_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def create_files_GaAs_W90(rootdir):
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "GaAs"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"] # Files to calculate if they do not exist
    data_dir = os.path.join(rootdir, "data", "GaAs_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def system_Fe_W90(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(seedname, berry=True, SHCqiao=True, SHCryoo=True,
           transl_inv=False, use_wcc_phase=False)

    return system


@pytest.fixture(scope="session")
def system_GaAs_W90(create_files_GaAs_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_GaAs_W90

    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.System_w90(seedname, berry=True, SHCqiao=True, SHCryoo=True,
           transl_inv=False, use_wcc_phase=False,degen_thresh=0.005)

    return system



@pytest.fixture(scope="session")
def system_GaAs_W90_wcc(create_files_GaAs_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_GaAs_W90

    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.System_w90(seedname, berry=True, SHCqiao=True, SHCryoo=True,
           transl_inv=False, use_wcc_phase=True,degen_thresh=0.005)

    return system



@pytest.fixture(scope="session")
def system_Fe_W90_wcc(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(seedname, berry=True, SHCqiao=True, SHCryoo=True, 
            transl_inv=False, use_wcc_phase=True)

    return system

@pytest.fixture(scope="session")
def system_Fe_tb():
    """Create system for Fe using _tb.dat data"""

    data_dir = os.path.join(rootdir, "data", "Te_Wannier90")

    # Load system
    seedname = 'Fe_tb.dat'
    system = wberri.System_tb(seedname, berry=True, use_wcc_phase=False)

    return system

@pytest.fixture(scope="session")
def system_Fe_tb_wcc():
    """Create system for Fe using _tb_dat data"""

    data_dir = os.path.join(rootdir, "data", "Fe_Wannier90")

    # Load system
    seedname = 'Fe_tb.dat'
    system = wberri.System_tb(seedname, berry=True, use_wcc_phase=True)

    return system

@pytest.fixture(scope="session")
def system_Fe_Tbmodels():
    """Create system for Fe using Tbmodels"""
    seedname = 'Fe'

    data_dir = os.path.join(rootdir, "data", "Fe_Wannier90")
    for tag in ['hr','wsvec']:
        if not os.path.isfile(os.path.join(data_dir, "{}_{}.dat".format(seedname, tag))):
            tar = tarfile.open(os.path.join(data_dir, "{}_{}.dat.tar.gz".format(seedname, tag)))
                for tarinfo in tar:
                    tar.extract(tarinfo, data_dir)
    model_tbmodels = tbmodels.Model.from_wannier_files(
                hr_file= data_dir+seedname+'_hr.dat',
                wsvec_file= data_dir+seedname+'_wsvec.dat',
                xyz_file= data_dir+seedname+'_centres.xyz',
                win_file= data_dir+seedname+'.win'
                )

    # Load system
    system = wberri.System_TBmodels(model_tbmodels, berry=True, use_wcc_phase=False)

    return system

@pytest.fixture(scope="session")
def system_Fe_Tbmodels_wcc():
    """Create system for Fe using Tbmodels"""
    seedname = 'Fe'

    data_dir = os.path.join(rootdir, "data", "Fe_Wannier90")
    for tag in ['hr','wsvec']:
        if not os.path.isfile(os.path.join(data_dir, "{}_{}.dat".format(seedname, tag))):
            tar = tarfile.open(os.path.join(data_dir, "{}_{}.dat.tar.gz".format(seedname, tag)))
                for tarinfo in tar:
                    tar.extract(tarinfo, data_dir)
    model_tbmodels = tbmodels.Model.from_wannier_files(
                hr_file= data_dir+seedname+'_hr.dat',
                wsvec_file= data_dir+seedname+'_wsvec.dat',
                xyz_file= data_dir+seedname+'_centres.xyz',
                win_file= data_dir+seedname+'.win'
                )

    # Load system
    system = wberri.System_TBmodels(model_tbmodels, berry=True, use_wcc_phase=True)

    return system



@pytest.fixture(scope="session")
def system_Fe_PythTB():
    """Create system for Fe using Tbmodels"""
    seedname = 'Fe'

    data_dir = os.path.join(rootdir, "data", "Fe_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "{}_tb.dat".format(seedname))):
        tar = tarfile.open(os.path.join(data_dir, "{}_tb.dat.tar.gz".format(seedname)))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)
    Te =w90(data_dir,seedname)
    model_pythtb=Te.model(min_hopping_norm=0.001)

    # Load system
    system = wberri.System_PythTB(model_pythtb, berry=True, use_wcc_phase=False)

    return system


@pytest.fixture(scope="session")
def system_Fe_PythTB_wcc():
    """Create system for Fe using Tbmodels"""
    seedname = 'Fe'

    data_dir = os.path.join(rootdir, "data", "Fe_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "{}_tb.dat".format(seedname))):
        tar = tarfile.open(os.path.join(data_dir, "{}_tb.dat.tar.gz".format(seedname)))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)
    Te =w90(data_dir,seedname)
    model_pythtb=Te.model(min_hopping_norm=0.001)

    # Load system
    system = wberri.System_PythTB(model_pythtb, berry=True, use_wcc_phase=True)

    return system
