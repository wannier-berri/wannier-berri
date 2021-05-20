"""Create system objects."""

import os
import tarfile
import shutil

import tbmodels
from pythtb import * 
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
def system_GaAs_tb(rootdir):
    """Create system for GaAs using _tb.dat data"""

    data_dir = os.path.join(rootdir, "data", "GaAs_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "GaAs_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "GaAs_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)

    seedname = os.path.join(data_dir, "GaAs_tb.dat")
    system = wberri.System_tb(seedname, berry=True, use_wcc_phase=False)

    return system

@pytest.fixture(scope="session")
def system_GaAs_tb_wcc(rootdir):
    """Create system for GaAs using _tb_dat data"""

    data_dir = os.path.join(rootdir, "data", "GaAs_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "GaAs_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "GaAs_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)
    # Load system
    seedname = os.path.join(data_dir, "GaAs_tb.dat")
    system = wberri.System_tb(seedname, berry=True, use_wcc_phase=True)

    return system

@pytest.fixture(scope="session")
def system_Haldane_TBmodels(rootdir):
    delta=0.2
    t=-1.0
    t2 =0.15*np.exp((1.j)*np.pi/2.)
    t2c=t2.conjugate()
    my_model = tbmodels.Model(
            on_site=[delta, -delta],uc = [[1.0,0.0,0.0],[0.5,np.sqrt(3.0)/2.0,0.0],[0.0,0.0,1.0]], dim=3, occ=1, pos=[[1./3.,1./3.,0.0],[2./3.,2./3.,0.0]]
            )
    my_model.add_hop(t, 0, 1, [ 0, 0, 0])
    my_model.add_hop(t, 1, 0, [ 1, 0, 0])
    my_model.add_hop(t, 1, 0, [ 0, 1, 0])
    my_model.add_hop(t2 , 0, 0, [ 1, 0, 0])
    my_model.add_hop(t2 , 1, 1, [ 1,-1, 0])
    my_model.add_hop(t2 , 1, 1, [ 0, 1, 0])
    my_model.add_hop(t2c, 1, 1, [ 1, 0, 0])
    my_model.add_hop(t2c, 0, 0, [ 1,-1, 0])
    my_model.add_hop(t2c, 0, 0, [ 0, 1, 0])
    
    # Load system
    system = wberri.System_TBmodels(my_model, berry=True, use_wcc_phase=False,periodic=(True,True,False))

    return system

@pytest.fixture(scope="session")
def system_Haldane_TBmodels_wcc(rootdir):
    """Create system for Fe using Tbmodels"""
    delta=0.2
    t=-1.0
    t2 =0.15*np.exp((1.j)*np.pi/2.)
    t2c=t2.conjugate()
    my_model = tbmodels.Model(
            on_site=[delta, -delta],uc = [[1.0,0.0,0.0],[0.5,np.sqrt(3.0)/2.0,0.0],[0.0,0.0,1.0]], dim=3, occ=1, pos=[[1./3.,1./3.,0.0],[2./3.,2./3.,0.0]]
            )
    my_model.add_hop(t, 0, 1, [ 0, 0, 0])
    my_model.add_hop(t, 1, 0, [ 1, 0, 0])
    my_model.add_hop(t, 1, 0, [ 0, 1, 0])
    my_model.add_hop(t2 , 0, 0, [ 1, 0, 0])
    my_model.add_hop(t2 , 1, 1, [ 1,-1, 0])
    my_model.add_hop(t2 , 1, 1, [ 0, 1, 0])
    my_model.add_hop(t2c, 1, 1, [ 1, 0, 0])
    my_model.add_hop(t2c, 0, 0, [ 1,-1, 0])
    my_model.add_hop(t2c, 0, 0, [ 0, 1, 0])
    
    # Load system
    system = wberri.System_TBmodels(my_model, berry=True, use_wcc_phase=True,periodic=(True,True,False))

    return system



@pytest.fixture(scope="session")
def system_Haldane_PythTB(rootdir):
    """Create system for Haldane model using PythTB"""
    lat=[[1.0,0.0,0.0],[0.5,np.sqrt(3.0)/2.0,0.0],[0.0,0.0,1.0]]
    orb=[[1./3.,1./3.,0.],[2./3.,2./3.,0.]]

    my_model=tb_model(2,3,lat,orb,per=[0,1])

    delta=0.2
    t=-1.0
    t2 =0.15*np.exp((1.j)*np.pi/2.)
    t2c=t2.conjugate()

    my_model.set_onsite([-delta,delta])
    my_model.set_hop(t, 0, 1, [ 0, 0, 0])
    my_model.set_hop(t, 1, 0, [ 1, 0, 0])
    my_model.set_hop(t, 1, 0, [ 0, 1, 0])
    my_model.set_hop(t2 , 0, 0, [ 1, 0, 0])
    my_model.set_hop(t2 , 1, 1, [ 1,-1, 0])
    my_model.set_hop(t2 , 1, 1, [ 0, 1, 0])
    my_model.set_hop(t2c, 1, 1, [ 1, 0, 0])
    my_model.set_hop(t2c, 0, 0, [ 1,-1, 0])
    my_model.set_hop(t2c, 0, 0, [ 0, 1, 0])
    # Load system
    system = wberri.System_PythTB(my_model, berry=True, use_wcc_phase=False ,periodic=(True,True,False))

    return system


@pytest.fixture(scope="session")
def system_Haldane_PythTB_wcc(rootdir):
    """Create system for Haldane model using PythTB"""
    lat=[[1.0,0.0,0.0],[0.5,np.sqrt(3.0)/2.0,0.0],[0.0,0.0,1.0]]
    orb=[[1./3.,1./3.,0.],[2./3.,2./3.,0.]]

    my_model=tb_model(2,3,lat,orb,per=[0,1])

    delta=0.2
    t=-1.0
    t2 =0.15*np.exp((1.j)*np.pi/2.)
    t2c=t2.conjugate()

    my_model.set_onsite([-delta,delta])
    my_model.set_hop(t, 0, 1, [ 0, 0, 0])
    my_model.set_hop(t, 1, 0, [ 1, 0, 0])
    my_model.set_hop(t, 1, 0, [ 0, 1, 0])
    my_model.set_hop(t2 , 0, 0, [ 1, 0, 0])
    my_model.set_hop(t2 , 1, 1, [ 1,-1, 0])
    my_model.set_hop(t2 , 1, 1, [ 0, 1, 0])
    my_model.set_hop(t2c, 1, 1, [ 1, 0, 0])
    my_model.set_hop(t2c, 0, 0, [ 1,-1, 0])
    my_model.set_hop(t2c, 0, 0, [ 0, 1, 0])
    
    # Load system
    system = wberri.System_PythTB(my_model, berry=True, use_wcc_phase=True,periodic=(True,True,False))

    return system



