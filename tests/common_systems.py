"""Create system objects."""

import os
import tarfile

import pytest
import numpy as np
import pickle
import wannierberri as wberri
import wannierberri.symmetry as SYM
from wannierberri import models as wb_models

from common import ROOT_DIR

symmetries_Fe = [SYM.C4z, SYM.C2x * SYM.TimeReversal, SYM.Inversion]
symmetries_Te = ["C3z", "C2x", "TimeReversal"]
symmetries_GaAs = [SYM.C4z * SYM.Inversion, SYM.TimeReversal, SYM.Rotation(3, [1, 1, 1])]
symmetries_Si = ["C4z", "C4x", "TimeReversal"]
symmetries_Mn3Sn = ["C3z"]

Efermi_Fe = np.linspace(17, 18, 11)
Efermi_Te_gpaw = np.linspace(4, 8, 11)
Efermi_Te_sparse = np.linspace(4, 8, 11)
Efermi_Fe_FPLO = np.linspace(-0.5, 0.5, 11)
Efermi_GaAs = np.linspace(7, 9, 11)
Efermi_Haldane = np.linspace(-3, 3, 11)
Efermi_CuMnAs_2d = np.linspace(-2, 2, 11)
Efermi_Chiral = np.linspace(-5, 8, 27)
omega_chiral = np.linspace(0, 1., 11)
omega_phonon = np.linspace(-0.01, 0.1, 23)
Efermi_Mn3Sn = np.linspace(2, 3, 11)


def create_W90_files(seedname, tags_needed, data_dir):
    """
    Extract the compressed amn and mmn data files.
    Create files listed in tags_needed using utils.mmn2uHu.
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

        nb_out_list = wberri.utils.mmn2uHu.run_mmn2uHu(
            seedname, INPUTDIR=data_dir, OUTDIR=str(data_dir) + "/reduced", **kwargs)
        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, "reduced_NB={0}".format(nb_out))
            os.rename(
                os.path.join(result_dir, "{0}_nbs={1}.{2}".format(seedname, nb_out, tag)),
                os.path.join(data_dir, "{}.{}".format(seedname, tag)))




@pytest.fixture(scope="session")
def create_files_Fe_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "Fe"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"]  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "Fe_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def create_files_GaAs_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "GaAs"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"]  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def system_Fe_W90(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.system.System_w90(
        seedname, berry=True, morb=True, SHCqiao=True, SHCryoo=True, transl_inv=False, use_wcc_phase=False)
    system.set_symmetry(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_sparse(create_files_Fe_W90, system_Fe_W90):
    """Create convert to sparse format (keeping all matrix elements) and back, to test interface"""

    params = system_Fe_W90.get_sparse({X: -1 for X in system_Fe_W90._XX_R.keys()})
    system = wberri.system.SystemSparse(real_lattice=params['real_lattice'], matrices=params['matrices'])
    system.set_symmetry(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_wcc(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.system.System_w90(seedname, morb=True, spin=True, SHCqiao=False, SHCryoo=False, transl_inv=False, use_wcc_phase=True)
    system.set_symmetry(symmetries_Fe)
    return system


def get_system_Fe_sym_W90(method=None, use_wcc_phase=False, use_ws=False, **kwargs):
    """Create system for Fe symmetrization using Wannier90 data"""

    data_dir = os.path.join(ROOT_DIR, "data", "Fe_sym_Wannier90")
    create_W90_files('Fe_sym', ['uHu'], data_dir)

    # Load system
    seedname = os.path.join(data_dir, "Fe_sym")
    system = wberri.system.System_w90(seedname, berry=True, morb=True, spin=True, use_ws=use_ws, use_wcc_phase=use_wcc_phase, **kwargs)
    system.set_symmetry(symmetries_Fe)
    if method is not None:
        system.symmetrize(
            proj=['Fe:sp3d2;t2g'],
            atom_name=['Fe'],
            positions=np.array([[0, 0, 0]]),
            magmom=[[0., 0., -2.31]],
            soc=True,
            DFT_code='qe',
            method=method)
    return system


@pytest.fixture(scope="session")
def system_Fe_sym_W90_old_wcc():
    return get_system_Fe_sym_W90(method="old", use_wcc_phase=True)


@pytest.fixture(scope="session")
def system_Fe_sym_W90_wcc():
    return get_system_Fe_sym_W90(method="new", use_wcc_phase=True)


@pytest.fixture(scope="session")
def system_Fe_W90_proj_set_spin(create_files_Fe_W90):
    system = get_system_Fe_sym_W90()
    system.set_spin_from_code(DFT_code="qe")
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_proj(create_files_Fe_W90):
    return get_system_Fe_sym_W90(SHCqiao=True)


@pytest.fixture(scope="session")
def system_Fe_W90_proj_ws(create_files_Fe_W90):
    return get_system_Fe_sym_W90(use_ws=True)


@pytest.fixture(scope="session")
def system_Fe_W90_disentangle(create_files_Fe_W90):
    """Create system for Fe symmetrization using Wannier90 data"""

    data_dir = os.path.join(ROOT_DIR, "data", "Fe_sym_Wannier90")
    create_W90_files('Fe_sym', ['uHu'], data_dir)
    w90data = wberri.system.Wannier90data(seedname=os.path.join(data_dir, 'Fe_sym'))
    with pytest.raises(RuntimeError):
        wberri.system.System_w90(w90data=w90data)
    # aidata.apply_outer_window(win_min=-8,win_max= 100 )
    w90data.disentangle(froz_min=-8,
                 froz_max=20,
                 num_iter=2000,
                 conv_tol=5e-7,
                 mix_ratio=0.9,
                 print_progress_every=100
                  )
    system = wberri.system.System_w90(w90data=w90data, berry=True, morb=True)
    del w90data
    return system


@pytest.fixture(scope="session")
def system_GaAs_W90(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data"""

    data_dir = create_files_GaAs_W90

    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, berry=True, morb=True, spin=True, transl_inv=False)
    system.set_symmetry(symmetries_GaAs)

    return system


@pytest.fixture(scope="session")
def system_GaAs_W90_wcc(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data with wcc phases"""

    data_dir = create_files_GaAs_W90
    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, morb=True, transl_inv=False, spin=True, use_wcc_phase=True)
    system.set_symmetry(symmetries_GaAs)

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
    system = wberri.system.System_tb(seedname, berry=True)
    system.set_symmetry(symmetries_GaAs)

    return system


def get_system_GaAs_sym_tb(method=None, use_wcc_phase=False, use_ws=False, symmetrize=True, berry=True):
    """Create system for GaAs using sym_tb.dat data"""

    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "GaAs_sym_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "GaAs_sym_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)

    seedname = os.path.join(data_dir, "GaAs_sym_tb.dat")
    system = wberri.system.System_tb(seedname, berry=berry, use_ws=use_ws, use_wcc_phase=use_wcc_phase)
    if symmetrize:
        system.symmetrize(
            positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
            atom_name=['Ga', 'As'],
            proj=['Ga:sp3', 'As:sp3'],
            soc=True,
            DFT_code='vasp',
            method=method)
    system.set_symmetry(symmetries_GaAs)
    return system


@pytest.fixture(scope="session")
def system_GaAs_sym_tb_old_wcc():
    """Create system for GaAs using sym_tb.dat data"""
    return get_system_GaAs_sym_tb(method="old", use_wcc_phase=True, use_ws=False)


@pytest.fixture(scope="session")
def system_GaAs_sym_tb_wcc():
    """Create system for GaAs using sym_tb.dat data"""
    return get_system_GaAs_sym_tb(method="new", use_wcc_phase=True, use_ws=False)


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
    system = wberri.system.System_tb(seedname, berry=True, use_wcc_phase=True)
    system.set_symmetry(symmetries_GaAs)
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
    system = wberri.system.System_tb(seedname, berry=True, use_wcc_phase=True)
    system.do_ws_dist(mp_grid=(2, 2, 2))
    return system


# Haldane model from TBmodels


@pytest.fixture(scope="session")
def system_Haldane_TBmodels():
    # Load system
    try:
        import tbmodels
        tbmodels  # just to avoid F401
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import tbmodels")
    model_tbmodels_Haldane = wb_models.Haldane_tbm(delta=0.2, hop1=-1.0, hop2=0.15)
    system = wberri.system.System_TBmodels(model_tbmodels_Haldane, berry=True)
    system.set_symmetry(["C3z"])
    return system


@pytest.fixture(scope="session")
def system_Haldane_TBmodels_internal():
    # Load system
    try:
        import tbmodels
        tbmodels  # just to avoid F401
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import tbmodels")
    model_tbmodels_Haldane = wb_models.Haldane_tbm(delta=0.2, hop1=-1.0, hop2=0.15)
    system = wberri.system.System_TBmodels(model_tbmodels_Haldane, berry=False)
    system.set_symmetry(["C3z"])
    return system


# Haldane model from PythTB
model_pythtb_Haldane = wb_models.Haldane_ptb(delta=0.2, hop1=-1.0, hop2=0.15)


@pytest.fixture(scope="session")
def system_Haldane_PythTB():
    """Create system for Haldane model using PythTB"""
    # Load system
    system = wberri.system.System_PythTB(model_pythtb_Haldane, berry=True)
    system.set_symmetry(["C3z"])
    return system


@pytest.fixture(scope="session")
def system_Haldane_PythTB_wrong_mat():
    """Create system for Haldane model using PythTB - contains a wrong R-matrix to test exception"""
    # Load system
    system = wberri.system.System_PythTB(model_pythtb_Haldane, use_wcc_phase=True)
    system.set_R_mat('abracadabra', system.get_R_mat('Ham') * 4)
    return system


# Chiral model
# A chiral system that also breaks time-reversal. It can be used to test almost any quantity.
model_Chiral_left = wb_models.Chiral(
    delta=2, hop1=1, hop2=1. / 3, phi=np.pi / 10, hopz_left=0.2, hopz_right=0.0, hopz_vert=0)
model_Chiral_left_TR = wb_models.Chiral(
    delta=2, hop1=1, hop2=1. / 3, phi=-np.pi / 10, hopz_left=0.2, hopz_right=0.0, hopz_vert=0)
model_Chiral_right = wb_models.Chiral(
    delta=2, hop1=1, hop2=1. / 3, phi=np.pi / 10, hopz_left=0.0, hopz_right=0.2, hopz_vert=0)


@pytest.fixture(scope="session")
def system_Chiral_left():
    system = wberri.system.System_PythTB(model_Chiral_left, use_wcc_phase=True)
    system.set_symmetry(["C3z"])
    system.set_spin([1, -1])
    return system


@pytest.fixture(scope="session")
def system_Chiral_left_TR():
    system = wberri.system.System_PythTB(model_Chiral_left_TR, use_wcc_phase=True)
    system.set_symmetry(["C3z"])
    system.set_spin([-1, 1])
    return system


@pytest.fixture(scope="session")
def system_Chiral_right():
    system = wberri.system.System_PythTB(model_Chiral_right, use_wcc_phase=True)
    system.set_symmetry(["C3z"])
    system.set_spin([1, -1])
    return system


# Systems from FPLO code interface


@pytest.fixture(scope="session")
def system_Fe_FPLO():
    """Create system for Fe using  FPLO  data"""
    path = os.path.join(ROOT_DIR, "data", "Fe_FPLO", "+hamdata")
    system = wberri.system.System_fplo(path, use_wcc_phase=False, morb=True, spin=True, use_ws=False)
    system.set_symmetry(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_FPLO_wcc():
    """Create system for Fe using  FPLO  data"""
    path = os.path.join(ROOT_DIR, "data", "Fe_FPLO", "+hamdata")
    system = wberri.system.System_fplo(path, use_wcc_phase=True, morb=True, spin=True, use_ws=False)
    system.set_symmetry(symmetries_Fe)
    return system


# CuMnAs 2D model
# These parameters provide ~0.4eV gap between conduction and valence bands
# and splitting into subbands is within 0.04 eV
model_CuMnAs_2d_broken = wb_models.CuMnAs_2d(nx=0, ny=1, nz=0, hop1=1, hop2=0.08, l=0.8, J=1, dt=0.01)


@pytest.fixture(scope="session")
def system_CuMnAs_2d_broken():
    system = wberri.system.System_PythTB(model_CuMnAs_2d_broken, use_wcc_phase=True)
    return system


# Systems from ASE+gpaw code interface

@pytest.fixture(scope="session")
def data_Te_ASE():
    """read data for Te from ASE+GPAW"""
    try:
        import gpaw
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import gpaw")
    import ase
    import ase.dft.wannier

    path = os.path.join(ROOT_DIR, "data", "Te_ASE")
    calc = gpaw.GPAW(os.path.join(path, "Te.gpw"))
    wan = ase.dft.wannier.Wannier(nwannier=12, calc=calc, file=os.path.join(path, 'wannier-12.json'))
    return wan, calc


@pytest.fixture(scope="session")
def system_Te_ASE(data_Te_ASE):
    """Create system for Te using  ASE+GPAW data with use_wcc_phase=False"""
    wan, calc = data_Te_ASE
    system = wberri.system.System_ASE(wan, ase_calc=calc, use_wcc_phase=False, berry=True)
    system.set_symmetry(symmetries_Te)
    return system


@pytest.fixture(scope="session")
def system_Te_ASE_wcc(data_Te_ASE):
    """Create system for Te using  ASE+GPAW data with use_wcc_phase=True"""
    wan, calc = data_Te_ASE
    system = wberri.system.System_ASE(wan, ase_calc=calc, use_wcc_phase=True, berry=False)
    system.set_symmetry(symmetries_Te)
    return system


@pytest.fixture(scope="session")
def system_Te_sparse():
    """Create system for Te using symmetrized Wannier functions through a sparse interface"""
    path = os.path.join(ROOT_DIR, "data", "Te_sparse", "parameters_Te_low.pickle")
    param = pickle.load(open(path, "rb"))
    system = wberri.system.SystemSparse(**param)
    system.set_symmetry(symmetries_Te)
    return system


@pytest.fixture(scope="session")
def system_Phonons_Si():
    """Create system of phonons of Si using  QE data"""
    path = os.path.join(ROOT_DIR, "data", "Si_phonons/si")
    system = wberri.system.System_Phonon_QE(path, use_ws=True, asr=True)
    system.set_symmetry(symmetries_Si)
    return system


@pytest.fixture(scope="session")
def system_Phonons_GaAs():
    """Create system of phonons of Si using  QE data"""
    path = os.path.join(ROOT_DIR, "data", "GaAs_phonons/GaAs")
    system = wberri.system.System_Phonon_QE(path, use_ws=True, asr=True)
    system.set_symmetry(symmetries_GaAs)
    return system




def get_system_Mn3Sn_sym_tb(method, use_wcc_phase=False, use_ws=False):
    data_dir = os.path.join(ROOT_DIR, "data", "Mn3Sn_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "Mn3Sn_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "Mn3Sn_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)

    seedname = os.path.join(data_dir, "Mn3Sn_tb.dat")
    system = wberri.system.System_tb(seedname, berry=True, use_ws=use_ws, use_wcc_phase=use_wcc_phase)
    system.symmetrize(
            positions=np.array([
                [0.6666667, 0.8333333, 0],
                [0.1666667, 0.3333333, 0],
                [0.6666667, 0.3333333, 0],
                [0.3333333, 0.1666667, 0.5],
                [0.8333333, 0.6666667, 0.5],
                [0.3333333, 0.6666667, 0.5],
                [0.8333333, 0.1666667, 0.5],
                [0.1666667, 0.8333333, 0]]),
            atom_name=['Mn'] * 6 + ['Sn'] * 2,
            proj=['Mn:s;d', 'Sn:p'],
            soc=True,
            magmom=[
                [0, 2, 0],
                [np.sqrt(3), -1, 0],
                [-np.sqrt(3), -1, 0],
                [0, 2, 0],
                [np.sqrt(3), -1, 0],
                [-np.sqrt(3), -1, 0],
                [0, 0, 0],
                [0, 0, 0]],
            DFT_code='vasp',
            method=method)
    return system


@pytest.fixture(scope="session")
def system_Mn3Sn_sym_tb_old_wcc():
    """Create system for Mn3Sn using _tb.dat data"""
    return get_system_Mn3Sn_sym_tb(method="old", use_wcc_phase=True, use_ws=False)


@pytest.fixture(scope="session")
def system_Mn3Sn_sym_tb_wcc():
    """Create system for Mn3Sn using _tb.dat data"""
    return get_system_Mn3Sn_sym_tb(method="new", use_wcc_phase=True, use_ws=False)


###################################
# Isotropic effective mas s model #
###################################
mass_kp_iso = 1.912
kmax_kp = 2.123


def ham_mass_iso(k):
    return np.array([[np.dot(k, k) / (2 * mass_kp_iso)]])


def dham_mass_iso(k):
    return np.array(k).reshape(1, 1, 3) / mass_kp_iso


def d2ham_mass_iso(k):
    return np.eye(3).reshape(1, 1, 3, 3) / mass_kp_iso


@pytest.fixture(scope="session")
def system_kp_mass_iso_0():
    return wberri.system.SystemKP(Ham=ham_mass_iso, kmax=kmax_kp)


@pytest.fixture(scope="session")
def system_kp_mass_iso_1():
    return wberri.system.SystemKP(Ham=ham_mass_iso, derHam=dham_mass_iso, kmax=kmax_kp)


@pytest.fixture(scope="session")
def system_kp_mass_iso_2():
    return wberri.system.SystemKP(Ham=ham_mass_iso, derHam=dham_mass_iso, der2Ham=d2ham_mass_iso, kmax=kmax_kp)


###################################
# AnIsotropic effective mas s model #
###################################
kmax_kp_aniso = 2.1

inv_mass_kp_aniso = np.array([[0.86060064, 0.19498375, 0.09798235],
 [0.01270294, 0.77373333, 0.00816169],
 [0.15613272, 0.11770323, 0.71668436]])


def ham_mass_aniso(k):
    e = np.dot(k, np.dot(inv_mass_kp_aniso, k))
    return np.array([[e]])


def dham_mass_aniso(k):
    return (np.dot(k, inv_mass_kp_aniso) + np.dot(inv_mass_kp_aniso, k)).reshape(1, 1, 3)


def d2ham_mass_aniso(k):
    return (inv_mass_kp_aniso + inv_mass_kp_aniso.T).reshape(1, 1, 3, 3)



@pytest.fixture(scope="session")
def system_kp_mass_aniso_0():
    return wberri.system.SystemKP(Ham=ham_mass_aniso, kmax=kmax_kp_aniso)


@pytest.fixture(scope="session")
def system_kp_mass_aniso_1():
    return wberri.system.SystemKP(Ham=ham_mass_aniso, derHam=dham_mass_aniso, kmax=kmax_kp_aniso)


@pytest.fixture(scope="session")
def system_kp_mass_aniso_2():
    return wberri.system.SystemKP(Ham=ham_mass_aniso, derHam=dham_mass_aniso, der2Ham=d2ham_mass_aniso, kmax=kmax_kp_aniso)


@pytest.fixture(scope="session")
def system_random():
    system = wberri.system.SystemRandom(num_wann=6, nRvec=20, max_R=4, berry=True, morb=True, SHCryoo=True, SHCqiao=True)
    # system.save_npz("randomsys")
    return system


@pytest.fixture(scope="session")
def system_random_load_bare():
    system = wberri.system.System_R(berry=True, morb=True, SHCryoo=True, SHCqiao=True)
    system.load_npz(path=os.path.join(ROOT_DIR, "data", "random"))
    return system
