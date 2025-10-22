"""Create system objects."""

import os
import tarfile
from gpaw import GPAW
import pytest
import numpy as np
import pickle
import wannierberri as wberri
from wannierberri import point_symmetry as SYM
from pathlib import Path
from wannierberri import models as wb_models
from irrep.spacegroup import SpaceGroup

from wannierberri.system.system_R import System_R
from wannierberri.w90files.soc import SOC
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.w90files.w90data_soc import Wannier90dataSOC

from wannierberri.system.system_soc import SystemSOC
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from .common import OUTPUT_DIR, ROOT_DIR

symmetries_Fe = [SYM.C4z, SYM.C2x * SYM.TimeReversal, SYM.Inversion]
symmetries_Te = ["C3z", "C2x", "TimeReversal"]
symmetries_GaAs = [SYM.C4z * SYM.Inversion, SYM.TimeReversal, SYM.Rotation(3, [1, 1, 1])]
symmetries_Si = ["C4z", "C4x", "TimeReversal"]

Efermi_Fe = np.linspace(17, 18, 11)
Efermi_Te_gpaw = np.linspace(4, 8, 11)
Efermi_Te_qe = np.linspace(4, 8, 11)
Efermi_Te_sparse = Efermi_Te_qe
Efermi_Fe_FPLO = np.linspace(-0.5, 0.5, 11)
Efermi_GaAs = np.linspace(7, 9, 11)
Efermi_Haldane = np.linspace(-3, 3, 11)
Efermi_CuMnAs_2d = np.linspace(-2, 2, 11)
Efermi_Chiral = np.linspace(-5, 8, 27)
omega_chiral = np.linspace(0, 1., 11)
omega_phonon = np.linspace(-0.01, 0.1, 23)
Efermi_Mn3Sn = np.linspace(2, 3, 11)


def create_W90_files(seedname, tags_needed, data_dir, tags_untar=["mmn", "amn"]):
    """
    Extract the compressed amn and mmn data files.
    Create files listed in tags_needed using utils.mmn2uHu.
    """

    # Extract files if is not already done
    for tag in tags_untar:
        if not os.path.isfile(os.path.join(data_dir, f"{seedname}.{tag}")):
            tar = tarfile.open(os.path.join(data_dir, f"{seedname}.{tag}.tar.gz"))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)

    # Compute tags only if the corresponding files do not exist
    tags_compute = []
    for tag in tags_needed:
        if not os.path.isfile(os.path.join(data_dir, f"{seedname}.{tag}")):
            tags_compute.append(tag)

    if len(tags_compute) > 0:
        kwargs = {}
        for tag in tags_compute:
            kwargs["write" + tag.upper()] = True

        nb_out_list = wberri.utils.mmn2uHu.run_mmn2uHu(
            seedname, INPUTDIR=data_dir, OUTDIR=str(data_dir) + "/reduced", **kwargs)
        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, f"reduced_NB={nb_out}")
            os.rename(
                os.path.join(result_dir, f"{seedname}_nbs={nb_out}.{tag}"),
                os.path.join(data_dir, f"{seedname}.{tag}"))




@pytest.fixture(scope="session")
def create_files_Fe_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "Fe"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"]  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "Fe_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def create_files_Fe_W90_npz(create_files_Fe_W90, system_Fe_W90):
    """Create symbolic links to the npz files"""

    seedname = "Fe"
    data_dir = Path(create_files_Fe_W90)
    data_dir_new = data_dir.joinpath("NPZ")
    data_dir_new.mkdir(exist_ok=True)

    def _link(ext):
        f = seedname + "." + ext
        try:
            data_dir_new.joinpath(f).symlink_to(data_dir.joinpath(f))
        except FileExistsError:
            pass

    _link("chk")
    for ext in ["eig", "mmn", "spn", "uHu", "sHu", "sIu"]:
        _link(ext + ".npz")
    return data_dir_new


@pytest.fixture(scope="session")
def create_files_GaAs_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "GaAs"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"]  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


def create_files_tb(dir, file):
    data_dir = os.path.join(ROOT_DIR, "data", dir)
    path_tb_file = os.path.join(data_dir, file)
    if not os.path.isfile(path_tb_file):
        tar = tarfile.open(os.path.join(data_dir, file + ".tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)
    return path_tb_file


@pytest.fixture(scope="session")
def create_files_Si_W90():
    """Create data files for Si: uHu, and uIu"""

    seedname = "Si"
    tags_needed = []  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "Si_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def system_Fe_W90(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.system.System_w90(
        seedname, berry=True, morb=True, SHCqiao=True, SHCryoo=True, transl_inv_MV=False,
        read_npz=False, overwrite_npz=True, write_npz_list=["uHu", "uIu", "spn", "sHu", "sIu"],
        write_npz_formatted=True,
        ws_dist_tol=1e-5)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_npz(create_files_Fe_W90_npz):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90_npz

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.system.System_w90(
        seedname, berry=True,
        morb=True, SHCqiao=True, SHCryoo=True,
        transl_inv_MV=False,
        read_npz=True, write_npz_list=[], overwrite_npz=False, write_npz_formatted=False,
        ws_dist_tol=1e-5)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_sparse(create_files_Fe_W90, system_Fe_W90):
    """Create convert to sparse format (keeping all matrix elements) and back, to test interface"""

    params = system_Fe_W90.get_sparse({X: -1 for X in system_Fe_W90._XX_R.keys()})
    system = wberri.system.SystemSparse(**params)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_WB_irreducible():
    """Create system for Fe from WB wannierisation based on irreducible kpoints only"""
    w90data = wberri.w90files.Wannier90data().from_npz(
        seedname=os.path.join(ROOT_DIR, "data", "Fe-444-sitesym", "wann-irred", "Fe_wan"),
        files=['chk', 'amn', 'mmn', 'spn', 'eig', 'symmetrizer'],
    )
    return wberri.system.System_w90(
        w90data=w90data, berry=True, spin=True, SHCqiao=True)



def get_system_Fe_sym_W90(symmetrize=False,
                          TR=False,
                          **kwargs):
    """Create system for Fe symmetrization using Wannier90 data"""

    data_dir = os.path.join(ROOT_DIR, "data", "Fe_sym_Wannier90")
    create_W90_files('Fe_sym', ['uHu', 'uIu', 'sHu', 'sIu'], data_dir)

    # Load system
    seedname = os.path.join(data_dir, "Fe_sym")
    system = wberri.system.System_w90(seedname, berry=True, morb=True, spin=True, SHCryoo=True,
                                      OSD=True, SHCqiao=True,
                                      ws_dist_tol=1e-5,
                                      transl_inv_MV=True,  # legacy
                                      **kwargs)
    system.set_pointgroup(symmetries_Fe)
    if symmetrize:
        system.symmetrize(
            proj=['Fe:sp3d2;t2g'],
            atom_name=['Fe'],
            positions=np.array([[0, 0, 0]]),
            magmom=[[0, 0, 0]] if TR else [[0., 0., -2.31]],
            soc=True,)
    return system


@pytest.fixture(scope="session")
def system_Fe_sym_W90():
    return get_system_Fe_sym_W90(symmetrize=True)


@pytest.fixture(scope="session")
def system_Fe_sym_W90_TR():
    return get_system_Fe_sym_W90(symmetrize=True, TR=False)


@pytest.fixture(scope="session")
def system_Fe_W90_proj_set_spin(create_files_Fe_W90):
    system = get_system_Fe_sym_W90()
    system.set_spin_interlaced()
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_proj(create_files_Fe_W90):
    return get_system_Fe_sym_W90()


PATH_Fe_GPAW = os.path.join(ROOT_DIR, "data", "Fe_gpaw")


@pytest.fixture(scope="session")
def soc_Fe_gpaw():
    """Create SOC object for Fe from GPAW calculation"""
    selected_bands_up = np.load(os.path.join(PATH_Fe_GPAW, "system_up.chk.npz"))['selected_bands']
    selected_bands_down = np.load(os.path.join(PATH_Fe_GPAW, "system_dw.chk.npz"))['selected_bands']
    soc = SOC.from_gpaw(os.path.join(PATH_Fe_GPAW, "Fe-nscf.gpw"))
    soc.select_bands(selected_bands_up=selected_bands_up, selected_bands_down=selected_bands_down)
    return soc



@pytest.fixture(scope="session")
def system_Fe_gpaw_up():
    """Create system for Fe up channel using GPAW wannierisation data"""
    system = System_R().load_npz(os.path.join(PATH_Fe_GPAW, "system_up"), load_all_XX_R=True)
    spacegroup = SpaceGroup.from_cell(real_lattice=system.real_lattice,
                                positions=[[0, 0, 0]], typat=[1],
                            magmom=[[0, 0, 0]])
    system.set_pointgroup(spacegroup=spacegroup)
    return system


@pytest.fixture(scope="session")
def system_Fe_gpaw_dw():
    """Create system for Fe down channel using GPAW wannierisation data"""
    system = System_R().load_npz(os.path.join(PATH_Fe_GPAW, "system_dw"), load_all_XX_R=True)
    spacegroup = SpaceGroup.from_cell(real_lattice=system.real_lattice,
                                positions=[[0, 0, 0]], typat=[1],
                            magmom=[[0, 0, 0]])
    system.set_pointgroup(spacegroup=spacegroup)
    return system


@pytest.fixture(scope="session")
def get_system_Fe_gpaw_soc(system_Fe_gpaw_up, system_Fe_gpaw_dw, soc_Fe_gpaw):
    def _inner(phi_deg=0, theta_deg=0, alpha_soc=1.0,
               do_symmetrize=False):
        theta = theta_deg / 180 * np.pi
        phi = phi_deg / 180 * np.pi

        sg = SpaceGroup.from_cell(real_lattice=2.87 * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]]) / 2,
                                positions=[[0, 0, 0]], typat=[1], include_TR=True, spinor=False)
        cell = dict(positions=[[0, 0, 0]], typat=[1], magmoms_on_axis=[1])

        system_dw = system_Fe_gpaw_dw
        system_up = system_Fe_gpaw_up

        soc = soc_Fe_gpaw
        chk_up = CHK.from_npz(os.path.join(PATH_Fe_GPAW, "system_up.chk.npz"))
        chk_dw = CHK.from_npz(os.path.join(PATH_Fe_GPAW, "system_dw.chk.npz"))
        system_soc = SystemSOC(system_up=system_up, system_down=system_dw, cell=cell)
        system_soc.set_soc_R(soc, chk_up=chk_up, chk_down=chk_dw)
        if do_symmetrize:
            projection_sp3d2 = Projection(position_num=[0, 0, 0], orbital='sp3d2', spacegroup=sg)
            projection_t2g = Projection(position_num=[0, 0, 0], orbital='t2g', spacegroup=sg)
            proj_set = ProjectionsSet([projection_sp3d2, projection_t2g])
            symmetrizer_up = SAWF().set_spacegroup(sg).set_D_wann_from_projections(proj_set)
            symmetrizer_dw = SAWF().set_spacegroup(sg).set_D_wann_from_projections(proj_set)
            system_soc.symmetrize2(symmetrizer_up=symmetrizer_up, symmetrizer_down=symmetrizer_dw, silent=False)
        system_soc.set_soc_axis(theta=theta, phi=phi, alpha_soc=alpha_soc)
        # system_soc.set_pointgroup(spacegroup=mg)
        path_save = os.path.join(OUTPUT_DIR, "systems", f"Fe_gpaw_soc_theta{theta_deg:.2f}_phi{phi_deg:.2f}_alpha{alpha_soc:.2f}" +
                                 f"{'_symmetrized' if do_symmetrize else ''}")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        system_soc.save_npz(path_save)
        return system_soc
    return _inner


@pytest.fixture(scope="session")
def system_Fe_gpaw_soc_z(get_system_Fe_gpaw_soc):
    return get_system_Fe_gpaw_soc(phi_deg=0, theta_deg=0)


@pytest.fixture(scope="session")
def system_Fe_gpaw_soc_z_symmetrized(get_system_Fe_gpaw_soc):
    return get_system_Fe_gpaw_soc(phi_deg=0, theta_deg=0, do_symmetrize=True)


@pytest.fixture(scope="session")
def system_Fe_gpaw_soc_angle(get_system_Fe_gpaw_soc):
    return get_system_Fe_gpaw_soc(phi_deg=33, theta_deg=49, alpha_soc=1.0)


@pytest.fixture(scope="session")
def system_Fe_gpaw_soc_angle_symmetrized(get_system_Fe_gpaw_soc):
    return get_system_Fe_gpaw_soc(phi_deg=33, theta_deg=49, alpha_soc=1.0, do_symmetrize=True)


@pytest.fixture(scope="session")
def system_Fe_gpaw_soc_111(get_system_Fe_gpaw_soc):
    return get_system_Fe_gpaw_soc(phi_deg=45, theta_deg=np.arccos(1 / np.sqrt(3)) * 180 / np.pi, alpha_soc=1.0)


@pytest.fixture(scope="session")
def system_Fe_gpaw_soc_111_symmetrized(get_system_Fe_gpaw_soc):
    return get_system_Fe_gpaw_soc(phi_deg=45, theta_deg=np.arccos(1 / np.sqrt(3)) * 180 / np.pi, alpha_soc=1.0, do_symmetrize=True)


# @pytest.fixture(scope="session")
# def _system_Fe_gpaw_soc_111_irred():
#     PATH_Fe_GPAW_irred = os.path.join(ROOT_DIR, "data", "Fe-gpaw-irred")

#     gpaw_calc = GPAW(os.path.join(PATH_Fe_GPAW_irred, "Fe-nscf-irred-222.gpw"))
#     sg = SpaceGroup.from_gpaw(gpaw_calc)
#     projection_sp3d2 = Projection(position_num=[0, 0, 0], orbital='sp3d2', spacegroup=sg)
#     projection_t2g = Projection(position_num=[0, 0, 0], orbital='t2g', spacegroup=sg)
#     proj_set = ProjectionsSet([projection_sp3d2, projection_t2g])
#     path = os.path.join(OUTPUT_DIR, "Fe-gpaw-soc-irred")
#     os.makedirs(path, exist_ok=True)
#     kwargs_w90data = dict(calculator=gpaw_calc,
#                           projections=proj_set,
#         #                     ecut_pw=300,
#         # ecut_sym=150,
#         mp_grid=(2, 2, 2),
#         read_npz_list=[],
#         spacegroup=sg,
#         # write_npz_list=[],
#         # irreducible=True,
#         # files=["amn", "mmn", "eig", "symmetrizer"],
#         unitary_params=dict(error_threshold=0.1,
#                             warning_threshold=0.01,
#                             nbands_upper_skip=8),
#     )

#     w90data_up = wberri.w90files.Wannier90data().from_gpaw(
#         spin_channel=0,
#         seedname=os.path.join(path, "Fe-irred-spin-0"),
#         **kwargs_w90data)
#     w90data_dw = wberri.w90files.Wannier90data().from_gpaw(
#         spin_channel=1,
#         seedname=os.path.join(path, "Fe-irred-spin-1"),
#         **kwargs_w90data)
#     w90data_dw.select_bands(win_min=-100,
#                          win_max=50)
#     w90data_up.select_bands(win_min=-100,
#                             win_max=50)

#     w90data_up.wannierise(
#         froz_min=-np.inf,
#         froz_max=17,
#         num_iter=500,
#         print_progress_every=10,
#         sitesym=True,
#         localise=True,
#     )

#     w90data_dw.wannierise(
#         froz_min=-np.inf,
#         froz_max=17,
#         num_iter=500,
#         print_progress_every=10,
#         sitesym=True,
#         localise=True,
#     )
#     system_up = System_w90(w90data=w90data_up, berry=True)
#     system_dw = System_w90(w90data=w90data_dw, berry=True)
#     system_soc = SystemSOC(system_up=system_up, system_down=system_dw)
#     chk_up = w90data_up.get_file('chk')
#     chk_dw = w90data_dw.get_file('chk')
#     soc = SOC.from_gpaw(gpaw_calc)
#     kptirr, weights_k = w90data_up.kptirr_system
#     system_soc.set_soc_R(soc, chk_up=chk_up, chk_down=chk_dw, kptirr=kptirr, weights_k=weights_k)
#     system_soc.symmetrize2(symmetrizer_up=w90data_up.get_file('symmetrizer'),
#                            symmetrizer_down=w90data_dw.get_file('symmetrizer'),)
#     theta = np.arccos(1 / np.sqrt(3))
#     phi = np.pi / 4
#     system_soc.set_soc_axis(theta=theta, phi=phi, alpha_soc=1.0)
#     mg = SpaceGroup.from_gpaw_magnetic(gpaw_calc, theta=theta, phi=phi)
#     system_soc.set_pointgroup(spacegroup=mg)
#     return system_soc


@pytest.fixture(scope="session")
def system_Fe_gpaw_soc_111_irred():
    PATH_Fe_GPAW_irred = os.path.join(ROOT_DIR, "data", "Fe-gpaw-irred")

    gpaw_calc = GPAW(os.path.join(PATH_Fe_GPAW_irred, "Fe-nscf-irred-222.gpw"))
    sg = SpaceGroup.from_gpaw(gpaw_calc)
    projection_sp3d2 = Projection(position_num=[0, 0, 0], orbital='sp3d2', spacegroup=sg)
    projection_t2g = Projection(position_num=[0, 0, 0], orbital='t2g', spacegroup=sg)
    proj_set = ProjectionsSet([projection_sp3d2, projection_t2g])

    # path = os.path.join(OUTPUT_DIR, "Fe-gpaw-soc-irred")
    # os.makedirs(path, exist_ok=True)
    w90data = Wannier90dataSOC.from_gpaw(
        calculator=gpaw_calc,
        projections=proj_set,
        mp_grid=(2, 2, 2),
        read_npz_list=[],
        spacegroup=sg,
    )

    w90data.select_bands(win_min=-100,
                         win_max=50)

    w90data.wannierise(
        froz_min=-np.inf,
        froz_max=17,
        num_iter=500,
        print_progress_every=10,
        sitesym=True,
        localise=True,
    )

    theta = np.arccos(1 / np.sqrt(3))
    phi = np.pi / 4

    system = SystemSOC.from_wannier90data_soc(w90data=w90data, berry=True, silent=False)
    system.set_soc_axis(theta=theta, phi=phi, alpha_soc=1.0)
    system.save_npz(os.path.join(OUTPUT_DIR, "systems", "Fe_gpaw_soc_theta54.74_phi45.00_alpha1.00_irred"))

    return system


@pytest.fixture(scope="session")
def system_GaAs_W90(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data"""

    data_dir = create_files_GaAs_W90

    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, berry=True, morb=True, spin=True,
                                      transl_inv_MV=True,
                                      read_npz=False,
                                      ws_dist_tol=-1e-5)
    system.set_pointgroup(symmetries_GaAs)
    return system


@pytest.fixture(scope="session")
def system_GaAs_W90_JM(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data with wcc phases"""

    data_dir = create_files_GaAs_W90
    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, morb=True,
                                      transl_inv_JM=True, spin=True,
                                      OSD=True, SHCryoo=True,
                                      read_npz=False,
                                      ws_dist_tol=-1e-5)
    system.set_pointgroup(symmetries_GaAs)
    return system


def get_system_GaAs_tb(symmetrize=True, berry=True):
    """Create system for GaAs using sym_tb.dat data"""
    seedname = create_files_tb(dir="GaAs_Wannier90", file=f"GaAs{'_sym' if symmetrize else ''}_tb.dat")
    system = wberri.system.System_tb(seedname, berry=berry, ws_dist_tol=-1e-5,)
    system.do_ws_dist(mp_grid=(2, 2, 2))

    system.spin_block2interlace()  # the stored system is from old VASP, with spin-block ordering
    if symmetrize:
        system.symmetrize(
            positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
            atom_name=['Ga', 'As'],
            proj=['Ga:sp3', 'As:sp3'],
            soc=True,
        )
    system.spin_interlace2block()
    system.set_pointgroup(symmetries_GaAs)
    return system


@pytest.fixture(scope="session")
def system_GaAs_tb():
    """Create system for GaAs using _tb.dat data"""
    return get_system_GaAs_tb(symmetrize=False)


@pytest.fixture(scope="session")
def system_GaAs_sym_tb():
    """Create system for GaAs using sym_tb.dat data"""
    return get_system_GaAs_tb(symmetrize=True)


@pytest.fixture(scope="session")
def system_GaAs_tb_noAA():
    """Create system for GaAs using _tb_dat data"""
    return get_system_GaAs_tb(symmetrize=False, berry=False)


def get_system_Si_W90_JM(data_dir, transl_inv=False, transl_inv_JM=False,
                         matrices=dict(OSD=True),
                         symmetrize=False,
                         double=False,
                         ):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""

    for tag in ('uHu', 'uIu'):
        if not os.path.isfile(os.path.join(data_dir, f"Si.{tag}")):
            tar = tarfile.open(os.path.join(data_dir, f"Si.{tag}.tar.gz"))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)
    # Load system
    seedname = os.path.join(data_dir, "Si")
    system = wberri.system.System_w90(seedname,
                                      transl_inv_MV=transl_inv,
                                      transl_inv_JM=transl_inv_JM,
                                      **matrices)
    if double:
        system.double_spin()
    if symmetrize:
        iRold = [tuple(R) for R in system.rvec.iRvec]
        print("Rvectors before symmetrization", system.rvec.nRvec, "\n", system.rvec.iRvec)
        print(f"wannier-diff {system.wannier_centers_red[:, None, :] - system.wannier_centers_red[None, :, :]}")
        system.symmetrize(
            positions=np.array([[-0.125, -0.125, 0.375],
                                [0.375, -0.125, -0.125],
                                [-0.125, 0.375, -0.125],
                                [-0.125, -0.125, -0.125]]),
            atom_name=['bond'] * 4,
            proj=['bond:s'],
            soc=double)
        print("Rvectors after symmetrization", system.rvec.nRvec, "\n", system.rvec.iRvec)
        iRnew = [tuple(R) for R in system.rvec.iRvec]
        for rnew in iRnew:
            if rnew not in iRold:
                print("New Rvector", rnew)
                for r in iRold:
                    if np.all(np.array(rnew) - np.array(r) % 2 == 0):
                        print("    Old Rvector", r)


    return system


@pytest.fixture(scope="session")
def system_Si_W90_JM(create_files_Si_W90):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""
    data_dir = create_files_Si_W90
    return get_system_Si_W90_JM(data_dir,
                                transl_inv_JM=True)



@pytest.fixture(scope="session")
def system_Si_W90_JM_sym(create_files_Si_W90):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""
    data_dir = create_files_Si_W90
    system = get_system_Si_W90_JM(data_dir, transl_inv_JM=True,
                                  symmetrize=True)
    return system




@pytest.fixture(scope="session")
def system_Si_W90(create_files_Si_W90):
    """Create system for Si using Wannier90 data without Jae-Mo's approach for real-space matrix elements"""
    data_dir = create_files_Si_W90
    return get_system_Si_W90_JM(data_dir, transl_inv=True)


@pytest.fixture(scope="session")
def system_Si_W90_sym(create_files_Si_W90):
    """Create system for Si using Wannier90 data with symmetrization"""
    data_dir = create_files_Si_W90
    system = get_system_Si_W90_JM(data_dir, transl_inv=True,
                                  symmetrize=True)
    # system = get_system_Si_W90_JM(data_dir, transl_inv_JM=True, matrices=dict(berry=True) )
    return system


@pytest.fixture(scope="session")
def system_Si_W90_sym_double(create_files_Si_W90):
    """Create system for Si using Wannier90 data with symmetrization"""
    data_dir = create_files_Si_W90
    system = get_system_Si_W90_JM(data_dir, transl_inv=True,
                                  symmetrize=True, double=True)
    return system


@pytest.fixture(scope="session")
def system_Si_W90_double(create_files_Si_W90):
    """Create system for Si using Wannier90 data with symmetrization"""
    data_dir = create_files_Si_W90
    system = get_system_Si_W90_JM(data_dir, transl_inv=True,
                                  symmetrize=False, double=True)
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
    system = wberri.system.System_TBmodels(model_tbmodels_Haldane)
    system.set_pointgroup(["C3z"])
    return system




@pytest.fixture(scope="session")
def system_Haldane_PythTB():
    """Create system for Haldane model using PythTB"""
    # Load system
    model_pythtb_Haldane = wb_models.Haldane_ptb(delta=0.2, hop1=-1.0, hop2=0.15)
    system = wberri.system.System_PythTB(model_pythtb_Haldane)
    system.set_pointgroup(["C3z"])
    return system


@pytest.fixture(scope="session")
def system_KaneMele_odd_PythTB():
    """Create system for Haldane model using PythTB"""
    # Load system
    model_pythtb_KaneMele_odd = wb_models.KaneMele_ptb('odd')
    system = wberri.system.System_PythTB(model_pythtb_KaneMele_odd, spin=True)
    system.set_pointgroup(["C3z", "TimeReversal"])
    return system


@pytest.fixture(scope="session")
def system_Chiral_OSD():
    """Create system for Haldane model using PythTB"""
    # Load system
    model_pythtb_Chiral_OSD = wb_models.Chiral_OSD()
    system = wberri.system.System_PythTB(model_pythtb_Chiral_OSD, spin=True)
    # system.set_pointgroup(["C3z","TimeReversal"])
    return system


# Chiral model
# A chiral system that also breaks time-reversal. It can be used to test almost any quantity.


@pytest.fixture(scope="session")
def system_Chiral_left():
    model_Chiral_left = wb_models.Chiral(
        delta=2, hop1=1, hop2=1. / 3, phi=np.pi / 10, hopz_left=0.2, hopz_right=0.0, hopz_vert=0)
    system = wberri.system.System_PythTB(model_Chiral_left)
    system.set_pointgroup(["C3z"])
    system.set_spin_eigenstates([1, -1])
    return system


@pytest.fixture(scope="session")
def system_Chiral_left_TR():
    model_Chiral_left_TR = wb_models.Chiral(
        delta=2, hop1=1, hop2=1. / 3, phi=-np.pi / 10, hopz_left=0.2, hopz_right=0.0, hopz_vert=0)
    system = wberri.system.System_PythTB(model_Chiral_left_TR)
    system.set_pointgroup(["C3z"])
    system.set_spin_eigenstates([-1, 1])
    return system


@pytest.fixture(scope="session")
def system_Chiral_right():
    model_Chiral_right = wb_models.Chiral(
        delta=2, hop1=1, hop2=1. / 3, phi=np.pi / 10, hopz_left=0.0, hopz_right=0.2, hopz_vert=0)
    system = wberri.system.System_PythTB(model_Chiral_right)
    system.set_pointgroup(["C3z"])
    system.set_spin_eigenstates([1, -1])
    return system


# Systems from FPLO code interface
@pytest.fixture(scope="session")
def system_Fe_FPLO():
    """Create system for Fe using  FPLO  data"""
    path = os.path.join(ROOT_DIR, "data", "Fe_FPLO", "+hamdata")
    system = wberri.system.System_fplo(path, morb=True, spin=True, mp_grid=2)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_CuMnAs_2d_broken():
    # CuMnAs 2D model
    # These parameters provide ~0.4eV gap between conduction and valence bands
    # and splitting into subbands is within 0.04 eV
    model_CuMnAs_2d_broken = wb_models.CuMnAs_2d(nx=0, ny=1, nz=0, hop1=1, hop2=0.08, l=0.8, J=1, dt=0.01)
    system = wberri.system.System_PythTB(model_CuMnAs_2d_broken)
    return system


# Systems from ASE+gpaw code interface

@pytest.fixture(scope="session")
def data_Te_ASE():
    """read data for Te from ASE+GPAW"""
    try:
        import gpaw
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import gpaw")
    import ase.dft.wannier

    path = os.path.join(ROOT_DIR, "data", "Te_ASE")
    calc = gpaw.GPAW(os.path.join(path, "Te.gpw"))
    wan = ase.dft.wannier.Wannier(nwannier=12, calc=calc, file=os.path.join(path, 'wannier-12.json'))
    return wan


@pytest.fixture(scope="session")
def system_Te_ASE(data_Te_ASE):
    """Create system for Te using  ASE+GPAW data"""
    wan = data_Te_ASE
    system = wberri.system.System_ASE(wan, ws_dist_tol=1e-5)
    system.set_pointgroup(symmetries_Te)
    return system


@pytest.fixture(scope="session")
def system_Te_sparse():
    """Create system for Te using symmetrized Wannier functions through a sparse interface"""
    path = os.path.join(ROOT_DIR, "data", "Te_sparse", "parameters_Te_low_interlaced.pickle")
    param = pickle.load(open(path, "rb"))
    param["wannier_centers_red"] = param["wannier_centers_reduced"]
    del param["wannier_centers_reduced"]
    system = wberri.system.SystemSparse(**param)
    system.set_pointgroup(symmetries_Te)
    return system


@pytest.fixture(scope="session")
def system_Phonons_Si():
    """Create system of phonons of Si using  QE data"""
    path = os.path.join(ROOT_DIR, "data", "Si_phonons/si")
    system = wberri.system.System_Phonon_QE(path, asr=True, ws_dist_tol=1e-5)
    system.set_pointgroup(symmetries_Si)
    return system


@pytest.fixture(scope="session")
def system_Phonons_GaAs():
    """Create system of phonons of Si using  QE data"""
    path = os.path.join(ROOT_DIR, "data", "GaAs_phonons/GaAs")
    system = wberri.system.System_Phonon_QE(path, asr=True, ws_dist_tol=1e-5)
    system.set_pointgroup(symmetries_GaAs)
    return system


def get_system_Mn3Sn_sym_tb():
    data_dir = os.path.join(ROOT_DIR, "data", "Mn3Sn_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "Mn3Sn_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "Mn3Sn_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)

    seedname = os.path.join(data_dir, "Mn3Sn_tb.dat")
    system = wberri.system.System_tb(seedname, berry=True, ws_dist_tol=-1e-5)
    system.do_ws_dist(mp_grid=(2, 2, 2))
    system.spin_block2interlace()  # the stored system is from old VASP, with spin-block ordering
    system.symmetrize(
        positions=np.array([
            [0.6666666666667, 0.8333333333333, 0],
            [0.1666666666667, 0.3333333333333, 0],
            [0.6666666666667, 0.3333333333333, 0],
            [0.3333333333333, 0.1666666666667, 0.5],
            [0.8333333333333, 0.6666666666667, 0.5],
            [0.3333333333333, 0.6666666666667, 0.5],
            [0.8333333333333, 0.1666666666667, 0.5],
            [0.1666666666667, 0.8333333333333, 0]]),
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
        reorder_back=True,
    )
    return system


@pytest.fixture(scope="session")
def system_Mn3Sn_sym_tb():
    """Create system for Mn3Sn using _tb.dat data"""
    return get_system_Mn3Sn_sym_tb()


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



def model_1d_pythtb():
    import pythtb
    lat = [[1.0]]
    orb = [[0], [0.5]]
    orb2 = [orb[0]] * 2 + [orb[1]] * 2

    model1d_1 = pythtb.tb_model(1, 1, lat, orb, nspin=2)
    model1d_2 = pythtb.tb_model(1, 1, lat, orb2, nspin=1)
    Delta = 1
    model1d_1.set_onsite([-Delta, Delta])
    model1d_2.set_onsite([-Delta] * 2 + [Delta] * 2)
    a = np.random.random(4)
    b = np.random.random(4)
    model1d_1.set_hop(a, 0, 1, [0])
    model1d_1.set_hop(b, 1, 0, [-1])

    pauli = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    for i in range(2):
        for j in range(2):
            model1d_2.set_hop(a.dot(pauli[:, i, j]), 0 + i, 2 + j, [0])
            model1d_2.set_hop(b.dot(pauli[:, i, j]), 2 + i, 0 + j, [-1])

    return model1d_1, model1d_2


@pytest.fixture(scope="session")
def system_random():
    system = wberri.system.SystemRandom(num_wann=6, nRvec=20, max_R=4,
                                        berry=True, morb=True, spin=True,
                                        SHCryoo=True, SHCqiao=True, OSD=True,
                                        ws_dist_tol=-1e-5,)
    # system.save_npz("randomsys")
    return system


@pytest.fixture(scope="session")
def system_random_load_bare():
    system = wberri.system.System_R(berry=True, morb=True, spin=True,
                                    SHCryoo=True, SHCqiao=True, OSD=True,
                                    ws_dist_tol=-1e-5)
    system.load_npz(path=os.path.join(ROOT_DIR, "data", "random"), legacy=True)
    return system


@pytest.fixture(scope="session")
def system_random_GaAs():
    return wberri.system.SystemRandom(num_wann=16, nRvec=30, max_R=4,
                                      real_lattice=np.ones(3) - np.eye(3),
                                      berry=True, spin=True, SHCryoo=True,
                                      ws_dist_tol=-1e-5,
                                      )


def get_system_random_GaAs_load_sym(sym=False, use_ws=True):
    system = wberri.system.System_R(berry=True, spin=True, SHCryoo=True, ws_dist_tol=-1e-5)
    system.load_npz(path=os.path.join(ROOT_DIR, "data", "random_GaAs"), legacy=True)
    if use_ws:
        system.do_ws_dist(mp_grid=6)
    if sym:
        system.symmetrize(
            proj=['Ga:sp3', 'As:sp3'],
            atom_name=['Ga', 'As'],
            positions=np.array([[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]]),
            soc=True,
        )
        system.set_structure(
            atom_labels=['Ga', 'As'],
            positions=np.array([[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]])
        )
        system.set_pointgroup_from_structure()
    return system


@pytest.fixture(scope="session")
def system_random_GaAs_load_bare():
    return get_system_random_GaAs_load_sym(use_ws=False, sym=False)


@pytest.fixture(scope="session")
def system_random_GaAs_load_sym():
    return get_system_random_GaAs_load_sym(use_ws=True, sym=True)


@pytest.fixture(scope="session")
def system_Te_QE():
    """Create system for Te using QE data"""
    system = wberri.system.System_R(berry=True, morb=True, spin=True,
                                    ).load_npz(os.path.join(ROOT_DIR, "data", "Te_qe", "system"), legacy=True)
    system.set_pointgroup(symmetries_Te)
    return system
