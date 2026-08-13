from wannierberri.w90files.w90data_soc import WannierDataSOC
from wannierberri.system.system_soc import SystemSOC
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.grid import Path
from gpaw import GPAW
from irrep.spacegroup import SpaceGroup
import os
import numpy as np
from .common import DATA_DIR, OUTPUT_DIR, REF_DIR

from wannierberri.symmetry.projections import Projection, ProjectionsSet


def test_wannierise(check_system):
    calc = GPAW(os.path.join(DATA_DIR, "MnTe_gpaw", "MnTe-nscf-irred-332.gpw"), txt=None)
    sg = SpaceGroup.from_gpaw(calc, include_TR=True)


    positions_Mn = [[0, 0, 0],
                    [0, 0, 1 / 2]]

    positions_Te = [[1 / 3, 2 / 3, 1 / 4],
                    [2 / 3, 1 / 3, 3 / 4]]

    proj_Mn1_d = Projection(position_num=positions_Mn[0], orbital='d', spacegroup=sg)

    proj_Te_sp2 = Projection(position_num=positions_Te, orbital='sp2', spacegroup=sg, xaxis=[0, -1, 0], rotate_basis=True)
    proj_Te_pz = Projection(position_num=positions_Te, orbital='pz', spacegroup=sg)

    proj_set_up = ProjectionsSet([proj_Mn1_d, proj_Te_sp2, proj_Te_pz])

    wandata = WannierDataSOC.from_gpaw(calc,
                                       altermagnetic=True,
                                       IBstart=8,
                                       projections=proj_set_up,
                                       mp_grid=[3, 3, 2],
                                       files=["mmn", "amn", "eig", "soc"],
                                       seedname=os.path.join(OUTPUT_DIR, "MnTe", "wannier_soc_altermagnetic")
                                         )
    wandata.to_npz()
    wandata.wannierise(
        froz_min=-10,
        froz_max=7,
        outer_min=-10,
        outer_max=np.inf,
        num_iter=20,
        print_progress_every=50,
        sitesym=True,
        localise=True,
        savechk=True
    )
    theta = 90
    phi = 90
    # wandata = Wannier90dataSOC.from_npz("wannier_soc_altermagnetic")
    print(f"wandata has files: {wandata._files}, {wandata.nspin}")
    system_soc = SystemSOC.from_wannierdata(wandata=wandata, berry=True, silent=False)
    system_soc.set_soc_axis(theta=theta, phi=phi, alpha_soc=1.0, units="degrees")
    system_soc.save_npz(os.path.join(OUTPUT_DIR, "systems", "MnTe_soc_altermagnetic"))
    check_system(
        system_soc, "MnTe_soc_altermagnetic",
        matrices=['overlap_up_down', 'dV_soc'],
        properties=['num_wann', 'real_lattice', 'periodic', 'is_phonon', 'wannier_centers_cart', 'iRvec'],
        precision_matrix_elements=1e-4,
        precision_wcc=1e-6,
    )
    check_system(
        system_soc.system_up, "MnTe_soc_altermagnetic/system_up",
        matrices=['Ham', 'AA', 'dV_soc'],
        properties=['num_wann', 'real_lattice', 'periodic', 'is_phonon', 'wannier_centers_cart', 'iRvec', ],
        precision_matrix_elements=1e-4,
        precision_wcc=1e-6,
    )





def test_MnTe_bandstructure_altermagnetic():
    wandata = WannierDataSOC.from_npz(os.path.join(DATA_DIR, "MnTe_gpaw", "wannier_soc_altermagnetic"),
                                      files=["mmn", "amn", "eig", "soc", "chk", "symmetrizer"])
    system_soc = SystemSOC.from_wannierdata(wandata=wandata, berry=True, silent=False)
    lattice = system_soc.real_lattice
    c = lattice[2, 2]

    kz = 0.35 / (2 * np.pi / c)
    path = Path.from_nodes(real_lattice=lattice,
                nodes=[
                    [2 / 3, -1 / 3, 0],
                    [0, 0, 0],
                    [-2 / 3, 1 / 3, 0],
                    None,
                    [-0.5, 0, kz],
                    [0, 0, kz],
                    [0.5, 0, kz],
                ],
        # labels=[r"${\rm K}\leftarrow$",
        #             r"$\Gamma$",
        #             r"$\rightarrow{\rm K}$",
        #             r"$\overline{\rm M}\leftarrow$",
        #             r"$\overline{\Gamma}$",
        #             r"$\rightarrow\overline{\rm M}$"],
        dk=0.05)

    from wannierberri.calculators.tabulate import Spin, BerryCurvature
    tab_spin = Spin()
    tab_berry_int = BerryCurvature(kwargs_formula={"external_terms": False})
    tab_berry_ext = BerryCurvature(kwargs_formula={"internal_terms": False})
    bands_wannier_soc = evaluate_k_path(system_soc, path=path, tabulators=dict(spin=tab_spin, berry_int=tab_berry_int, berry_ext=tab_berry_ext))
    bands_wannier_up = evaluate_k_path(system_soc.system_up, path=path, tabulators=dict(berry_int=tab_berry_int, berry_ext=tab_berry_ext))
    bands_wannier_dw = evaluate_k_path(system_soc.system_down, path=path, tabulators=dict(berry_int=tab_berry_int, berry_ext=tab_berry_ext))

    fname = "bandstructure_MnTe_altermagnetic.npz"
    output_file = os.path.join(OUTPUT_DIR, fname)
    ref_file = os.path.join(REF_DIR, fname)
    energies_soc = bands_wannier_soc.get_eigenvalues()
    energies_up = bands_wannier_up.get_eigenvalues()
    energies_dw = bands_wannier_dw.get_eigenvalues()
    spins = bands_wannier_soc.get_data("spin")
    np.savez(output_file, K_list=path.K_list,
             energiesup=energies_up, energiesdw=energies_dw, energiessoc=energies_soc,
             spins=spins, berry_soc_int=bands_wannier_soc.get_data("berry_int"), berry_soc_ext=bands_wannier_soc.get_data("berry_ext"),
             berry_up_int=bands_wannier_up.get_data("berry_int"), berry_up_ext=bands_wannier_up.get_data("berry_ext"),
             berry_dw_int=bands_wannier_dw.get_data("berry_int"), berry_dw_ext=bands_wannier_dw.get_data("berry_ext"))
    print(f"Saved bandstructure data to {output_file}")

    ref_data = np.load(ref_file)
    for key in ref_data.keys():
        diff = np.abs(ref_data[key] - np.load(output_file)[key])
        max_diff = np.max(diff)
        assert max_diff < 1e-4, f"Mismatch in {key} between reference and output data, max difference: {max_diff}"
