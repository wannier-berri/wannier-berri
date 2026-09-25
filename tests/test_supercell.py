"""Tests for supercells of a System_R: graphene sqrt3 x sqrt3 and 2x2."""

import itertools

import numpy as np
import pytest

from wannierberri.evaluate_k import evaluate_k
from wannierberri.fourier.rvectors import Rvectors
from wannierberri.system.system_R import System_R
from wannierberri.system.system_supercell import add_proximity_potential

SQRT3 = [[2, -1, 0], [1, 1, 0], [0, 0, 1]]
TWO_BY_TWO = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]


def graphene_nn(t=1.0):
    """Nearest-neighbour graphene (a=1): E(k) = +-t|1 + exp(-2 pi i k_1) + exp(-2 pi i k_2)|"""
    real_lattice = np.array([[1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0, 0, 10]])
    iRvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    Ham_R = np.zeros((len(iRvec), 2, 2), dtype=complex)
    Ham_R[[0, 2, 4], 0, 1] = -t  # <A,0|H|B,R> for R = 0, -a1, -a2
    Ham_R[[0, 1, 3], 1, 0] = -t  # hermitian conjugate
    system = System_R(periodic=(True, True, False), silent=True)
    system.real_lattice = real_lattice
    system.num_wann = 2
    system.wannier_centers_cart = np.array([[0, 0, 0], [1 / 3, 1 / 3, 0]]) @ real_lattice
    system.rvec = Rvectors(lattice=real_lattice, iRvec=iRvec, shifts_left_red=system.wannier_centers_red)
    system.set_R_mat("Ham", Ham_R)
    return system


def Ham_k(system, k_red):
    Ham = np.einsum("R,Rij->ij", np.exp(2j * np.pi * system.rvec.iRvec @ k_red), system.get_R_mat("Ham"))
    np.testing.assert_allclose(Ham, Ham.T.conj(), atol=1e-12)
    return Ham


@pytest.mark.parametrize("supercell_matrix, length_sc, E_gamma", [
    (SQRT3, np.sqrt(3), [-3, 0, 0, 0, 0, 3]),  # the Dirac points K and K' fold onto Gamma
    (TWO_BY_TWO, 2, [-3, -1, -1, -1, 1, 1, 1, 3]),  # the three M points fold onto Gamma
], ids=["sqrt3xsqrt3", "2x2"])
def test_graphene_supercell(supercell_matrix, length_sc, E_gamma):
    sc = graphene_nn().make_supercell(supercell_matrix)
    assert sc.num_wann == len(E_gamma)
    np.testing.assert_allclose(np.linalg.norm(sc.real_lattice[:2], axis=1), length_sc)

    # every folded hopping is still a nearest-neighbour bond, 3 bonds per orbital
    iR, i, j = np.nonzero(abs(sc.get_R_mat("Ham")) > 1e-12)
    bonds = sc.rvec.cRvec[iR] + sc.wannier_centers_cart[j] - sc.wannier_centers_cart[i]
    assert len(bonds) == 3 * sc.num_wann
    np.testing.assert_allclose(np.linalg.norm(bonds, axis=1), 1 / np.sqrt(3))

    np.testing.assert_allclose(np.linalg.eigvalsh(Ham_k(sc, np.zeros(3))), E_gamma, atol=1e-12)


@pytest.mark.parametrize("supercell_matrix", [SQRT3, TWO_BY_TWO], ids=["sqrt3xsqrt3", "2x2"])
def test_graphene_proximity_potential(supercell_matrix):
    M = np.array(supercell_matrix)
    num_cells = round(abs(np.linalg.det(M)))
    # <w_a,L+r1|V|w_b,L+r2> = v for all supercell lattice vectors L
    elements = [(0, 0, (0, 0, 0), (0, 0, 0), 0.3),
                (0, 1, (0, 0, 0), (0, 0, 0), 0.2 + 0.1j), (1, 0, (0, 0, 0), (0, 0, 0), 0.2 - 0.1j),
                (0, 1, (0, 0, 0), (-1, 0, 0), -0.15 + 0.05j), (1, 0, (-1, 0, 0), (0, 0, 0), -0.15 - 0.05j)]

    def VV(k1, k2):
        """<psi_k1|V|psi_k2>, nonzero only if k1 - k2 is a reciprocal lattice vector of the supercell"""
        V = np.zeros((2, 2), dtype=complex)
        dk = M @ (k1 - k2)
        if np.allclose(dk, np.round(dk)):
            for a, b, r1, r2, v in elements:
                V[a, b] += v * np.exp(2j * np.pi * (k2 @ r2 - k1 @ r1)) / num_cells
        return V

    prim = graphene_nn()
    sc = prim.make_supercell(M)
    mp_grid = (6, 6, 1)
    kpoints = np.array(list(np.ndindex(*mp_grid))) / mp_grid
    add_proximity_potential(sc, [[VV(k1, k2) for k2 in kpoints] for k1 in kpoints], mp_grid, M)

    # at a generic k the supercell spectrum is the one of <psi_k|H0 + V|psi_k'> over all k folding onto it
    k_sc = np.array([0.123, 0.345, 0])
    sc_to_prim = sc.recip_lattice @ np.linalg.inv(prim.recip_lattice)
    k_folded = {}
    for g in itertools.product(range(num_cells), range(num_cells), [0]):
        k = (k_sc + np.array(g)) @ sc_to_prim
        k_folded[tuple(np.round(k % 1, 8) % 1)] = k
    k_folded = list(k_folded.values())
    assert len(k_folded) == num_cells
    Ham_unfolded = np.block([[VV(k1, k2) + (Ham_k(prim, k1) if n1 == n2 else 0) for n2, k2 in enumerate(k_folded)]
                             for n1, k1 in enumerate(k_folded)])
    np.testing.assert_allclose(np.linalg.eigvalsh(Ham_k(sc, k_sc)), np.linalg.eigvalsh(Ham_unfolded), atol=1e-10)


@pytest.fixture(scope="module")
def graphene_gpaw(tmp_path_factory):
    """pz Wannier functions of graphene from GPAW: LDA, PW(400), Gamma-centred 6x6x1 (a1, a2 at 120°)."""
    pytest.importorskip("gpaw")
    from ase import Atoms
    from gpaw import GPAW, PW
    from irrep.spacegroup import SpaceGroup
    from wannierberri.symmetry.projections import Projection, ProjectionsSet
    from wannierberri.w90files import WannierData

    a = 2.46
    atoms = Atoms("C2", cell=a * np.array([[np.sqrt(3) / 2, 1 / 2, 0], [-np.sqrt(3) / 2, 1 / 2, 0], [0, 0, 10]]),
                  scaled_positions=[[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0]], pbc=True)
    atoms.calc = GPAW(mode=PW(400), xc="LDA", kpts={"size": (6, 6, 1), "gamma": True}, symmetry="off",
                      nbands=12, convergence={"bands": 8}, txt=None)
    atoms.get_potential_energy()
    calc = atoms.calc
    E_F = calc.get_fermi_level()
    iK = next(i for i, k in enumerate(calc.get_ibz_k_points()) if np.allclose(k, [1 / 3, 1 / 3, 0]))
    E_K = calc.get_eigenvalues(kpt=iK)
    E_dirac_dft = np.sort(E_K[np.argsort(abs(E_K - E_F))[:2]])

    sg = SpaceGroup.from_gpaw(calc)
    projections = ProjectionsSet(projections=[Projection(position_num=sg.positions, orbital="pz", spacegroup=sg)])
    wandata = WannierData.from_gpaw(calculator=calc, projections=projections, irreducible=False,
                                    files=["amn", "mmn", "eig"],
                                    seedname=str(tmp_path_factory.mktemp("graphene_gpaw") / "graphene"))
    wandata.wannierise(froz_min=E_F - 2, froz_max=E_F + 1, num_iter=100, conv_tol=1e-10, sitesym=False,
                       parallel=False)
    system = System_R.from_wannierdata(wandata=wandata, berry=True, periodic=(True, True, False))
    return system, E_dirac_dft


@pytest.mark.parametrize("supercell_matrix, length_sc, n_dirac_gamma", [
    ([[2, 1, 0], [-1, 1, 0], [0, 0, 1]], np.sqrt(3), 4),  # K and K' fold onto Gamma
    ([[2, 0, 0], [0, 2, 0], [0, 0, 1]], 2, 0),
], ids=["sqrt3xsqrt3", "2x2"])
def test_graphene_gpaw_supercell(graphene_gpaw, supercell_matrix, length_sc, n_dirac_gamma):
    prim, E_dirac_dft = graphene_gpaw
    E_dirac = evaluate_k(prim, k=[1 / 3, 1 / 3, 0], quantities=["energy"])
    np.testing.assert_allclose(E_dirac, E_dirac_dft, atol=1e-4)  # the Wannier model has the DFT Dirac point

    sc = prim.make_supercell(supercell_matrix)
    np.testing.assert_allclose(np.linalg.norm(sc.real_lattice[:2], axis=1), length_sc * 2.46)
    E_gamma = evaluate_k(sc, k=[0, 0, 0], quantities=["energy"])
    assert np.sum(abs(E_gamma - E_dirac.mean()) < 1e-6) == n_dirac_gamma

    # at a generic k the supercell bands and velocities are those of the primitive cell at all k folding onto it
    quantities = ["energy", "band_gradients"]
    k_sc = np.array([0.123, 0.345, 0])
    sc_to_prim = sc.recip_lattice @ np.linalg.inv(prim.recip_lattice)
    ncell = sc.num_wann // prim.num_wann
    k_folded = {}
    for g in itertools.product(range(ncell), range(ncell), [0]):
        k = (k_sc + np.array(g)) @ sc_to_prim
        k_folded[tuple(np.round(k % 1, 8) % 1)] = k
    assert len(k_folded) == ncell
    res_prim = [evaluate_k(prim, k=k, quantities=quantities) for k in k_folded.values()]
    res_sc = evaluate_k(sc, k=k_sc, quantities=quantities)
    order_prim = np.argsort(np.concatenate([r["energy"] for r in res_prim]))
    order_sc = np.argsort(res_sc["energy"])
    for q in quantities:
        np.testing.assert_allclose(res_sc[q][order_sc], np.concatenate([r[q] for r in res_prim])[order_prim],
                                   atol=1e-8)
