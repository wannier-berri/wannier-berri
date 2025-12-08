import os
import numpy as np
from pytest import approx
from wannierberri.w90files.w90data import BKVectors
from tests.common import ROOT_DIR


def test_bkvec_nnkp():
    path = os.path.join(ROOT_DIR, "data", "Fe_Wannier90", "Fe.nnkp")
    bkvec = BKVectors.from_nnkp(path, kmesh_tol=1e-5, bk_complete_tol=1e-5)
    expected_bk_latt = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [0, 1, 1],
                                [1, 0, 0],
                                [1, 1, 0],
                                [0, 0, -1],
                                [1, 0, -1],
                                [0, -1, 0],
                                [0, -1, -1],
                                [-1, 0, 0],
                                [-1, 0, 1],
                                [-1, -1, 0]])
    assert np.array_equal(bkvec.bk_latt, expected_bk_latt)
    assert np.allclose(bkvec.wk[0], 0.23472230182715031)


def check_bkvec_kpoints(mp_grid, recip_lattice):
    kpt_red = np.array([[i / mp_grid[0], j / mp_grid[1], k / mp_grid[2]]
                        for i in range(mp_grid[0])
                        for j in range(mp_grid[1])
                        for k in range(mp_grid[2])])

    bkvec = BKVectors.from_kpoints(recip_lattice=recip_lattice,
                                   mp_grid=mp_grid,
                                   kpoints_red=kpt_red)
    print(f"bkvec.bk_latt:\n{repr(bkvec.bk_latt)}")
    print(f"bkvec.wk:\n{repr(bkvec.wk)}")
    return bkvec


def test_bkvec_kpoints_cubic():
    # Simple cubic
    mp_grid = [4, 4, 4]
    recip_lattice = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    check_bkvec_kpoints(mp_grid, recip_lattice)


def test_bkvec_kpoints_bcc():
    # Face-centered cubic
    mp_grid = [4, 4, 4]
    recip_lattice = np.array([[0, 0.5, 0.5],
                              [0.5, 0, 0.5],
                              [0.5, 0.5, 0]])
    bkvec = check_bkvec_kpoints(mp_grid, recip_lattice)
    assert bkvec.wk == approx(8)

    # Body-centered cubic


def test_bkvec_kpoints_fcc():
    mp_grid = [4, 4, 4]
    recip_lattice = np.array([[1, 1, -1],
                              [-1, 1, 1],
                              [1, -1, 1]])
    bkvec = check_bkvec_kpoints(mp_grid, recip_lattice)
    assert bkvec.wk == approx(2)


def test_bkvec_kpoints_random():
    np_grid = [3, 6, 7]
    recip_lattice = np.random.rand(3, 3)
    check_bkvec_kpoints(np_grid, recip_lattice)


def test_bkvec_monoclinic():
    mp_grid = [5, 5, 7]
    a = (3.3296, 0, 0)
    b = (0, 4.6065, 0)
    c = (-2.7659, 0, 4.6475)
    recip_lattice = np.array([a, b, c])
    bkvec = check_bkvec_kpoints(mp_grid, recip_lattice)
    assert bkvec.wk == approx(np.array([0.85383576, 0.85383576, 0.67304432, 0.67304432, 0.461254,
       0.461254, 0.5890713, 0.5890713]))
