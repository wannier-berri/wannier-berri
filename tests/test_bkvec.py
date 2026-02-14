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
    # We pre-generate random lattices and grids to make the test deterministic. 
    # There is always a risk that the random test will fail because the basis is linear dependent, 
    # or something like that
    random_lattices = np.array([[[0.79567055, 0.71833666, 0.00406184],
        [0.34090422, 0.80504342, 0.49112307],
        [0.84941711, 0.5742208 , 0.43935193]],

       [[0.9968882 , 0.16819596, 0.16635478],
        [0.992871  , 0.23949889, 0.68057601],
        [0.97787386, 0.48877679, 0.67592279]],

       [[0.11790659, 0.2113633 , 0.6568254 ],
        [0.83041598, 0.29768511, 0.5454481 ],
        [0.71869594, 0.94695072, 0.87228108]],

       [[0.17878295, 0.89191524, 0.73837778],
        [0.87533689, 0.33407418, 0.36220004],
        [0.54075786, 0.80295494, 0.55640685]],

       [[0.50260904, 0.96352272, 0.48958911],
        [0.13071509, 0.50296406, 0.95918434],
        [0.97547228, 0.46145055, 0.89608503]],

       [[0.38041909, 0.78736291, 0.67825379],
        [0.90666352, 0.84688487, 0.71728552],
        [0.01017912, 0.95500468, 0.97072365]],

       [[0.04883063, 0.88631937, 0.90663672],
        [0.010744  , 0.1863595 , 0.42406487],
        [0.65594051, 0.34627157, 0.84554079]],

       [[0.23516183, 0.43059436, 0.20277219],
        [0.69143411, 0.76788859, 0.56028781],
        [0.28795695, 0.38458305, 0.36087306]],

       [[0.12243931, 0.95013183, 0.23806296],
        [0.35779445, 0.95428047, 0.16880009],
        [0.29210909, 0.19987705, 0.545579  ]],

       [[0.27712157, 0.95739719, 0.14922617],
        [0.61072997, 0.9152314 , 0.49923091],
        [0.34622259, 0.82327733, 0.45853866]]])
    random_grids = np.array([[ 9,  8,  2],
       [ 1, 10,  9],
       [10,  4,  6],
       [10, 10,  7],
       [ 5,  9, 10],
       [ 4,  7,  1],
       [ 9,  7, 10],
       [ 2,  3,  5],
       [ 2,  6,  8],
       [ 3,  5,  1]])
    
    for recip_lattice, mp_grid in zip(random_lattices, random_grids):
        check_bkvec_kpoints(mp_grid, recip_lattice)
    

def test_bkvec_monoclinic():
    mp_grid = [5, 5, 7]
    a = (3.3296, 0, 0)
    b = (0, 4.6065, 0)
    c = (-2.7659, 0, 4.6475)
    recip_lattice = np.array([a, b, c])
    bkvec = check_bkvec_kpoints(mp_grid, recip_lattice)
    assert bkvec.wk == approx(np.array([0.85383576, 0.85383576, 0.67304432, 0.67304432, 0.461254,
       0.461254, 0.5890713, 0.5890713]))
