"""test auxilary functions"""

import glob
import os
import shutil
import irrep
import numpy as np
from pytest import approx

from tests.common import OUTPUT_DIR, ROOT_DIR
from wannierberri.w90files.amn import amn_from_bandstructure
from wannierberri.symmetry.projections import Projection, ProjectionsSet, get_perpendicular_coplanar_vector, read_xzaxis
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF
import wannierberri as wberri
from irrep.spacegroup import SpaceGroup
sq2 = np.sqrt(2)


def test_perpedicular_coplanar():
    for a, b, c in [([0, 0, 1], [1, 0, 0], [1, 0, 0]),
                    ([0, 0, 2], [1.23, 0, 0], [1, 0, 0]),
        ([0, 0, 1], [-1, 0, 0], [-1, 0, 0]),
        ([0, 0, -1], [1, 0, 0], [1, 0, 0]),
        ([0, 0, 1], [0, 1, 0], [0, 1, 0]),
        ([0, 0, 1], [0, -1, 0], [0, -1, 0]),
        ([0, 0, -1], [0, 1, 0], [0, 1, 0]),
        ([0, 0, -1], [0, -1, 0], [0, -1, 0]),
        ([0, 0, 1], [1, 1, 1], [1 / sq2, 1 / sq2, 0]),
        ([1, 1, -1], [-1, 1, 1], None),
        # below some 10 radom float pairs of vectors, and None for c
        ([-0.15012488, -0.2856678, 0.20514487], [0.03702015, 0.02495025, -0.37321347], None),
        ([0.38176324, -0.37329558, 0.10156163], [-0.39433382, 0.46215642, -0.46285261], None),
        ([0.08890408, 0.2226281, -0.30013063], [-0.30962228, -0.09061451, -0.4966369], None),
        ([-0.16871012, 0.11181298, -0.43920932], [0.25166256, 0.05484402, 0.25447236], None),
        ([-0.18034208, 0.12461937, 0.31991662], [-0.40384707, -0.24047836, -0.25645305], None),
        ([0.27452463, -0.00945541, -0.4089212], [0.2302086, -0.00179898, -0.41654984], None),
        ([0.04250247, -0.10983353, 0.08883795], [-0.3604843, -0.12807952, 0.22141534], None),
        ([0.35772208, 0.01895294, -0.15505931], [-0.41709066, 0.49268116, -0.14679098], None),
        ([-0.17070323, 0.16496933, -0.13690481], [0.23608187, -0.2274575, -0.34102037], None),
        ([-0.34367349, -0.38767689, 0.48765111], [-0.23867199, -0.4965295, 0.29827543], None),
        ([0.15023788, 0.42916742, 0.07188563], [0.0665572, -0.00965801, 0.35089044], None),
        ([0.26131288, -0.36010153, -0.270887], [-0.28689433, -0.20509269, -0.39165767], None)
    ]:
        c1 = get_perpendicular_coplanar_vector(a, b)
        assert np.dot(c1, a) == approx(0), f"c1 expected to be perpendicular to a but got a.c1={c1}"
        assert np.linalg.det([a, b, c1]) == approx(0), f"c1 expected to be coplanar with a,b but got det(a,b,c1)={np.linalg.det([a, b, c1])}"
        assert np.dot(c1, b) > 1e-3, f"c1 expected to project on positive b but got c1.b={np.dot(c1, b)}<=0"
        if c is not None:
            assert c1 == approx(c), f"input {a},{b}, Expected output {c} but got {c1}"


def test_readxzaxis():
    from numpy import array
    for xaxis, zaxis in [(None, None),
                        (array([0.41634019, 0.92766421, 0.75097172]), array([0.58468234, 0.95944391, 0.66306702])),
                        (array([0.64848659, 0.95517729, 0.92391961]), array([0.76522869, 0.38188162, 0.42384584])),
                        (array([0.96669811, 0.54455679, 0.77165972]), array([0.04501268, 0.722713, 0.95940886])),
                        (array([0.7864632, 0.55334491, 0.77537296]), array([0.27227999, 0.25407389, 0.96270407])),
                        (array([0.47383802, 0.25641088, 0.26546724]), array([0.42425543, 0.49411283, 0.11516953])),
                        (array([0.62030242, 0.60524421, 0.89371466]), array([0.57050537, 0.20924648, 0.80842307])),
                        (array([0.17249441, 0.74303532, 0.21318336]), array([0.59538258, 0.53710577, 0.47388691])),
                        (array([0.61783452, 0.28180076, 0.41994137]), array([0.36197028, 0.97740799, 0.8432547])),
                        (array([0.15113692, 0.16766317, 0.79286341]), array([0.40549447, 0.34405382, 0.694409])),
                        (array([0.40306808, 0.18023721, 0.34033624]), array([0.10893417, 0.32535143, 0.11423508])),
                        (None, array([0.01844338, 0.0406837, 0.71377461])),
                        (None, array([0.71933593, 0.18565351, 0.08254735])),
                        (None, array([0.84413632, 0.00395343, 0.26674889])),
                        (None, array([0.30880547, 0.38733775, 0.09063426])),
                        (None, array([0.65526834, 0.6931212, 0.85886408])),
                        (None, array([0.02300518, 0.77884875, 0.8736552])),
                        (None, array([0.05375946, 0.7943356, 0.53410123])),
                        (None, array([0.11237847, 0.97802273, 0.09950183])),
                        (None, array([0.74248219, 0.42876194, 0.74999412])),
                        (None, array([0.06783609, 0.90228458, 0.57391315])),
                        (array([0.23665075, 0.15839566, 0.98100161]), None),
                        (array([0.81354893, 0.7566733, 0.2185613]), None),
                        (array([0.38604432, 0.45733984, 0.34638067]), None),
                        (array([0.4913055, 0.31010592, 0.44975588]), None),
                        (array([0.06649211, 0.39511785, 0.86323766]), None),
                        (array([0.70314138, 0.24555993, 0.29859182]), None),
                        (array([0.49724855, 0.70778647, 0.99702118]), None),
                        (array([0.6388335, 0.64201507, 0.86172648]), None),
                        (array([0.48537197, 0.69752168, 0.73477955]), None),
                        (array([0.51093713, 0.35681036, 0.12544163]), None)
                        ]:
        if xaxis is not None and zaxis is not None:
            xaxis = get_perpendicular_coplanar_vector(zaxis, xaxis)
        basis = read_xzaxis(xaxis, zaxis)
        assert np.linalg.norm(basis, axis=1) == approx(1), f"basis vectors are not normalized : {np.linalg.norm(basis, axis=1)}"
        assert np.dot(basis[0, :], basis[1, :]) == approx(0), f"basis vectors x={basis[0]} and y={basis[1]} are not orthogonal"
        assert np.dot(basis[0, :], basis[2, :]) == approx(0), f"basis vectors x={basis[0]} and z={basis[2]} are not orthogonal"
        assert np.dot(basis[1, :], basis[2, :]) == approx(0), f"basis vectors y={basis[1]} and z={basis[2]} are not orthogonal"
        if xaxis is not None:
            xaxis = xaxis / np.linalg.norm(xaxis)
            assert xaxis == approx(basis[0, :], abs=1e-6), f"Expected xaxis {xaxis} but got {basis[0, :]} ({basis})"
        if zaxis is not None:
            zaxis = zaxis / np.linalg.norm(zaxis)
            assert zaxis == approx(basis[2, :], abs=1e-6), f"Expected zaxis {zaxis} but got {basis[2, :]}  ({basis})"




def test_projection_basis_Telike_gen():
    lattice = np.array([[1, 0, 0], [-1 / 2, np.sqrt(3) / 2, 0], [0, 0, 1.2]])
    x = 0.2
    positions = np.array([[x, 0, 0], [0, x, 1 / 3], [-x, -x, 2 / 3]])
    numbers = [1, 1, 1]
    spacegroup = SpaceGroup(cell=(lattice, positions, numbers), spinor=False)
    spacegroup.show()
    for i, s in enumerate(spacegroup.symmetries):
        print(i + 1, "\n", s.rotation_cart)
    # wyckoff_pos = WyckoffPosition("x,0,0",spacegroup=spacegroup)
    # wyckoff_pos = WyckoffPositionNumeric([x,0,0],spacegroup=spacegroup)
    # for rot,tr in zip( wyckoff_pos.rotations, wyckoff_pos.translations):
    #     print (rot.dot([0.1,0,0])+tr)
    # print (wyckoff_pos.rotations)
    # print (wyckoff_pos.rotations_cart)
    proj = Projection(orbital="p", position_num=[0.1, 0, 0.01], spacegroup=spacegroup, rotate_basis=True, xaxis=[1, 0, 0])
    print(repr(np.array(proj.basis_list)))
    reference = np.array([[[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],

        [[-0.5, 0.8660254, 0.],
        [-0.8660254, -0.5, 0.],
        [0., 0., 1.]],

        [[-0.5, -0.8660254, 0.],
        [0.8660254, -0.5, 0.],
        [0., 0., 1.]],

        [[-0.5, 0.8660254, 0.],
        [0.8660254, 0.5, 0.],
        [0., 0., -1.]],

        [[1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.]],

        [[-0.5, -0.8660254, 0.],
        [-0.8660254, 0.5, 0.],
        [0., 0., -1.]]])
    assert reference == approx(np.array(proj.basis_list), abs=1e-6)
    symmetrizer = SAWF()
    symmetrizer.set_spacegroup(spacegroup)
    symmetrizer.set_D_wann_from_projections([proj])
    print(symmetrizer.rot_orb_list[0].shape)
    assert symmetrizer.rot_orb_list[0] - np.eye(3)[None, None, :, :] == approx(0, abs=1e-6)
    assert symmetrizer.rot_orb_dagger_list[0] - np.eye(3)[None, None, :, :] == approx(0, abs=1e-6)
    print(symmetrizer.D_wann_blocks)
    print(symmetrizer.kpoints_all)


def test_projection_basis_Telike_onatom():
    lattice = np.array([[1, 0, 0], [-1 / 2, np.sqrt(3) / 2, 0], [0, 0, 1.2]])
    x = 0.2
    positions = np.array([[x, 0, 0], [0, x, 1 / 3], [-x, -x, 2 / 3]])
    numbers = [1, 1, 1]
    spacegroup = SpaceGroup(cell=(lattice, positions, numbers), spinor=False)
    spacegroup.show()
    for i, s in enumerate(spacegroup.symmetries):
        print(i + 1, "\n", s.rotation_cart)
    # wyckoff_pos = WyckoffPosition("x,0,0",spacegroup=spacegroup)
    # wyckoff_pos = WyckoffPositionNumeric([x,0,0],spacegroup=spacegroup)
    # for rot,tr in zip( wyckoff_pos.rotations, wyckoff_pos.translations):
    #     print (rot.dot([0.1,0,0])+tr)
    # print (wyckoff_pos.rotations)
    # print (wyckoff_pos.rotations_cart)
    proj = Projection(orbital="p", position_num=[0.1, 0, 0], spacegroup=spacegroup, rotate_basis=True, xaxis=[1, 0, 0])
    print(repr(np.array(proj.basis_list)))
    reference = np.array([[[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],

        [[-0.5, 0.8660254, 0.],
        [-0.8660254, -0.5, 0.],
        [0., 0., 1.]],

        [[-0.5, -0.8660254, 0.],
        [0.8660254, -0.5, 0.],
        [0., 0., 1.]]])
    assert reference == approx(np.array(proj.basis_list), abs=1e-6)
    symmetrizer = SAWF()
    symmetrizer.set_spacegroup(spacegroup)
    symmetrizer.set_D_wann_from_projections([proj])
    print(symmetrizer.rot_orb_list[0].shape)
    print(symmetrizer.rot_orb_list[0])
    assert abs(symmetrizer.rot_orb_list[0]) - np.eye(3)[None, None, :, :] == approx(0, abs=1e-6)
    # assert symmetrizer.rot_orb_list[0] - np.eye(3)[None,None,:,:] == approx(0, abs=1e-6)
    # assert symmetrizer.rot_orb_daggerlist[0] - np.eye(3)[None,None,:,:] == approx(0, abs=1e-6)
    # print (symmetrizer.D_wann_list[0])


def test_orbital_rotator():
    from wannierberri.symmetry.orbitals import OrbitalRotator
    rotator = OrbitalRotator()
    rot_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    R = rotator("s", rot_matrix)
    assert R == approx(np.eye(1), abs=1e-6)
    R = rotator("p", rot_matrix)
    assert R == approx(rot_matrix, abs=1e-6)
    rot_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R = rotator("p", rot_matrix)
    assert R == approx(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), abs=1e-6)
    R = rotator("d", rot_matrix)
    assert R == approx(np.array([[1, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, -1, 0, 0, 0],
                                 [0, 0, 0, -1, 0],
                                 [0, 0, 0, 0, -1]]), abs=1e-6)
    R = rotator("f", rot_matrix)
    assert R == approx(np.array([[1, 0, 0, 0, 0, 0, 0.],
                                 [0, 0, 1, 0, 0, 0, 0.],
                                 [0, -1, 0, 0, 0, 0, 0.],
                                 [0, 0, 0, -1, 0, 0, 0.],
                                 [0, 0, 0, 0, -1, 0, 0.],
                                 [0, 0, 0, 0, 0, 0, -1.],
                                 [0, 0, 0, 0, 0, 1, 0.]]), abs=1e-6)


def test_orbital_rotator_random():
    from wannierberri.symmetry.orbitals import OrbitalRotator
    rotator = OrbitalRotator()
    rot_matrix = np.array([[0.89395259, -0.36868402, -0.25479574],
       [0., 0.56853546, -0.82265875],
        [0.44816155, 0.73541792, 0.50824375]])
    R = rotator("s", rot_matrix)
    print(repr(R))
    assert R == approx(np.eye(1), abs=1e-6)
    R = rotator("p", rot_matrix)
    assert R @ R.T == approx(np.eye(3), abs=1e-6)
    assert R == approx(np.array([[5.08243747e-01, 4.48161549e-01, 7.35417921e-01],
       [-2.54795733e-01, 8.93952582e-01, -3.68684021e-01],
        [-8.22658757e-01, -4.24578578e-09, 5.68535471e-01]]), abs=1e-6)
    print(repr(R))
    R = rotator("d", rot_matrix)
    print(repr(R))
    assert R @ R.T == approx(np.eye(5), abs=1e-6)
    assert R == approx(np.array([[-0.11253244, 0.3945184, 0.64739134, -0.29444062, 0.57085976],
       [-0.2242977, 0.34015616, -0.3747627, 0.67177201, 0.49219875],
        [-0.72418979, -0.36868402, -0.31604339, -0.41811118, 0.25479573],
        [-0.52987466, -0.22777531, 0.5616498, 0.49322795, -0.32958603],
        [0.36305507, -0.73541792, 0.15844073, 0.20960994, 0.50824375]]), abs=1e-6)
    R = rotator("f", rot_matrix)
    print(repr(R))
    assert R @ R.T == approx(np.eye(7), abs=1e-6)
    assert R == approx(np.array([[-0.43415235, 0.08001584, 0.13130329, -0.33462222, 0.6487636,
        -0.5037016, 0.03587627],
       [-0.04549184, -0.07993056, -0.26496174, 0.60832422, 0.26275337,
        -0.05897527, 0.69192564],
        [-0.14687948, -0.46845337, -0.72727578, -0.11487639, -0.22395013,
        -0.36286242, -0.18715882],
        [-0.60218537, -0.39982008, 0.09561705, -0.26444744, -0.08331024,
        0.56758262, 0.26334572],
        [0.41260031, -0.44245394, 0.37105758, -0.33957324, -0.21152077,
        -0.34272336, 0.46776217],
        [0.39589407, -0.52959349, -0.01235803, 0.05727529, 0.61493005,
        0.30418194, -0.29782511],
        [0.31348092, 0.36286243, -0.48646235, -0.56236123, 0.17347102,
        0.28107202, 0.32874179]]), abs=1e-6)


def test_create_amn_diamond_s_bond():
    data_dir = os.path.join(ROOT_DIR, "data", "diamond")

    bandstructure = irrep.bandstructure.BandStructure(prefix=data_dir + "/di", Ecut=100,
                                                      code="espresso",
                                                    include_TR=False,
                                                      )

    projection = Projection(position_num=[[0, 0, 0], [0, 0, 1 / 2], [0, 1 / 2, 0], [1 / 2, 0, 0]], orbital='s', spacegroup=bandstructure.spacegroup)

    amn = amn_from_bandstructure(bandstructure=bandstructure, projections=ProjectionsSet([projection]),
                           normalize=True, return_object=True, spinor=False)

    tmp_dir = os.path.join(OUTPUT_DIR, "diamond+create_amn")

    # Check if the directory exists
    if os.path.exists(tmp_dir):
        # Remove the directory and all its contents
        shutil.rmtree(tmp_dir)
        print(f"Directory {tmp_dir} has been removed.")
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    data_dir = os.path.join(ROOT_DIR, "data", "diamond")
    prefix = "diamond"
    for ext in ["mmn", "eig", "win", "sawf.npz"]:
        shutil.copy(os.path.join(data_dir, prefix + "." + ext),
                    os.path.join(tmp_dir, prefix + "." + ext))
    print("prefix = ", prefix)
    symmetrizer = SAWF().from_npz(prefix + ".sawf.npz")
    # try:
    # symmetrizer.spacegroup.show()
    # except AttributeError as err:
    #     print("Error: ", err, " spacegroup could not be shown")
    w90data = wberri.w90files.Wannier90data(seedname=prefix, readfiles=["mmn", "eig", "win"])
    print("amn.shape = ", amn.data.shape)
    print("mmn.shape = ", w90data.mmn.data.shape)
    print("eig.shape = ", w90data.eig.data.shape)
    w90data.set_file("amn", amn)
    w90data.set_symmetrizer(symmetrizer=symmetrizer)
    # Now wannierise the system
    w90data.wannierise(
        froz_min=-8,
        froz_max=20,
        num_iter=100,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        num_iter_converge=20,
        print_progress_every=20,
        sitesym=True,
        localise=True
    )

    wannier_centers = w90data.chk.wannier_centers_cart
    print("wannierr_centers = ", wannier_centers)
    assert wannier_centers == approx(0.806995 * np.array([[0, 0, 0], [-1, 1, 0], [0, 1, 1], [-1, 0, 1]]), abs=1e-6)
    wannier_spreads = w90data.chk.wannier_spreads
    print("wannier_spreads = ", wannier_spreads)
    assert wannier_spreads == approx(.398647548, abs=1e-5)



def test_create_amn_diamond_p_bond():
    data_dir = os.path.join(ROOT_DIR, "data", "diamond")

    bandstructure = irrep.bandstructure.BandStructure(prefix=data_dir + "/di", Ecut=100,
                                                      code="espresso",
                                                    include_TR=False,
                                                      )
    lattice = bandstructure.lattice
    positions = np.array([[1, 1, 1], [-1, -1, -1]])
    zaxis = (positions[0] - positions[1]) @ lattice
    print("zaxis = ", zaxis)


    projection = Projection(position_num=[0, 0, 0], orbital='pz', zaxis=zaxis, spacegroup=bandstructure.spacegroup, rotate_basis=True)
    print("positions_cart = ", projection.positions @ lattice)

    amn = amn_from_bandstructure(bandstructure=bandstructure, projections=ProjectionsSet([projection]),
                           normalize=True, return_object=True, spinor=False)
    symmetrizer = SAWF().from_irrep(bandstructure)
    symmetrizer.set_D_wann_from_projections([projection])

    tmp_dir = os.path.join(OUTPUT_DIR, "diamond+create_amn")

    # Check if the directory exists
    if os.path.exists(tmp_dir):
        # Remove the directory and all its contents
        shutil.rmtree(tmp_dir)
        print(f"Directory {tmp_dir} has been removed.")
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    prefix = "diamond"
    for ext in ["mmn", "eig", "win"]:
        shutil.copy(os.path.join(data_dir, prefix + "." + ext),
                    os.path.join(tmp_dir, prefix + "." + ext))
    for f in glob.glob(data_dir + "/UNK*"):
        shutil.copy(f, tmp_dir)
    print("prefix = ", prefix)
    symmetrizer.spacegroup.show()

    w90data = wberri.w90files.Wannier90data(seedname=os.path.join(tmp_dir, prefix),
                                            readfiles=["mmn", "eig", "win", "unk"])
    w90data.set_file("amn", amn)
    w90data.set_symmetrizer(symmetrizer=symmetrizer)
    amn_symm_prec = symmetrizer.check_amn(amn, ignore_upper_bands=2)
    w90data.select_bands(win_min=20, win_max=100)
    print(f"amn is symmetric with accuracy {amn_symm_prec}")
    # Now wannierise the system
    w90data.wannierise(
        froz_min=22,
        froz_max=35,
        num_iter=20,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        print_progress_every=1,
        sitesym=True,
        localise=True
    )
    w90data.plotWF()

    wannier_centers = w90data.chk.wannier_centers_cart
    print("wannierr_centers = ", wannier_centers)
    # assert wannier_centers == approx(0.806995*np.array([[0, 0, 0], [-1,1,0], [0,1,1], [-1,0,1]]), abs=1e-6)
    wannier_spreads = w90data.chk.wannier_spreads
    print("wannier_spreads = ", wannier_spreads)
    # assert wannier_spreads == approx(.398647548, abs=1e-5)

    a = -wannier_centers[1, 0]

    wannier_centers_ab = np.array([[0, 0, 0], [-1, 0, 1], [-1, 1, 0], [0, 1, 1]]) * a
    assert wannier_centers == approx(wannier_centers_ab, abs=1e-6)
    assert wannier_spreads == approx(wannier_spreads.mean(), abs=1e-6)

    expected_spread = 1.574684543725
    expected_a = -lattice[0, 0] / 2
    assert a == approx(expected_a, abs=1e-6)
    assert wannier_spreads == approx(expected_spread, abs=1e-2)


def test_create_amn_diamond_sp3():
    data_dir = os.path.join(ROOT_DIR, "data", "diamond")

    bandstructure = irrep.bandstructure.BandStructure(prefix=data_dir + "/di", Ecut=100,
                                                      code="espresso",
                                                    include_TR=False,
                                                      )
    lattice = bandstructure.lattice
    positions = np.array([[1, 1, 1], [-1, -1, -1]]) / 8
    projection_sp3 = Projection(position_num=positions, orbital='sp3',
                            spacegroup=bandstructure.spacegroup, rotate_basis=True)
    projections = ProjectionsSet([projection_sp3])
    print(f"lattice = {lattice}")
    amn = amn_from_bandstructure(bandstructure=bandstructure, projections=projections,
                           normalize=True, return_object=True, spinor=False)
    symmetrizer = SAWF().from_irrep(bandstructure)
    symmetrizer.set_D_wann_from_projections(projections)

    tmp_dir = os.path.join(OUTPUT_DIR, "diamond+create_amn")

    # Check if the directory exists
    if os.path.exists(tmp_dir):
        # Remove the directory and all its contents
        shutil.rmtree(tmp_dir)
        print(f"Directory {tmp_dir} has been removed.")
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    prefix = "diamond"
    for ext in ["mmn", "eig", "win"]:
        shutil.copy(os.path.join(data_dir, prefix + "." + ext),
                    os.path.join(tmp_dir, prefix + "." + ext))
    symmetrizer.spacegroup.show()

    w90data = wberri.w90files.Wannier90data(seedname=prefix, readfiles=["mmn", "eig", "win"])
    w90data.set_file("amn", amn)
    w90data.set_symmetrizer(symmetrizer=symmetrizer)
    amn_symm_prec = symmetrizer.check_amn(amn, ignore_upper_bands=2)
    # w90data.apply_window(win_min=20, win_max=100)
    print(f"amn is symmetric with accuracy {amn_symm_prec}")
    # Now wannierise the system
    w90data.wannierise(
        # froz_min=-20,
        # froz_max=35,
        num_iter=20,
        conv_tol=1e-10,
        mix_ratio_z=1,
        mix_ratio_u=1,
        print_progress_every=1,
        sitesym=True,
        localise=True
    )


    wannier_centers = w90data.chk.wannier_centers_cart
    print("wannierr_centers = ", wannier_centers)
    # assert wannier_centers == approx(0.806995*np.array([[0, 0, 0], [-1,1,0], [0,1,1], [-1,0,1]]), abs=1e-6)
    wannier_spreads = w90data.chk.wannier_spreads
    print("wannier_spreads = ", wannier_spreads)

    a = -wannier_centers[0, 0]
    b = wannier_centers[0, 1]
    wannier_centers_ab = np.array([[-a, b, b],
                                   [-a, a, a],
                                   [-b, b, a],
                                   [-b, a, b],
                                   [a, -a, -a],
                                   [b, -a, -b],
                                   [a, -b, -b],
                                   [b, -b, -a]])
    assert wannier_centers == approx(wannier_centers_ab, abs=1e-6)
    assert wannier_spreads == approx(wannier_spreads.mean(), abs=1e-6)


    expected_spread = 0.432977363501
    expected_a = 0.2504700607869765
    expected_b = 0.55652500
    assert a == approx(expected_a, abs=1e-2)
    assert b == approx(expected_b, abs=1e-2)
    assert wannier_spreads == approx(expected_spread, abs=1e-2)

    # assert wannier_spreads == approx(.398647548, abs=1e-5)
