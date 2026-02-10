import os
import pickle
import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from wannierberri.calculators import tabulate as caltab
from wannierberri.calculators import TabulatorAll

from .common import OUTPUT_DIR, REF_DIR


def test_path_1(system_Haldane_PythTB):
    # Test the construction of Path class
    nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path.from_nodes(system=system_Haldane_PythTB, nodes=nodes, nk=3)
    assert path.labels == {0: '1', 2: '2'}, "path.labels is wrong"
    assert path.K_list == approx(np.array([[0, 0., 0.], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])), "path.K_list is wrong"


def test_path_2(system_Haldane_PythTB):
    # Test the construction of Path class
    nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]
    k_labels = ["A", "B", "C"]
    path = wberri.Path.from_nodes(system=system_Haldane_PythTB, nodes=nodes, labels=k_labels, nk=4)
    assert path.labels == {0: 'A', 3: 'B', 6: 'C'}, "path.labels is wrong"
    assert path.K_list == approx(
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [3, 2, 2],
            [3, 1, 1],
            [3, 0, 0],
        ]) / 6), "path.K_list is wrong"


def test_path_3(system_Haldane_PythTB):
    # Test the construction of Path class
    nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path.from_nodes(system_Haldane_PythTB, nodes=nodes, dk=1.0)
    assert path.labels == {0: '1', 3: '2', 8: '3'}, "path.labels is wrong"
    assert path.K_list[:4, :] == approx(
        np.array([[0, 0, 3], [0, 0, 2], [0, 0, 1], [0, 0, 0]]) / 6), "path.K_list is wrong"
    assert path.K_list[3:, :] == approx(
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]) / 10), "path.K_list is wrong"


def test_path_4(system_Haldane_PythTB):
    # Test where nodes is a list of numpy arrays
    nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    nodes_npy = [np.array(k) for k in nodes]
    path = wberri.Path.from_nodes(system_Haldane_PythTB, nodes=nodes, dk=1.0)
    path_npy = wberri.Path.from_nodes(system_Haldane_PythTB, nodes=nodes_npy, dk=1.0)
    assert path_npy.labels == path.labels, "path.labels is wrong"
    assert path_npy.K_list == approx(path.K_list), "path.K_list is wrong"



@pytest.mark.parametrize("lattice_type", ["cubic", "fcc", "bcc"])
def test_seek_path(lattice_type):
    # Test the construction of Path class with seekpath
    lattice = {
        'cubic': np.eye(3),
        'fcc': np.ones((3, 3)) - np.eye(3),
        'bcc': np.ones((3, 3)) / 2 - np.eye(3)
    }[lattice_type]
    lattice = lattice * 5.0  # make
    positions = np.array([[0, 0, 0]])
    numbers = np.array([1])
    path = wberri.Path.seekpath(lattice=lattice, positions=positions, numbers=numbers, dk=0.5)
    output_file = os.path.join(OUTPUT_DIR, f"seekpath_{lattice_type}.npz")
    label_ind = np.array(sorted(path.labels.keys()))
    label_val = np.array([path.labels[i] for i in label_ind])
    np.savez(output_file, K_list=path.K_list, labels_ind=label_ind, labels_val=label_val)
    ref_file = os.path.join(REF_DIR, f"seekpath_{lattice_type}.npz")
    data_ref = np.load(ref_file)
    assert path.K_list == approx(data_ref["K_list"]), "path.K_list is wrong"
    assert label_ind == approx(data_ref["labels_ind"]), "path.labels keys are wrong"
    assert label_val == approx(data_ref["labels_val"]), "path.labels values are wrong"


    print(f"lattice type: {lattice_type}")
    print("k-points", path.K_list)
    print("labels", path.labels)



def test_path_sphere(system_Haldane_PythTB):
    # Test the construction of Path class with spherical k-list
    recip_lattice = system_Haldane_PythTB.recip_lattice
    path = wberri.Path.sphere(system=system_Haldane_PythTB, r1=0.5, ntheta=3, nphi=5)
    assert path.K_list.shape == (15, 3), "path.K_list shape is wrong"
    print(repr(np.round(path.K_list, 10)))
    k_cart = path.K_list.dot(recip_lattice)
    assert (np.allclose(np.linalg.norm(k_cart, axis=1), 0.5)), "k-points are not on the sphere of radius r1"
    expected_points = np.array([[0., 0., 0.07957747],
                                [0.07957747, 0.03978874, 0.],
                                [0., 0., -0.07957747],
                                [0., 0., 0.07957747],
                                [0., 0.06891611, 0.],
                                [0., 0., -0.07957747],
                                [0., 0., 0.07957747],
                                [-0.07957747, -0.03978874, 0.],
                                [-0., -0., -0.07957747],
                                [0., 0., 0.07957747],
                                [-0., -0.06891611, 0.],
                                [-0., -0., -0.07957747],
                                [0., 0., 0.07957747],
                                [0.07957747, 0.03978874, 0.],
                                [0., 0., -0.07957747]])
    assert np.allclose(path.K_list, expected_points), "path.K_list is wrong"
    # # Check some known points on the sphere
    # for point in expected_points:
    #     assert any(np.allclose(kpt, point) for kpt in path.K_list), f"Expected point {point} not found in K_list"


def test_path_spheroid(system_Haldane_PythTB):
    # Test the construction of Path class with spheroidal k-list
    recip_lattice = system_Haldane_PythTB.recip_lattice

    r1_cart = 0.5
    r2_cart = 1.0
    path = wberri.Path.spheroid(system=system_Haldane_PythTB, r1=r1_cart, r2=r2_cart, ntheta=3, nphi=4)
    assert path.K_list.shape == (12, 3), "path.K_list shape is wrong"
    print(repr(np.round(path.K_list, 10)))
    k_cart = path.K_list.dot(recip_lattice)
    for i, kpt in enumerate(k_cart):
        theta = np.arccos(kpt[2] / np.linalg.norm(kpt))
        expected_radius = np.sqrt((r1_cart * np.sin(theta))**2 + (r2_cart * np.cos(theta))**2)
        radius = np.linalg.norm(kpt)
        assert np.isclose(radius, expected_radius), f"k-point [{i}]= {kpt} is not on the spheroid surface. Expected radius: {expected_radius}, got: {radius}"
    expected_points = np.array([[0., 0., 0.15915494],
                                [0.07957747, 0.03978874, 0.],
                                [0., 0., -0.15915494],
                                [0., 0., 0.15915494],
                                [-0.03978874, 0.03978874, 0.],
                                [-0., 0., -0.15915494],
                                [0., 0., 0.15915494],
                                [-0.03978874, -0.07957747, 0.],
                                [-0., -0., -0.15915494],
                                [0., 0., 0.15915494],
                                [0.07957747, 0.03978874, 0.],
                                [0., 0., -0.15915494]])
    assert np.allclose(path.K_list, expected_points), "path.K_list is wrong"
    # # Check some known points on the spheroid


def test_path_list(system_Haldane_PythTB):
    # Test the construction of Path class with spherical k-list
    kpoints = np.random.rand(10, 3)
    path = wberri.Path(system_Haldane_PythTB, k_list=kpoints)
    assert path.K_list.shape == (10, 3), "path.K_list shape is wrong"
    assert np.allclose(path.K_list, kpoints), "path.K_list is wrong"


def test_tabulate_path(system_Haldane_PythTB, check_run):
    param_tab = {'degen_thresh': 5e-2, }

    calculators = {}
    calculators["tabulate"] = TabulatorAll(
        {
            "Energy": caltab.Energy(),  # yes, in old implementation degen_thresh was applied to qunatities,
            # but not to energies
            "V": caltab.Velocity(**param_tab),
            "Der_berry": caltab.DerBerryCurvature(**param_tab),
            "berry": caltab.BerryCurvature(**param_tab),
            'morb': caltab.OrbitalMoment(**param_tab),
            'Der_morb': caltab.DerOrbitalMoment(**param_tab),
        },
        ibands=[0],
        mode="path")


    nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path.from_nodes(system_Haldane_PythTB, nodes=nodes, dk=1.0)
    print("k-points", path.K_list)
    # print (f"forcing internal terms: {system_Haldane_PythTB.force_internal_terms_only}")


    result = check_run(
        system=system_Haldane_PythTB,
        grid=path,
        calculators=calculators,
        fout_name="berry_Fe_W90",
        suffix="run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        use_symmetry=True,  # should have no effect, but will check the cases and give a warning
        #                parameters_K = parameters_K,
        #                frmsf_name = None,
        #                degen_thresh = degen_thresh, degen_Kramers = degen_Kramers
        skip_compare=['tabulate', ])


    tab_result = result.results["tabulate"]

    filename = "path_tab.pickle"
    fout = open(os.path.join(OUTPUT_DIR, filename), "wb")

    data = {}
    quantities = result.results["tabulate"].results.keys()
    for quant in quantities:
        result_quant = tab_result.results.get(quant)
        for comp in result_quant.get_component_list():
            data[(quant, comp)] = result_quant.get_component(comp)
    pickle.dump(data, fout)

    data_ref = pickle.load(open(os.path.join(REF_DIR, filename), "rb"))

    for quant in quantities:
        for comp in tab_result.results.get(quant).get_component_list():
            _data = data[(quant, comp)]
            try:
                _data_ref = data_ref[(quant, comp)]
            except KeyError:
                if comp == "trace":
                    _data_ref = data_ref[(quant, "xx")] + data_ref[(quant, "yy")] + data_ref[(quant, "zz")]
                elif comp is None:
                    _data_ref = data_ref[(quant, "")]
                else:
                    raise KeyError(f"Component `{comp}` is not found for quantity `{quant}`")

            assert _data.shape == _data_ref.shape
            assert _data == approx(_data_ref), (
                f"tabulation along path gave a wrong result for quantity {quant} component {comp} " +
                f"with a maximal difference {max(abs(_data - _data_ref))}")

    # only checks that the plot runs without errors, not checking the result of the plot
    tab_result.plot_path_fat(
        path,
        quantity='berry',
        component='z',
        save_file=os.path.join(OUTPUT_DIR, "Haldane-berry-VB.pdf"),
        Eshift=0,
        Emin=-2,
        Emax=2,
        iband=None,
        mode="fatband",
        fatfactor=20,
        cut_k=True,
        show_fig=False)


def test_tabulate_fail(system_Haldane_PythTB):
    nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path.from_nodes(system_Haldane_PythTB, nodes=nodes, dk=1.0)

    quantities = {
        "Energy": wberri.calculators.tabulate.Energy(),
        "berry": wberri.calculators.tabulate.BerryCurvature(),
    }

    key = "tabulate_grid"
    calculators_fail = {"tabulate_grid": wberri.calculators.TabulatorAll(quantities,
                                                                         ibands=[0],
                                                                         mode="grid"),
                        "ahc": wberri.calculators.static.AHC(Efermi=np.array([0, 1, 2]))}
    for key, val in calculators_fail.items():
        with pytest.raises(ValueError,
                           match=f"Calculation along a Path is running, but calculator `{key}` is not compatible with a Path"):
            wberri.run(system=system_Haldane_PythTB, grid=path, calculators={key: val})


@pytest.mark.parametrize("system_type", ["Haldane_PythTB", "Fe_gpaw", "SSH"])
def test_get_bandstructure(system_type,
                      system_Haldane_PythTB,
                      system_Fe_gpaw_soc_z_symmetrized,
                      system_SSH_PythTB,):
    if system_type == "Haldane_PythTB":
        system = system_Haldane_PythTB
    elif system_type == "Fe_gpaw":
        system = system_Fe_gpaw_soc_z_symmetrized
    elif system_type == "SSH":
        system = system_SSH_PythTB
    else:
        raise ValueError(f"Unknown system type {system_type}")
    path, bandstructure = system.get_bandstructure(dk=0.5, parallel=False, return_path=True)
    ref_file = os.path.join(REF_DIR, f"bandstructure_{system_type}.npz")
    output_file = os.path.join(OUTPUT_DIR, f"bandstructure_{system_type}.npz")
    energies = bandstructure.get_eigenvalues()
    np.savez(output_file, K_list=path.K_list, energies=energies)
    if system_type == "SSH":
        k = np.linspace(-0.5, 0.5, 13)
        data_ref = {'K_list': np.zeros((13, 3))}
        data_ref['K_list'][:, 0] = k
        hop1 = 1.0
        hop2 = 0.5
        delta = 0.1
        Delta = hop1 + hop2 * np.exp(-2j * np.pi * k)
        E = np.sqrt(delta**2 + abs(Delta)**2)
        data_ref['energies'] = np.array([-E, E]).T
    else:
        data_ref = np.load(ref_file)
    assert path.get_kpoints() == approx(data_ref["K_list"]), "path.K_list is wrong"
    assert energies == approx(data_ref["energies"]), "bandstructure energies are wrong"



def test_insert_closed():
    from wannierberri.grid.path_order import insert_closed_loop, insert_all_closed_loops
    loops = [[0, 1, 2], [3, 4, 5, 3], [5, 6, 4, 7, 8]]
    new_loops_ref = [[0, 1, 2], [5, 6, 4, 5, 3, 4, 7, 8]]
    new_loops, success = insert_closed_loop(loops)
    assert success, "insert_closed_loop did not find the closed loop"
    assert new_loops == new_loops_ref, f"insert_all_closed_loops did not insert the closed loop correctly. Expected {new_loops_ref}, got {new_loops}"
    new_loops = insert_all_closed_loops(loops)
    assert new_loops == new_loops_ref, f"insert_all_closed_loops did not insert the closed loop correctly. Expected {new_loops_ref}, got {new_loops}"


def test_connect_segments():
    from wannierberri.grid.path_order import connect_segments
    segments = [(0, 1), (2, 3), (1, 2), (4, 5), (3, 5), (3, 1)]
    new_segments_ref = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 4), (3, 1)]
    new_segments = connect_segments(segments)
    assert new_segments == new_segments_ref, f"connect_segments did not connect the segments correctly. Expected {new_segments_ref}, got {new_segments}"


def test_connect_segments_a():
    from wannierberri.grid.path_order import connect_segments
    segments = [(0, 'a'), (2, 3), ('a', 2), (4, 'gamma'), (3, 'gamma'), (3, 'a')]
    new_segments_ref = [(0, 'a'), ('a', 2), (2, 3), (3, 'gamma'), ('gamma', 4), (3, 'a')]
    new_segments = connect_segments(segments)
    assert new_segments == new_segments_ref, f"connect_segments did not connect the segments correctly. Expected {new_segments_ref}, got {new_segments}"


def test_flatten_path():
    from wannierberri.grid.path_order import flatten_path
    point_coords = {0: [0, 0, 0], 1: [0.5, 0.5, 0.5], 2: [0.5, 0, 0], 3: [0.5, 0.5, 0], 4: [0.5, 0, 0.5], 5: [0, 0.5, 0.5], 6: [0, 0.5, 0]}
    path_seek = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (2, 0), (1, 3), (3, 1), (1, 4), (4, 1), (1, 4), (4, 3), (2, 3), (3, 2), (1, 5), (5, 2)]
    new_point_coords_ref = {0: [0, 0, 0], 3: [0.5, 0.5, 0.0], 2: [0.5, 0, 0], 6: [0, 0.5, 0]}
    new_path_seek_ref = [(3, 2), (2, 0), (0, 3), (3, 6), (6, 2)]
    new_point_coords, new_path_seek = flatten_path(nodes=point_coords, segments=path_seek, direction=2)
    for k, v in new_point_coords_ref.items():
        assert k in new_point_coords, f"Node {k} is missing in the flattened nodes"
        assert np.allclose(new_point_coords[k], v), f"Node {k} has wrong coordinates. Expected {v}, got {new_point_coords[k]}"
    for k, v in new_point_coords.items():
        assert k in new_point_coords_ref, f"Node {k} is extra in the flattened nodes"
    assert len(new_path_seek) == len(new_path_seek_ref), f"Flattened path seek has wrong number of segments. Expected {len(new_path_seek_ref)}, got {len(new_path_seek)}"
    for seg1, seg2 in zip(new_path_seek, new_path_seek_ref[1:]):
        assert seg1[1] == seg2[0], f"Segments {seg1} and {seg2} are not connected in the flattened path seek"
    assert new_path_seek == new_path_seek_ref, f"flatten_path did not return the correct path seek. Expected {new_path_seek_ref}, got {new_path_seek}"
