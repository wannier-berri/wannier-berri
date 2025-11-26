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
    path = wberri.Path(system_Haldane_PythTB, nodes=nodes, nk=3)
    assert path.labels == {0: '1', 2: '2'}, "path.labels is wrong"
    assert path.K_list == approx(np.array([[0, 0., 0.], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])), "path.K_list is wrong"


def test_path_2(system_Haldane_PythTB):
    # Test the construction of Path class
    nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]
    k_labels = ["A", "B", "C"]
    path = wberri.Path(system_Haldane_PythTB, nodes=nodes, labels=k_labels, nk=4)
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
    path = wberri.Path(system_Haldane_PythTB, nodes=nodes, dk=1.0)
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
    path = wberri.Path(system_Haldane_PythTB, nodes=nodes, dk=1.0)
    path_npy = wberri.Path(system_Haldane_PythTB, nodes=nodes_npy, dk=1.0)
    assert path_npy.labels == path.labels, "path.labels is wrong"
    assert path_npy.K_list == approx(path.K_list), "path.K_list is wrong"


def test_path_sphere(system_Haldane_PythTB):
    # Test the construction of Path class with spherical k-list
    recip_lattice = system_Haldane_PythTB.recip_lattice
    path = wberri.Path(system_Haldane_PythTB, k_list='sphere', r1=0.5, ntheta=3, nphi=5)
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
    path = wberri.Path(system_Haldane_PythTB, k_list='spheroid', r1=r1_cart, r2=r2_cart, ntheta=3, nphi=4)
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
    path = wberri.Path(system_Haldane_PythTB, nodes=nodes, dk=1.0)
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
    path = wberri.Path(system_Haldane_PythTB, nodes=nodes, dk=1.0)

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
