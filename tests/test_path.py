import os
import pickle
import numpy as np
from pytest import approx

import wannierberri as wberri

from common import OUTPUT_DIR, REF_DIR


def test_path_1(system_Haldane_PythTB):
    # Test the construction of Path class
    k_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, nk=3)
    assert path.labels == {0: '1', 2: '2'}, "path.labels is wrong"
    assert path.K_list == approx(np.array([[0, 0., 0.], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])), "path.K_list is wrong"


def test_path_2(system_Haldane_PythTB):
    # Test the construction of Path class
    k_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]
    k_labels = ["A", "B", "C"]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, labels=k_labels, nk=4)
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
    k_nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, dk=1.0)
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
    # Test where k_nodes is a list of numpy arrays
    k_nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    k_nodes_npy = [np.array(k) for k in k_nodes]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, dk=1.0)
    path_npy = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes_npy, dk=1.0)
    assert path_npy.labels == path.labels, "path.labels is wrong"
    assert path_npy.K_list == approx(path.K_list), "path.K_list is wrong"


def test_tabulate_path(system_Haldane_PythTB):


def test_Fe(check_run, system_Fe_W90, compare_any_result, compare_fermisurfer):
    param = {'Efermi': Efermi_Fe}
    param_tab = {'degen_thresh': 5e-2}
    calculators = {k: v(**param) for k, v in calculators_Fe.items()}
    quantities = ['V', 'berry', 'Der_berry', 'morb', 'Der_morb']
    calculators["tabulate"] = calc.TabulatorAll(
        {
            "Energy": calc.tabulate.Energy(),  # yes, in old implementation degen_thresh was applied to qunatities,
            # but not to energies
            "V": calc.tabulate.Velocity(**param_tab),
            "Der_berry": calc.tabulate.DerBerryCurvature(**param_tab),
            "berry": calc.tabulate.BerryCurvature(**param_tab),
            'morb': calc.tabulate.OrbitalMoment(**param_tab),
            'Der_morb': calc.tabulate.DerOrbitalMoment(**param_tab),
        },
        ibands=[5, 6, 7, 8])

    result = check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="run",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"Morb": -1e-6},
        skip_compare=['tabulate', 'opt_conductivity', 'opt_SHCqiao', 'opt_SHCryoo'])

    for quant in 'opt_conductivity', 'opt_SHCryoo', 'opt_SHCryoo':
        compare_any_result(
            "berry_Fe_W90",
            quant + "-run",
            0,
            fout_name_ref="kubo_Fe_W90",
            suffix_ref=quant,
            precision=-1e-8,
            result_type=EnergyResult)

    extra_precision = {'Morb': 1e-6, 'Der_berry':5e-8}
    for quant in result.results.get("tabulate").results.keys(): # ["Energy", "berry","Der_berry","spin","morb"]:
        for comp in result.results.get("tabulate").results.get(quant).get_component_list():
            _quant = "E" if quant == "Energy" else quant
            _comp = "-" + comp if comp != "" else ""
            #            data=result.results.get(quant).data
            #            assert data.shape[0] == len(Efermi)
            #            assert np.all( np.array(data.shape[1:]) == 3)
            prec = extra_precision[quant] if quant in extra_precision else 2e-8
            #            comparer(frmsf_name, quant+_comp+suffix,  suffix_ref=compare_quant(quant)+_comp+suffix_ref ,precision=prec )
            compare_fermisurfer(
                fout_name="berry_Fe_W90-tabulate",
                suffix=_quant + _comp + "-run",
                suffix_ref=_quant + _comp,
                fout_name_ref="tabulate_Fe_W90",
                precision=prec)



    quantities = ['V', 'berry', 'Der_berry', 'morb', 'Der_morb']

    k_nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, dk=1.0)

    tab_result = wberri.tabulate(
        system=system_Haldane_PythTB,
        grid=path,
        quantities=quantities,
        user_quantities={},
        parameters={"external_terms": False},
        #                    specific_parameters = specific_parameters,
        ibands=[0],
        use_irred_kpt=True,
        symmetrize=True,  # should have no effect, but will check the cases and give a warning
        #                parameters_K = parameters_K,
        #                frmsf_name = None,
        #                degen_thresh = degen_thresh, degen_Kramers = degen_Kramers
    )

    filename = "path_tab.pickle"
    fout = open(os.path.join(OUTPUT_DIR, filename), "wb")

    data = {}
    for quant in ["Energy"] + quantities:
        result_quant = tab_result.results.get(quant)
        for comp in result_quant.get_component_list():
            data[(quant, comp)] = result_quant.get_component(comp)
    pickle.dump(data, fout)

    data_ref = pickle.load(open(os.path.join(REF_DIR, filename), "rb"))

    for quant in ["Energy"] + quantities:
        for comp in tab_result.results.get(quant).get_component_list():
            _data = data[(quant, comp)]
            _data_ref = data_ref[(quant, comp)]
            assert _data == approx(_data_ref), (
                f"tabulation along path gave a wrong result for quantity {quant} component {comp} "
                + "with a maximal difference {}".format(max(abs(data - data_ref))))

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
