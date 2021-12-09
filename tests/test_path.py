import wannierberri as wberri
from create_system import pythtb_Haldane, system_Haldane_PythTB
import numpy as np
from pytest import approx

def test_path(system_Haldane_PythTB):
    k_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, nk=3)
    assert path.labels == {0: '1', 2: '2'}, "path.labels is wrong"
    assert path.K_list == approx(np.array([[0,  0., 0.], [0.25, 0.25 ,0.25], [0.5, 0.5, 0.5]])), "path.K_list is wrong"

    k_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]
    k_labels = ["A", "B", "C"]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, labels=k_labels, nk=4)
    assert path.labels == {0: 'A', 3: 'B', 6:'C'}, "path.labels is wrong"
    assert path.K_list == approx(np.array([[0, 0, 0],
                                           [1, 1, 1],
                                           [2, 2, 2],
                                           [3, 3, 3],
                                           [3, 2, 2],
                                           [3, 1, 1],
                                           [3, 0, 0],]) / 6), "path.K_list is wrong"


    k_nodes = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, dk=1.0)
    print(path.labels)
    print(path.K_list)
    assert path.labels == {0: '1', 3: '2', 8:'3'}, "path.labels is wrong"
    assert path.K_list[:4, :] == approx(np.array([[0, 0, 3],
                                                  [0, 0, 2],
                                                  [0, 0, 1],
                                                  [0, 0, 0]]) / 6), "path.K_list is wrong"
    assert path.K_list[3:, :] == approx(np.array([[0, 0, 0],
                                                  [1, 1, 1],
                                                  [2, 2, 2],
                                                  [3, 3, 3],
                                                  [4, 4, 4],
                                                  [5, 5, 5],]) / 10), "path.K_list is wrong"
