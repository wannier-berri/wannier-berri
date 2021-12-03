import wannierberri as wberri
from create_system import pythtb_Haldane, system_Haldane_PythTB
import numpy as np

def test_path(system_Haldane_PythTB):
    k_nodes = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    path = wberri.Path(system_Haldane_PythTB, k_nodes=k_nodes, nk=3)

    assert path.labels == {0: '1', 2: '2'}, "path.labels is wrong"
    assert np.all(path.K_list == np.array([[0,  0., 0.], [0.25, 0.25 ,0.25], [0.5, 0.5, 0.5]])), "path.K_list is wrong"

