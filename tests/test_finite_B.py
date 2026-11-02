from copy import deepcopy
from .common import OUTPUT_DIR, REF_DIR
import numpy as np
import os


def test_finite_field(system_Fe_W90):
    B_tesla = np.array([0.123, 0.456, 0.789]) * 100
    system = deepcopy(system_Fe_W90)
    system.add_Zeeman_spin(B=B_tesla)
    H_Z = system.get_R_mat('Ham')
    fname = "Fe_W90_Ham_with_B.npy"
    np.save(os.path.join(OUTPUT_DIR, fname), H_Z)
    H_Z_ref = np.load(os.path.join(REF_DIR, "systems", "Fe_W90_Ham_with_B.npy"))
    assert np.allclose(H_Z, H_Z_ref, atol=1e-8), f"Hamiltonian with B field does not match reference for {fname}"
