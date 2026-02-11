import numpy as np
import pytest


@pytest.mark.parametrize("system", [
    "system_Haldane_PythTB",
    "system_Si_W90_JM_sym",
    "system_Si_W90_sym",
    "system_Si_W90_double",
    "system_kp_mass_iso_2",
    "system_kp_mass_aniso_0",
    "system_Chiral_left",
    "system_Chiral_left_TR",
    "system_Fe_WB_irreducible",
    "system_Fe_sym_W90",
    "system_Fe_gpaw_soc_z_symmetrized"
], indirect=True)
def test_symmetry(system):

    prec = 1e-6
    kpoint = np.array([0.123, 0.456, 0.789])
    errors, kpoints = system.check_symmetry(kpoint=kpoint)
    for key, value in errors.items():
        print(f"Quantity: {key[0]}, Symmetry {key[1]} k={kpoints[key[1]]}: max difference = {value:.2e}")
        if value > prec:
            raise AssertionError(f"Symmetry check failed for quantity {key[0]} and symmetry {key[1]}: max difference = {value:.2e} exceeds precision {prec:.2e}")
