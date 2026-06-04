import os
import numpy as np
import wannierberri as wberri
from wannierberri.calculators.static import TorkanceEven, TorkanceOdd

# Clean relative import to grab the path variables from tests/common.py
from .common import REF_DIR_INTEGRATE


def test_torkance_even_calculation(system_Fe_gpaw_soc_angle_symmetrized):
    """Validates the interband even component of the Torkance tensor (Fermi-sea)."""
    system_soc = system_Fe_gpaw_soc_angle_symmetrized
    # this system has SOT operators
        
    grid = wberri.Grid(system_soc, length=20)
    Efermi = np.linspace(-2.0, 2.0, 5)
    calc_even = TorkanceEven(Efermi=Efermi, Emin=-2.0, Emax=2.0)

    run_results = wberri.run(
        system_soc,
        grid=grid,
        calculators={"torkance_even": calc_even},
        use_irred_kpt=False,
        symmetrize=False
    )

    current_tensor = run_results.results["torkance_even"].data
    ref_tensor = np.load(os.path.join(REF_DIR_INTEGRATE, "ref_torkance_even.npy"))

    np.testing.assert_allclose(current_tensor, ref_tensor, rtol=1e-5, atol=1e-8)


def test_torkance_odd_calculation(system_Fe_gpaw_soc_angle_symmetrized):
    """Validates the intraband odd component of the Torkance tensor (Fermi-surface)."""
    system_soc = system_Fe_gpaw_soc_angle_symmetrized
    
    # Must explicitly set operators for the rotated basis matching the fixture
    system_soc.set_torque_operators_R(theta=49, phi=33, units="degrees")
    
    grid = wberri.Grid(system_soc, length=20)
    Efermi = np.linspace(-2.0, 2.0, 5)
    calc_odd = TorkanceOdd(Efermi=Efermi, Emin=-2.0, Emax=2.0)

    run_results = wberri.run(
        system_soc,
        grid=grid,
        calculators={"torkance_odd": calc_odd},
        use_irred_kpt=False,
        symmetrize=False
    )

    current_tensor = run_results.results["torkance_odd"].data
    ref_tensor = np.load(os.path.join(REF_DIR_INTEGRATE, "ref_torkance_odd.npy"))

    np.testing.assert_allclose(current_tensor, ref_tensor, rtol=1e-5, atol=1e-8)
