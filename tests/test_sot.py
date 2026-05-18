# import pytest
import numpy as np
from pathlib import Path
from wannierberri.system.system_soc import SystemSOC
from wannierberri.calculators.static import Torkance
import wannierberri as wberri

# I am assuming your test data directory structure here based on your imports.
HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE / "data/Fe_system_soc"


def test_sot_operator_generation():
    """Validates the generation of the SOT operator matrix in real space."""
    system_soc = SystemSOC.from_npz(str(DATA_DIR / "system_soc"))

    # Generate current matrix
    system_soc.set_torque_operators_R(theta=0, phi=0)
    current_SOT_R = system_soc.get_R_mat('SOT')

    # Load reference matrix
    ref_SOT_R = np.load(HERE / "ref_data/ref_SOT_R.npy")

    # Compare (atol sets the absolute tolerance for near-zero values)
    np.testing.assert_allclose(current_SOT_R, ref_SOT_R, rtol=1e-5, atol=1e-8)


def test_torkance_calculation():
    """Validates the output tensor of the Torkance calculator."""
    system_soc = SystemSOC.from_npz(str(DATA_DIR / "system_soc"))
    grid = wberri.Grid(system_soc, length=20)
    Efermi = np.linspace(-2.0, 2.0, 5)

    torkance_calc = Torkance(Efermi=Efermi, Emin=-2.0, Emax=2.0)

    run_results = wberri.run(
        system_soc,
        grid=grid,
        calculators={"torkance": torkance_calc},
        use_irred_kpt=False,
        symmetrize=False
    )

    current_tensor = run_results.results["torkance"].data
    ref_tensor = np.load(HERE / "ref_data/ref_torkance.npy")

    np.testing.assert_allclose(current_tensor, ref_tensor, rtol=1e-5, atol=1e-8)
