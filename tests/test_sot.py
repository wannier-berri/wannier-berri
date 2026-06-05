import os
import numpy as np
import wannierberri as wberri
from wannierberri.calculators.static import TorkanceEven, TorkanceOdd

# Clean relative import to grab the path variables from tests/common.py
from .common import OUTPUT_DIR_RUN, REF_DIR_INTEGRATE


def test_torkance(system_Fe_gpaw_soc_angle):
    """Validates the interband even component of the Torkance tensor (Fermi-sea)."""
    system_soc = system_Fe_gpaw_soc_angle
    # this system has SOT operators

    grid = wberri.Grid(system_soc, NKFFT=(4, 4, 4), NKdiv=(2, 2, 2))
    Efermi = np.linspace(8.5, 10, 16)  # As in the other tests for gpaw
    calc_even = TorkanceEven(Efermi=Efermi)
    calc_odd = TorkanceOdd(Efermi=Efermi)


    run_results = wberri.run(
        system_soc,
        grid=grid,
        calculators={"torkance_even": calc_even, "torkance_odd": calc_odd},
        use_irred_kpt=False,
        symmetrize=False,
        fout_name=os.path.join(OUTPUT_DIR_RUN, "Fe_gpaw_soc_angle")
    )

    for aprt in "even", "odd":
        current_tensor = run_results.results[f"torkance_{aprt}"].data
        ref_tensor = np.load(os.path.join(REF_DIR_INTEGRATE, f"Fe_gpaw_soc_angle-torkance_{aprt}_iter-0000.npz"))["data"]
        np.testing.assert_allclose(current_tensor, ref_tensor, rtol=1e-5, atol=1e-8)
