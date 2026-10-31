import pytest
from wannierberri.calculators.static import LFAHE_spin_fsea, LFAHE_spin_fsurf, AHC, LFAHE_spin
from wannierberri import run
from wannierberri.grid import Grid
import numpy as np
from copy import deepcopy
from .common import OUTPUT_DIR

from wannierberri.models import Haldane_ptb_double
plot_result = False  # Set to True to plot the results for visual inspection


@pytest.fixture(scope="module")
def system_Haldane_ptb_double():
    return Haldane_ptb_double(delta=0.2, hop1=-1.0, hop2=0.15, phi=np.pi / 2, soc_term=0.3)


@pytest.mark.parametrize("B_tesla", [10, 50, 100])
def test_LFAHE_vs_FiniteField(B_tesla, system_Haldane_ptb_double):
    system = system_Haldane_ptb_double
    system_magnetic = deepcopy(system)
    system_magnetic.add_Zeeman_spin(B=np.array([0, 0, B_tesla]))
    EFermi = np.linspace(-1.8, -1.2, 61)
    kwargs = dict(Efermi=EFermi, tetra=True)
    kwargs_ext = dict(kwargs_formula=dict(internal_terms=False, external_terms=True), **kwargs)
    kwargs_int = dict(kwargs_formula=dict(internal_terms=True, external_terms=False), **kwargs)
    grid = Grid(system, NK=(60, 60, 1), NKFFT=(6, 6, 1))
    calculators_mag = {}
    calculators_nomag = {}
    calculators_mag["AHC_int"] = AHC(**kwargs_int)
    calculators_mag["AHC_ext"] = AHC(**kwargs_ext)
    calculators_nomag["LFAHE_spin_fsea_int"] = LFAHE_spin_fsea(**kwargs_int)
    calculators_nomag["LFAHE_spin_fsea_ext"] = LFAHE_spin_fsea(**kwargs_ext)
    calculators_nomag["LFAHE_spin_fsurf_int"] = LFAHE_spin_fsurf(**kwargs_int)
    calculators_nomag["LFAHE_spin_fsurf_ext"] = LFAHE_spin_fsurf(**kwargs_ext)

    kwargs_run = dict(grid=grid, use_irred_kpt=False, symmetrize=False)  # do not use symmetry because tetrahedron is not very symmetric
    result_non_magnetic = run(system, calculators=calculators_nomag, **kwargs_run).results
    result_magnetic = run(system_magnetic, calculators=calculators_mag, **kwargs_run).results

    for term in ["int", "ext"]:
        ahc_Z_fsurf = result_non_magnetic[f"LFAHE_spin_fsurf_{term}"].data[:, 2, 2]
        ahc_Z_fsea = result_non_magnetic[f"LFAHE_spin_fsea_{term}"].data[:, 2, 2]
        AHC_finiteB = result_magnetic[f"AHC_{term}"].data[:, 2]
        ahc_linear = (ahc_Z_fsurf + ahc_Z_fsea) * B_tesla
        if plot_result:
            from matplotlib import pyplot as plt
            plt.close('all')
            plt.plot(EFermi, AHC_finiteB, label=f"AHC_{term} with B={B_tesla} T")
            plt.plot(EFermi, ahc_linear, label=f"LFAHE_{term} with B={B_tesla} T", linestyle="--")
            plt.plot(EFermi, ahc_Z_fsea * B_tesla, label=f"fermi-sea {term} with B={B_tesla} T", linestyle=":")
            plt.plot(EFermi, ahc_Z_fsurf * B_tesla, label=f"fermi-surface {term} with B={B_tesla} T", linestyle=":")
            plt.title(f"AHC {term} comparison for B={B_tesla} T")
            plt.xlabel("Energy (eV)")
            plt.ylabel("AHC (S/m)")
            plt.legend()
            plt.savefig(f"{OUTPUT_DIR}/AHC_comparison_{term}_B{B_tesla}.png")
            plt.close()
        diff = np.max(np.abs(ahc_linear - AHC_finiteB))
        maxval = np.max(np.abs([ahc_linear, AHC_finiteB]))
        assert diff < maxval * 2e-2, f"LFAHE and finite B field AHC do not match for term {term} and B={B_tesla} T. The maximum difference is {diff} S/m with maxval={maxval} S/m"


@pytest.mark.parametrize("B_tesla", [10, 50, 100])
def test_sanity(B_tesla, system_Haldane_ptb_double):
    system = system_Haldane_ptb_double
    EFermi = np.linspace(-1.8, -1.2, 61)
    kwargs = dict(Efermi=EFermi, tetra=True)
    kwargs_ext = dict(kwargs_formula=dict(internal_terms=False, external_terms=True), **kwargs)
    kwargs_int = dict(kwargs_formula=dict(internal_terms=True, external_terms=False), **kwargs)
    grid = Grid(system, NK=(12, 12, 1), NKFFT=(6, 6, 1))
    calculators_nomag = {}
    calculators_nomag["LFAHE_spin_fsea_int"] = LFAHE_spin_fsea(**kwargs_int)
    calculators_nomag["LFAHE_spin_fsea_ext"] = LFAHE_spin_fsea(**kwargs_ext)
    calculators_nomag["LFAHE_spin_fsurf_int"] = LFAHE_spin_fsurf(**kwargs_int)
    calculators_nomag["LFAHE_spin_fsurf_ext"] = LFAHE_spin_fsurf(**kwargs_ext)
    calculators_nomag["LFAHE_spin_ext"] = LFAHE_spin(**kwargs_ext)
    calculators_nomag["LFAHE_spin_int"] = LFAHE_spin(**kwargs_int)
    calculators_nomag["LFAHE_spin"] = LFAHE_spin(**kwargs)
    calculators_nomag["LFAHE_spin_fsea"] = LFAHE_spin_fsea(**kwargs)
    calculators_nomag["LFAHE_spin_fsurf"] = LFAHE_spin_fsurf(**kwargs)
    B = np.random.rand(3)  # random B field direction
    B_normalized = B / np.linalg.norm(B)

    calculators_nomag["LFAHE_spin_fsea_dir"] = LFAHE_spin_fsea(**kwargs, kwargs_formula=dict(Bdirection=B))


    kwargs_run = dict(grid=grid, use_irred_kpt=False, symmetrize=False)  # do not use symmetry because tetrahedron is not very symmetric
    result_non_magnetic = run(system, calculators=calculators_nomag, **kwargs_run).results

    # first test additivity of fsea and fsurf
    for terms in ["_int", "_ext", ""]:
        r_fsea = result_non_magnetic[f"LFAHE_spin_fsea{terms}"]
        r_fsurf = result_non_magnetic[f"LFAHE_spin_fsurf{terms}"]
        r_sum = result_non_magnetic[f"LFAHE_spin{terms}"]
        diff = (r_sum - r_fsea - r_fsurf)._maxval
        assert diff < 1e-10, f"fsea{terms} and fsurf{terms} do not add up to total, maximum difference is {diff}"

    for term in ["", "_fsea", "_fsurf"]:
        r_ext = result_non_magnetic[f"LFAHE_spin{term}_ext"]
        r_int = result_non_magnetic[f"LFAHE_spin{term}_int"]
        r_sum = result_non_magnetic[f"LFAHE_spin{term}"]
        diff = (r_sum - r_ext - r_int)._maxval
        assert diff < 1e-10, f"ext{term} and int{term} do not add up to total, maximum difference is {diff}"

    LFAHE_on_direction = result_non_magnetic["LFAHE_spin_fsea"].data @ B_normalized
    LFAHE_on_direction_dir = result_non_magnetic["LFAHE_spin_fsea_dir"].data
    diff = np.max(np.abs(LFAHE_on_direction - LFAHE_on_direction_dir))
    assert diff < 1e-10, f"LFAHE_spin_fsea on direction {B} does not match LFAHE_spin_fsea_dir, maximum difference is {diff}"
