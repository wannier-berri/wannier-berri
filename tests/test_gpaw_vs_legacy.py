from .common import OUTPUT_DIR, DATA_DIR
from irrep.spacegroup import SpaceGroup
from wannierberri.system import SystemSOC
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.grid import Path
from wannierberri.w90files.w90data_soc import WannierDataSOC
from wannierberri.calculators.static import AHC, Spin
from wannierberri import run, Grid
import os
import numpy as np
from irrep.parsers.gpaw import is_new_gpaw

from gpaw import GPAW



def get_system(legacy=False, theta=0, phi=0):

    results = {}

    if legacy:
        calc = GPAW(os.path.join(DATA_DIR, "Fe-gpaw-irred", "Fe-nscf-irred-222.gpw"), txt=None,
                    legacy_gpaw=True)
    else:
        calc = GPAW(os.path.join(DATA_DIR, "Fe-gpaw-irred-new", "Fe-nscf-irred-222.gpw"), txt=None)
    sg = SpaceGroup.from_gpaw(calc, legacy_gpaw=legacy)
    projection_sp3d2 = Projection(position_num=[0, 0, 0], orbital='sp3d2', spacegroup=sg)
    projection_t2g = Projection(position_num=[0, 0, 0], orbital='t2g', spacegroup=sg)
    proj_set = ProjectionsSet([projection_sp3d2, projection_t2g])
    if legacy:
        assert not is_new_gpaw(calc)


    wandata = WannierDataSOC.from_gpaw(
        calculator=calc,
        projections=proj_set,
        mp_grid=(2, 2, 2),
        spacegroup=sg,
        legacy_gpaw=legacy,
        irreducible=True,

    )
    wandata.wannierise(
        froz_min=-100,
        froz_max=15,
        outer_min=-100,
        outer_max=np.inf,
        num_iter=50,
        print_progress_every=10,
        sitesym=True,
        localise=True,
    )

    results["spreads_up"] = wandata.data_up.chk.wannier_spreads
    results["spreads_down"] = wandata.data_down.chk.wannier_spreads

    system_soc = SystemSOC.from_wannierdata(wandata=wandata, berry=True, silent=False)
    system_soc.set_soc_axis(theta=theta, phi=phi, alpha_soc=1.0)

    # results["system_soc"] = system_soc

    results["wannier_centers_cart_up"] = system_soc.system_up.wannier_centers_cart
    results["wannier_centers_cart_down"] = system_soc.system_down.wannier_centers_cart

    points = {"G": [0.0, 0.0, 0.0],
          "H": [0.5, -0.5, -0.5],
          "P": [0.75, 0.25, -0.25],
          "N": [0.5, 0.0, -0.5],
         }

    path_str = "GHPNGP"
    path = Path.from_nodes(real_lattice=system_soc.real_lattice,
                nodes=[points[p] for p in path_str],
        labels=[p for p in path_str],
        length=1000)   # length [ Ang] ~= 2*pi/dk

    results["bands_wannier_soc"] = system_soc.get_bandstructure(path=path).get_data("Energy")
    results["bands_wannier_up"] = system_soc.system_up.get_bandstructure(path=path).get_data("Energy")
    results["bands_wannier_dw"] = system_soc.system_down.get_bandstructure(path=path).get_data("Energy")


    grid = Grid(system_soc, NK=48, NKFFT=8)

    EF = 9.53754
    tetra = False

    Efermi = np.linspace(EF - 0.5, EF + 0.5, 101)
    res_dir = os.path.join(OUTPUT_DIR, f"Fe-soc-001-{'legacy' if legacy else 'new'}")
    os.makedirs(res_dir, exist_ok=True)


    result_run = run(system_soc,
            grid=grid,
            fout_name=f"{res_dir}/Fe-soc-001",
            calculators={"ahc_int": AHC(Efermi=Efermi, tetra=tetra, kwargs_formula={"external_terms": False}),
                         "ahc_ext": AHC(Efermi=Efermi, tetra=tetra, kwargs_formula={"internal_terms": False}),
                         "spin": Spin(Efermi=Efermi, tetra=tetra)
                         },
            adpt_num_iter=50,
            )


    results["ahc_int"] = result_run.results["ahc_int"].data
    results["ahc_ext"] = result_run.results["ahc_ext"].data
    results["spin"] = result_run.results["spin"].data

    return results

# this test takes 5 minutes to run, so it is not included in the default test suite


def _test_compare_legacy_and_new_gpaw():
    results_legacy = get_system(legacy=True)
    results_new = get_system(legacy=False)

    atol = {
        "spreads_up": 1e-3,
        "spreads_down": 1e-3,
        "wannier_centers_cart_up": 1e-3,
        "wannier_centers_cart_down": 1e-3,
        "bands_wannier_up": 1e-3,
        "bands_wannier_dw": 1e-3,
        "bands_wannier_soc": 1e-3,
        "bands_wannier_soc_spin_z": 1e-3,
        "ahc_int": 2000,
        "ahc_ext": 20,
        "spin": 0.05
    }

    def compare(a, b, atol=1e-3, name=""):
        diff = np.max(np.abs(a - b))
        print(f"Max difference for {name}: {diff}, tolerance: {atol}, maxval a: {np.max(np.abs(a))}, maxval_b: {np.max(np.abs(b))}")
        assert diff < atol, f"Max difference {diff} for {name} exceeds tolerance {atol}"

    for key in results_legacy.keys():
        compare(results_legacy[key], results_new[key], atol=atol[key], name=key)
