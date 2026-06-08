import numpy as np
import pytest
from pythtb import tb_model

import wannierberri as wberri
from wannierberri import calculators as calc
from wannierberri import models
from wannierberri.system import System_R
from wannierberri.utility import alpha_A, beta_A


def _build_user_tb_model(t=0.0, t1=-1.0, t2=-0.1, t3=-1.0, exchange=1.0):
    sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    lat = [[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]]
    orb = [[0, 1 / 3], [1 / 3, 0], [1 / 3, 2 / 3], [2 / 3, 1 / 3]]

    model = tb_model(2, 2, lat, orb, nspin=2)

    onsite_1 = 4 * t * sigma_0 + exchange * sigma_z
    onsite_2 = 4 * t * sigma_0 - exchange * sigma_z
    onsite_3 = 4 * t * sigma_0 + exchange * sigma_z
    onsite_4 = 4 * t * sigma_0 - exchange * sigma_z
    model.set_onsite([onsite_1, onsite_2, onsite_3, onsite_4])

    model.set_hop(t1, 0, 1, [0, 0])
    model.set_hop(t1, 0, 1, [0, 1])
    model.set_hop(t1, 0, 1, [-1, 0])

    model.set_hop(t2, 2, 3, [0, 0])
    model.set_hop(t2, 2, 3, [0, 1])
    model.set_hop(t2, 2, 3, [-1, 0])

    model.set_hop(t3, 0, 2, [0, 0])
    model.set_hop(t3, 0, 3, [-1, 0])
    model.set_hop(t3, 1, 2, [0, -1])
    model.set_hop(t3, 1, 3, [0, 0])
    return model


def _reduce_spin_bcd_raw(spin_bcd_raw):
    return 0.5 * (
        spin_bcd_raw[:, :, alpha_A, beta_A, :]
        - spin_bcd_raw[:, :, beta_A, alpha_A, :]
    )


def _run_spin_bcd_pair(system, efermi, grid):
    calculators = {
        "spin_bcd_fsurf": calc.static.SpinBerryDipole_FermiSurf(
            Efermi=efermi,
            tetra=True,
            use_factor=False,
            kwargs_formula={"spin_current_type": "simple", "external_terms": False},
        ),
        "spin_bcd_fsea": calc.static.SpinBerryDipole_FermiSea(
            Efermi=efermi,
            tetra=True,
            use_factor=False,
            kwargs_formula={"spin_current_type": "simple", "external_terms": False},
        ),
    }
    return wberri.run(
        system,
        grid=grid,
        calculators=calculators,
        parallel=False,
        adpt_num_iter=0,
        use_irred_kpt=False,
        symmetrize=False,
        dump_results=False,
    )


def test_spin_bcd_fermi_surface_vs_sea_user_tb_model():
    model = _build_user_tb_model()
    system = System_R.from_pythtb(model, spin=True)
    grid = wberri.Grid(system, NKFFT=[6, 6, 1], NKdiv=[12, 12, 1])
    efermi = np.linspace(-4.5, 4.0, 121)

    result = _run_spin_bcd_pair(system, efermi, grid)

    fs_reduced = _reduce_spin_bcd_raw(result.results["spin_bcd_fsurf"].data)
    fsea_reduced = _reduce_spin_bcd_raw(result.results["spin_bcd_fsea"].data)

    spin_idx = 2
    fs_slice = fs_reduced[:, :, 2, spin_idx]
    fsea_slice = fsea_reduced[:, :, 2, spin_idx]
    scale = max(np.max(np.abs(fs_slice)), np.max(np.abs(fsea_slice)))

    max_abs_diff = np.max(np.abs(fs_slice - fsea_slice))

    assert scale > 1e-4, "spin-BCD test on the user TB model became trivially zero"
    assert max_abs_diff < 2.5e-3, (
        "SpinBerryDipole_FermiSurf and SpinBerryDipole_FermiSea disagree on "
        f"the user TB model by a maximal absolute difference of {max_abs_diff}."
    )


def test_spin_bcd_fermi_sea_requires_simple_current():
    model = _build_user_tb_model()
    system = System_R.from_pythtb(model, spin=True)
    grid = wberri.Grid(system, NKFFT=[3, 3, 1], NKdiv=[3, 3, 1])
    calculator = calc.static.SpinBerryDipole_FermiSea(
        Efermi=np.array([0.0]),
        tetra=True,
        use_factor=False,
        kwargs_formula={"spin_current_type": "ryoo", "external_terms": False},
    )

    with pytest.raises(NotImplementedError, match="simple"):
        wberri.run(
            system,
            grid=grid,
            calculators={"spin_bcd_fsea": calculator},
            parallel=False,
            adpt_num_iter=0,
            use_irred_kpt=False,
            symmetrize=False,
            dump_results=False,
        )


def test_spin_bcd_vanishes_in_tr_symmetric_kane_mele():
    model = models.KaneMele_ptb("odd")
    system = System_R.from_pythtb(model, spin=True)
    grid = wberri.Grid(system, NKFFT=[4, 4, 1], NKdiv=[6, 6, 1])
    efermi = np.linspace(-3.0, 3.0, 121)

    result = _run_spin_bcd_pair(system, efermi, grid)

    fs_reduced = _reduce_spin_bcd_raw(result.results["spin_bcd_fsurf"].data)
    fsea_reduced = _reduce_spin_bcd_raw(result.results["spin_bcd_fsea"].data)

    spin_idx = 2
    fs_slice = fs_reduced[:, :, 2, spin_idx]
    fsea_slice = fsea_reduced[:, :, 2, spin_idx]

    assert np.max(np.abs(fs_slice)) < 1e-8
    assert np.max(np.abs(fsea_slice)) < 1e-8
