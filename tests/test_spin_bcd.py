from .test_formula import get_datak, check_formula_output as _check_formula_output, FORMULA_REFERENCE_FILENAMES
from wannierberri.formula import covariant as frml_cov
import numpy as np
import pytest
from pythtb import tb_model

import wannierberri as wberri
from wannierberri import calculators as calc
from wannierberri.system import System_R
from wannierberri.utility import alpha_A, beta_A


check_formula_output = _check_formula_output


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


@pytest.fixture(scope="module")
def system_model():
    model = _build_user_tb_model()
    system = System_R.from_pythtb(model, spin=True)
    return system


def _reduce_spin_bcd_raw(spin_bcd_raw):
    return 0.5 * (
        spin_bcd_raw[:, :, alpha_A, beta_A, :] -
        spin_bcd_raw[:, :, beta_A, alpha_A, :]
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


def test_spin_bcd_fermi_surface_vs_sea_user_tb_model(system_model):
    system = system_model
    grid = wberri.Grid(system, NKFFT=[6, 6, 1], NKdiv=[6, 6, 1])
    efermi = np.linspace(-4.5, 4.0, 11)

    result = _run_spin_bcd_pair(system, efermi, grid)

    fs_reduced = _reduce_spin_bcd_raw(result.results["spin_bcd_fsurf"].data)
    fsea_reduced = _reduce_spin_bcd_raw(result.results["spin_bcd_fsea"].data)

    spin_idx = 2
    fs_slice = fs_reduced[:, :, 2, spin_idx]
    fsea_slice = fsea_reduced[:, :, 2, spin_idx]
    scale = max(np.max(np.abs(fs_slice)), np.max(np.abs(fsea_slice)))

    max_abs_diff = np.max(np.abs(fs_slice - fsea_slice))
    print(f"Spin_BCD scale: {scale}, max_abs_diff: {max_abs_diff}")

    assert scale > 1e-2, "spin-BCD test on the user TB model became trivially zero"
    assert max_abs_diff < 2.5e-3, (
        "SpinBerryDipole_FermiSurf and SpinBerryDipole_FermiSea disagree on "
        f"the user TB model by a maximal absolute difference of {max_abs_diff}."
    )


def test_spin_bcd_fermi_sea_requires_simple_current(system_model):
    system = system_model
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


def test_spin_bcd_vanishes_in_tr_symmetric_kane_mele(system_model):
    system = system_model
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




@pytest.fixture(scope="module")
def datak_model(system_model):
    system = system_model
    datak = get_datak(system, k=[0.1, 0.2, 0.0], NKFFT=[4, 4, 1])
    return datak


formula_nn = ["SpinOmega", "VelOmega", "DerSpinOmegaSimple"]
formula_ln = []


@pytest.mark.parametrize("formula_class_name", ["DerSpinOmegaSimple"])
def test_formula(formula_class_name, check_formula_output, datak_model):
    data = datak_model
    NB = data.num_wann
    degen_groups = data.get_bands_in_range_groups(emin=-10, emax=30, degen_thresh=1)
    kwargs = {}
    if formula_class_name == "SpinVelocity":
        kwargs["spin_current_type"] = "ryoo"

    try:
        formula = getattr(frml_cov, formula_class_name)
    except AttributeError:
        raise ValueError(f"unknown formula {formula_class_name}")

    value = {}
    allXkeys = ["Xnn", "Xll", "XlnXnl", "XnlXln", "XnnXnn", "XllXll"]
    for ik in range(data.nk):
        for Xkey in allXkeys:
            value[f"{Xkey}_ik={ik}"] = []
        for n in degen_groups[ik]:
            inn = np.arange(n[0], n[1])
            out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
            print(f"Testing {formula_class_name} for ik={ik}, inn={inn} out={out}")
            form = formula(data, **kwargs)
            if formula_class_name not in formula_nn:
                Xnl = form.ln(ik, inn, out)
                Xln = form.ln(ik, out, inn)
                value[f"XnlXln_ik={ik}"].append(np.einsum("nl...,ln...->...", Xnl, Xln))
                value[f"XlnXnl_ik={ik}"].append(np.einsum("ln...,nl...->...", Xln, Xnl))
            if formula_class_name not in formula_ln:
                Xll = form.ll(ik, inn, inn)
                Xnn = form.nn(ik, inn, inn)
                value[f"Xll_ik={ik}"].append(np.einsum("ll...->...", Xll))
                value[f"Xnn_ik={ik}"].append(np.einsum("nn...->...", Xnn))
                value[f"XllXll_ik={ik}"].append(np.einsum("ll...,ll...->...", Xll, Xll))
                value[f"XnnXnn_ik={ik}"].append(np.einsum("nn...,nn...->...", Xnn, Xnn))
        # we can compare only gauge-invariant combinations, so we sum over inn and out
        for Xkey in allXkeys:
            value[f"{Xkey}_ik={ik}"] = np.array(value[f"{Xkey}_ik={ik}"])
    if "Der" in formula_class_name or formula_class_name in ["SpinOmega", "VelOmega"]:
        rel_tol = 1e-5
        atol_zero = 2e-6
    else:
        rel_tol = 1e-6
        atol_zero = 1e-10
    reference_name = FORMULA_REFERENCE_FILENAMES.get(formula_class_name, formula_class_name)
    check_formula_output(value=value, filename=reference_name, rel_tol=rel_tol, atol_zero=atol_zero)
