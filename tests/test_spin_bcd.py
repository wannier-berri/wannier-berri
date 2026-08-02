from .test_formula import get_datak, check_formula_output as _check_formula_output, FORMULA_REFERENCE_FILENAMES
from wannierberri.formula import covariant as frml_cov
import numpy as np
import pytest
from pythtb import tb_model

import wannierberri as wberri
from wannierberri import calculators as calc
from wannierberri.system import System_R
from wannierberri.utility import alpha_A, beta_A
from .common import OUTPUT_DIR_RUN


check_formula_output = _check_formula_output


def _build_spin_bcd_tb_model(t=0.0, t1=-1.0, t2=-0.1, t3=-1.0, exchange=1.0):
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
    model = _build_spin_bcd_tb_model()
    system = System_R.from_pythtb(model, spin=True)
    return system


def _reduce_spin_bcd_raw(spin_bcd_raw):
    return 0.5 * (
        spin_bcd_raw[:, :, alpha_A, beta_A, :] -
        spin_bcd_raw[:, :, beta_A, alpha_A, :]
    )


def _run_spin_bcd_pair(system, efermi, grid, use_symmetry=False):
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
        use_irred_kpt=use_symmetry,
        symmetrize=use_symmetry,
        dump_results=False,
        fout_name=f"{OUTPUT_DIR_RUN}/SpinBCD-pair",
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


def test_spin_bcd_calculator_reference(check_run, system_model):
    efermi = np.linspace(-3.0, 3.0, 5)
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
    grid = wberri.Grid(system_model, NKFFT=[3, 3, 1], NKdiv=[3, 3, 1])

    check_run(
        system_model,
        calculators=calculators,
        grid=grid,
        fout_name="SpinBCD",
        precision=-1e-10,
    )


@pytest.mark.parametrize(
    "kwargs_formula",
    [
        {"spin_current_type": "ryoo", "external_terms": False},
        {"spin_current_type": "qiao", "external_terms": False},
        {"spin_current_type": "simple", "external_terms": True},
    ],
)
def test_spin_bcd_fermi_sea_rejects_unsupported_formula_settings(system_model, kwargs_formula):
    system = system_model
    grid = wberri.Grid(system, NKFFT=[3, 3, 1], NKdiv=[3, 3, 1])
    calculator = calc.static.SpinBerryDipole_FermiSea(
        Efermi=np.array([0.0]),
        tetra=True,
        use_factor=False,
        kwargs_formula=kwargs_formula,
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


@pytest.mark.parametrize("use_symmetry", [False, True])
def test_spin_bcd_vanishes_in_tr_symmetric_kane_mele(system_KaneMele_odd_PythTB, use_symmetry):
    system = system_KaneMele_odd_PythTB
    grid = wberri.Grid(system, NKFFT=[4, 4, 1], NKdiv=[6, 6, 1])
    efermi = np.linspace(-3.0, 3.0, 121)

    result = _run_spin_bcd_pair(system, efermi, grid, use_symmetry=use_symmetry)

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


def _rotate_degenerate_subspaces(data):
    _ = data.E_K
    rotation = np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)
    for ik, energies in enumerate(data.E_K):
        ib1 = 0
        for ib2 in list(np.where(np.diff(energies) > 1e-8)[0] + 1) + [len(energies)]:
            if ib2 - ib1 == 2:
                data._UU[ik, :, ib1:ib2] = data._UU[ik, :, ib1:ib2] @ rotation
            ib1 = ib2
    return data


def test_der_spin_omega_simple_degenerate_gauge(system_model):
    data = get_datak(system_model, k=[0.1, 0.2, 0.0], NKFFT=[4, 4, 1])
    data_rotated = _rotate_degenerate_subspaces(
        get_datak(system_model, k=[0.1, 0.2, 0.0], NKFFT=[4, 4, 1])
    )
    formula = frml_cov.DerSpinOmegaSimple(data)
    formula_rotated = frml_cov.DerSpinOmegaSimple(data_rotated)

    for ik in range(data.nk):
        degen_groups = data.get_bands_in_range_groups_ik(
            ik, emin=-10, emax=30, degen_thresh=1
        )
        for ib1, ib2 in degen_groups:
            inn = np.arange(ib1, ib2)
            out = np.concatenate((np.arange(0, ib1), np.arange(ib2, data.num_wann)))
            trace = formula.trace(ik, inn, out)
            trace_rotated = formula_rotated.trace(ik, inn, out)
            assert np.allclose(trace, trace_rotated, atol=1e-10, rtol=1e-10)


def test_der_spin_omega_simple_reference(check_formula_output, datak_model):
    data = datak_model
    NB = data.num_wann
    degen_groups = data.get_bands_in_range_groups(emin=-10, emax=30, degen_thresh=1)
    formula = frml_cov.DerSpinOmegaSimple(data)

    value = {}
    allXkeys = ["Xnn", "Xll"]
    for ik in range(data.nk):
        for Xkey in allXkeys:
            value[f"{Xkey}_ik={ik}"] = []
        for n in degen_groups[ik]:
            inn = np.arange(n[0], n[1])
            out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
            print(f"Testing DerSpinOmegaSimple for ik={ik}, inn={inn} out={out}")
            Xll = formula.ll(ik, inn, out)
            Xnn = formula.nn(ik, inn, out)
            value[f"Xll_ik={ik}"].append(np.einsum("ll...->...", Xll))
            value[f"Xnn_ik={ik}"].append(np.einsum("nn...->...", Xnn))
        for Xkey in allXkeys:
            value[f"{Xkey}_ik={ik}"] = np.array(value[f"{Xkey}_ik={ik}"])
    reference_name = FORMULA_REFERENCE_FILENAMES.get("DerSpinOmegaSimple", "DerSpinOmegaSimple")
    check_formula_output(value=value, filename=reference_name, rel_tol=1e-5, atol_zero=2e-6)
