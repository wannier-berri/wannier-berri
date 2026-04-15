import os

import numpy as np
import pytest
from pytest import approx
import wannierberri as wberri
from wannierberri.grid.Kpoint import KpointBZparallel

from .common import OUTPUT_DIR, REF_DIR

from wannierberri.formula import elementary  as frml_el
from wannierberri.formula import covariant as frml_cov


@pytest.fixture(scope="module")
def datak_Fe(system_Fe_W90):
    system = system_Fe_W90
    k = np.array([0.1, 0.2, -0.3])
    grid = wberri.Grid(system=system, NKFFT=[4, 3, 2], NKdiv=1, use_symmetry=False)
    dK = 1. / grid.div
    NKFFT = grid.FFT
    factor = 1. / np.prod(grid.div)
    kpoint = KpointBZparallel(K=k, dK=dK, NKFFT=NKFFT, factor=factor, pointgroup=None)
    assert kpoint.Kp_fullBZ == approx(k / grid.FFT)
    return wberri.data_K.Data_K_R(system=system, dK=[0, 0, 0], grid=grid, Kpoint=kpoint, fftlib='fftw')


def test_Hermitean(datak_Fe):
    data = datak_Fe
    NB = data.num_wann
    degen_groups = data.get_bands_in_range_groups(emin=-10, emax=30, degen_thresh=0.5)
    for ik in range(data.nk):
        # print (f"ik={ik}, degen_groups={degen_groups[ik]}")
        for n in degen_groups[ik]:
            inn = np.arange(n[0], n[1])
            out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
            for formula in [frml_el.Dcov, frml_el.DerDcov, frml_el.Der2Dcov]:
                form = formula(data)
                Xnl = form.ln(ik, inn, out)
                Xln = form.ln(ik, out, inn)
                assert np.allclose(Xnl, -Xln.conj().swapaxes(0, 1)), f"{formula.__name__} nl and ln are not Hermitean conjugate for ik={ik}, inn={inn}"
            for formula in [frml_cov.Spin, frml_cov.DerOmega, frml_cov.Omega, frml_cov.Der2Omega,
                            frml_cov.Der2A, frml_cov.Der2H, frml_cov.Der2O]:
                form = formula(data)
                Xll = form.ll(ik, inn, inn)
                Xnn = form.nn(ik, inn, inn)
                assert np.allclose(Xll, Xll.conj().swapaxes(0, 1)), f"{formula.__name__} ll is not Hermitean for ik={ik}, inn={inn}"
                assert np.allclose(Xnn, Xnn.conj().swapaxes(0, 1)), f"{formula.__name__} nn is not Hermitean for ik={ik}, inn={inn}"
            for formula in [frml_cov.Spin, frml_cov.Der2A]:
                form = formula(data)
                Xln = form.ln(ik, inn, inn)
                Xnl = form.nl(ik, inn, inn)
                assert np.allclose(Xln, Xnl.conj().swapaxes(0, 1)), f"{formula.__name__} ln and nl are not Hermitean conjugate for ik={ik}, inn={inn}"
            for key in ('Ham', 'AA'):
                for der in 0, 1, 2:
                    Xbar = data.covariant(key, der)
                    Xln = Xbar.ln(ik, inn, inn)
                    Xnl = Xbar.nl(ik, inn, inn)
                    assert np.allclose(Xln, Xnl.conj().swapaxes(0, 1)), f"Covariant {key} der{der} ln and nl are not Hermitean conjugate for ik={ik}, inn={inn}"
                    Xll = Xbar.ll(ik, inn, inn)
                    Xnn = Xbar.nn(ik, inn, inn)
                    assert np.allclose(Xll, Xll.conj().swapaxes(0, 1)), f"Covariant {key} der{der} ll is not Hermitean for ik={ik}, inn={inn}"


@pytest.fixture(scope="module")
def check_formula_output():
    def __inner(value, filename, abs_tol=1e-8, rel_tol=1e-6):
        path_out = os.path.join(OUTPUT_DIR, "formula")
        path_ref = os.path.join(REF_DIR, "formula")
        os.makedirs(path_out, exist_ok=True)
        np.savez(os.path.join(path_out, filename + ".npz"), **value)
        value_ref = np.load(os.path.join(path_ref, filename + ".npz"))
        for k, val in value_ref.items():
            print(f"Checking {filename} key {k}")
            val_out = value[k]
            assert np.allclose(val, val_out, atol=abs_tol, rtol=rel_tol), (
                f"Formula output {filename} key {k} does not match reference."
                f"value_ref = \n{val}\n  obtained = \n{val_out}\n"
                f"max abs diff is {np.max(abs(val - val_out))}")
            
    return __inner


formula_all = ["Dcov", "DerDcov", "Der2Dcov", "InvMass", "DerWln", "DEinv_ln",
    "Spin", "DerOmega", "Omega",
        "Der2Omega", "Der2A", "Der2B", "Der2H",
        "Der2O", "Der3E", "Hamiltonian", "Velocity", "Spin", "Der2Spin", "Morb_H",
        "morb", "DerMorb", "Dermorb", "Der2Morb", "Der2morb", "SpinOmega", "SpinVelocity",
        "VelOmega", "VelSpin", "VelVel", "VelVelVel", "VelMassVel", "OmegaS", "OmegaOmega",]

formula_ln = ["Dcov", "DerDcov", "Der2Dcov", "DEinv_ln"]
formula_nn = ["VelMassVel", "Dermorb", "VelVelVel", "Der2A", "OmegaS", "Der2Morb",
              "Der2H", "Der2Spin", "Der2Omega", "Der2O", "DerOmega", "SpinVelocity", "SpinOmega",
              "OmegaOmega", "VelOmega", "Der3E", "VelSpin", "DerMorb", "morb", "VelVel", "Morb_H",
              "VelVel", "Omega", "Der2morb"]
formula_all = set(formula_all + formula_ln + formula_nn)


@pytest.mark.parametrize("formula_class_name", formula_all)
def test_formula(datak_Fe, formula_class_name, check_formula_output):
    data = datak_Fe
    NB = data.num_wann
    degen_groups = data.get_bands_in_range_groups(emin=-10, emax=30, degen_thresh=0.5)
    kwargs = {}
    if formula_class_name == "SpinVelocity":
        kwargs["spin_current_type"] = "ryoo"

    try:
        formula = getattr(frml_el, formula_class_name)
    except AttributeError:
        try:
            formula = getattr(frml_cov, formula_class_name)
        except AttributeError:
            raise ValueError(f"unknown formula {formula_class_name}")

    value = {}
    for ik in range(data.nk):
        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []
        for n in degen_groups[ik]:
            inn = np.arange(n[0], n[1])
            out = np.concatenate((np.arange(0, n[0]), np.arange(n[1], NB)))
            form = formula(data, **kwargs)
            if formula_class_name not in formula_nn:
                Xnl = form.ln(ik, inn, out)
                Xln = form.ln(ik, out, inn)
                lst1.append(np.einsum("nl...,ln...->...", Xnl, Xln))
                lst2.append(np.einsum("ln...,nl...->...", Xln, Xnl))
            if formula_class_name not in formula_ln:
                Xll = form.ll(ik, inn, inn)
                Xnn = form.nn(ik, inn, inn)
                lst3.append(np.einsum("ll...->...", Xll))
                lst4.append(np.einsum("nn...->...", Xnn))
        # we can compare only gauge-invariant combinations, so we sum over inn and out
        value[f"XnlXln_ik={ik}"] = lst1
        value[f"XlnXnl_ik={ik}"] = lst2
        value[f"XllXll_ik={ik}"] = lst3
        value[f"XnnXnn_ik={ik}"] = lst4
    check_formula_output(value=value, filename=f"{formula_class_name}")
