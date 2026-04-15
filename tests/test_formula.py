import os

import numpy as np
import pytest
from pytest import approx
import wannierberri as wberri
from wannierberri.grid.Kpoint import KpointBZparallel

from .common import OUTPUT_DIR, REF_DIR

from wannierberri.formula import elementary  as frml_el
from wannierberri.formula import covariant as frml_cov
from wannierberri.formula import sdct as frml_sdct





def get_datak(system, k=[0.1, 0.2, -0.3], NKFFT=[4, 3, 2]):
    grid = wberri.Grid(system=system, NKFFT=NKFFT, NKdiv=1, use_symmetry=False)
    dK = 1. / grid.div
    factor = 1. / np.prod(grid.div)
    kpoint = KpointBZparallel(K=k, dK=dK, NKFFT=NKFFT, factor=factor, pointgroup=None)
    assert kpoint.Kp_fullBZ == approx(k / grid.FFT)
    data_k_class = wberri.data_K.get_data_k_class_from_system(system)
    data_k = data_k_class(system, dK=[0, 0, 0], grid=grid, Kpoint=kpoint, fftlib='fftw')
    return data_k


@pytest.fixture(scope="module")
def datak_Fe(system_Fe_sym_W90):
    return get_datak(system_Fe_sym_W90, k=[0.1, 0.2, -0.3], NKFFT=[1, 2, 3])


def test_Hermitean(datak_Fe):
    data = datak_Fe
    NB = data.num_wann
    degen_groups = data.get_bands_in_range_groups(emin=-10, emax=30, degen_thresh=0.01)
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
    def __inner(value, filename, rel_tol=1e-6, atol_zero=1e-10):
        path_out = os.path.join(OUTPUT_DIR, "formula")
        path_ref = os.path.join(REF_DIR, "formula")
        os.makedirs(path_out, exist_ok=True)
        np.savez(os.path.join(path_out, filename + ".npz"), **value)
        value_ref = np.load(os.path.join(path_ref, filename + ".npz"))
        for k, val in value_ref.items():
            print(f"Checking {filename} key {k}")
            val_out = value[k]
            assert val_out.shape == val.shape, f"Shape mismatch for {filename} key {k}: {val_out.shape} vs {val.shape}"
            if val.size == 0:
                continue
            maxval = np.max(abs(val))
            if maxval < atol_zero:
                assert np.allclose(val_out, 0, atol=atol_zero), f"{filename} key {k} is expected to be zero, but max value is {maxval} > {atol_zero}"
            else:
                adiff = np.max(abs(val - val_out))
                rdiff = adiff / maxval
                assert (rdiff < rel_tol), (
                    f"Formula output {filename} key {k} does not match reference."
                    f" max val: {maxval}, abs diff: {adiff}, rel diff :{rdiff} > {rel_tol}"
                )

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

formula_all = list(sorted(set(formula_all + formula_ln + formula_nn)))

formula_sdct = ["SDCT_sea_I", "SDCT_sea_II", "SDCT_surf_I", "SDCT_surf_II"]


@pytest.mark.parametrize("formula_class_name", formula_all)
def test_formula(datak_Fe, formula_class_name, check_formula_output):
    data = datak_Fe
    NB = data.num_wann
    degen_groups = data.get_bands_in_range_groups(emin=-10, emax=30, degen_thresh=0.01)
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
        value[f"XnlXln_ik={ik}"] = np.array(lst1)
        value[f"XlnXnl_ik={ik}"] = np.array(lst2)
        value[f"XllXll_ik={ik}"] = np.array(lst3)
        value[f"XnnXnn_ik={ik}"] = np.array(lst4)
    if formula_class_name in ["DerMorb", "Dermorb", "Der2Morb", "Der2morb"]:
        # it is weird that the result is changes so much, even on the same machine
        # TODO investigate the source of this sensitivity and try to improve it
        atol_zero = 2e-3  
        rel_tol = 1e-4
    elif "Der" in formula_class_name or formula_class_name in ["SpinOmega", "VelOmega"]:
        rel_tol = 1e-5
        atol_zero = 1e-6
    else:
        rel_tol = 1e-6
        atol_zero = 1e-10
    check_formula_output(value=value, filename=f"{formula_class_name}", rel_tol=rel_tol, atol_zero=atol_zero)


@pytest.mark.parametrize("formula_class_name", formula_sdct)
@pytest.mark.parametrize("term", ["M1", "E2", "V", "S"])
@pytest.mark.parametrize("sym_name", ["sym", "antisym"])
def test_formula_sdct(datak_Fe, formula_class_name, check_formula_output, term, sym_name):
    data = datak_Fe
    degen_groups = data.get_bands_in_range_groups(emin=-10, emax=30, degen_thresh=0.01)
    formula_class = getattr(frml_sdct, "Formula_" + formula_class_name)
    terms = {t: False for t in ["M1_terms", "E2_terms", "V_terms", "S_terms"]}
    terms[term + "_terms"] = True
    sym = {"sym": True, "antisym": False}[sym_name]
    formula = formula_class(data, sym=sym, **terms)
    value = {}
    for ik in range(data.nk):
        lst = []
        for n1 in degen_groups[ik]:
            for n2 in degen_groups[ik]:
                if n1 == n2:
                    continue
                inn1 = np.arange(n1[0], n1[1])
                inn2 = np.arange(n2[0], n2[1])
                lst.append(formula.trace_ln(ik, inn1, inn2))
        value[f"trace_ln_ik={ik}"] = np.array(lst)
    check_formula_output(value=value, filename=f"{formula_class_name}_{term}_{sym_name}")
