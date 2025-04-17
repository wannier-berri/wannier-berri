"""Test wberri.evaluatre_k() function"""

import os

import pytest

import wannierberri as wberri
from wannierberri import calculators as calc
import numpy as np
from .common import OUTPUT_DIR, REF_DIR
from .common_systems import Efermi_Fe


k = np.array([0.1, 0.2, 0.3])

quantities = ["energy", "berry_curvature"]
formulae = {"ham": wberri.formula.covariant.Hamiltonian,
            "vel": wberri.formula.covariant.Velocity,
            "mass": wberri.formula.covariant.InvMass,
            "morb": wberri.formula.covariant.morb
            }

calculators_Fe = {
    'ahc': calc.static.AHC,
    'conductivity_ohmic': calc.static.Ohmic_FermiSea,
    'conductivity_ohmic_fsurf': calc.static.Ohmic_FermiSurf,
    'Morb': calc.static.Morb,
    'dos': calc.static.DOS,
    'cumdos': calc.static.CumDOS,
}



def test_evaluate_k_all(system_Fe_W90):
    param = dict(Efermi=Efermi_Fe)
    calculators = {k: cal(**param) for k, cal in calculators_Fe.items()}
    result = wberri.evaluate_k(
        system_Fe_W90,
        k=k,
        quantities=quantities,
        formula=formulae,
        calculators=calculators,
        param_formula={"morb": {"external_terms": False}},
        iband=[4, 5]
    )
    result["kpoint"] = k


    result_ref = np.load(os.path.join(REF_DIR, "evaluate_k.npz"))
    acc = 1e-8
    for key, res in result.items():
        if isinstance(res, np.ndarray):
            data = res
        elif isinstance(res, wberri.result.Result):
            result[key] = res.data
            data = res.data
        else:
            raise ValueError(f"Uncomparable type of result : {type(res)}")
        # continue # uncomment to generate a new reference file
        data_ref = result_ref[key]

        assert data == pytest.approx(data_ref, rel=acc), (
            f"the result of evaluate_k for {key} is different from the reference data "
            f"by {np.max(abs(data - data_ref))} "
            f"greater than the required accuracy {acc}"
        )
    np.savez_compressed(os.path.join(OUTPUT_DIR, "evaluate_k.npz"), **result)



def test_evaluate_k_all_1band(system_Fe_W90):
    result = wberri.evaluate_k(
        system_Fe_W90,
        k=k,
        quantities=quantities,
        param_formula={"morb": {"external_terms": False}},
        iband=4
    )
    result["kpoint"] = k

    result_ref = np.load(os.path.join(REF_DIR, "evaluate_k.npz"))
    acc = 1e-8
    for key, res in result.items():
        if isinstance(res, np.ndarray):
            data = res
        elif isinstance(res, wberri.result.Result):
            result[key] = res.data
            data = res.data
        else:
            raise ValueError(f"Uncomparable type of result : {type(res)}")
        # continue # uncomment to generate a new reference file
        data_ref = result_ref[key]
        assert data[0] == pytest.approx(data_ref[0], rel=acc), (
            f"the result of evaluate_k for {key} is different from the reference data"
            f"by {np.max(abs(data - data_ref))}, greater than the required accuracy {acc}")
    # np.savez_compressed(os.path.join(OUTPUT_DIR, "evaluate_k.npz"), **result)




def test_evaluate_k_1q(system_Fe_W90):
    data_ref = np.load(os.path.join(REF_DIR, "evaluate_k.npz"))
    for key in quantities:
        result = wberri.evaluate_k(
            system_Fe_W90,
            k=k,
            quantities=[key],
            param_formula={"morb": {"external_terms": False}},
            iband=[4, 5],
            return_single_as_dict=False,
        )
        acc = 1e-8
        assert result == pytest.approx(data_ref[key], rel=acc), (
            f"the result of evaluate_k for {key} is different from the reference data "
            f"by {np.max(abs(result - data_ref[key]))} greater than the required accuracy {acc}")


def test_evaluate_k_1f(system_Fe_W90):
    data_ref = np.load(os.path.join(REF_DIR, "evaluate_k.npz"))
    for key, form in formulae.items():
        result = wberri.evaluate_k(
            system_Fe_W90,
            k=k,
            formula={key: form},
            param_formula={"morb": {"external_terms": False}},
            iband=[4, 5],
            return_single_as_dict=False,
        )
        acc = 1e-8
        assert result == pytest.approx(data_ref[key], rel=acc), (
            f"the result of evaluate_k for {key} is different from the reference data "
            f"by {np.max(abs(result - data_ref[key]))} greater than the required accuracy {acc}")


def test_evaluate_k_hlp():
    result = wberri.evaluate_k()
    assert result is None


def test_evaluate_fail(system_Fe_W90):
    with pytest.raises(ValueError):
        wberri.evaluate_k(system_Fe_W90, k=k, quantities=["abracadabra"], )
    with pytest.raises(ValueError):
        wberri.evaluate_k(system_Fe_W90, k=k, quantities=["ham"], formula={"ham": None})
    with pytest.raises(ValueError):
        wberri.evaluate_k(system_Fe_W90, k=k, quantities=["ham"], calculators={"ham": None})
    with pytest.raises(ValueError):
        wberri.evaluate_k(system_Fe_W90, k=k, calculators={"abracadabra": None}, formula={"abracadabra": None})
