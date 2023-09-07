"""Test wberri.evaluatre_k() function"""
import os

from pytest import approx

import wannierberri as wberri
import numpy as np

from common import OUTPUT_DIR, REF_DIR


k=np.array([0.1,0.2,0.3])

quantities = ["energy","berry_curvature"]
formulae =  {   "ham" :wberri.formula.covariant.Hamiltonian,
                "vel" :wberri.formula.covariant.Velocity,
                "mass":wberri.formula.covariant.InvMass,
                "morb":wberri.formula.covariant.morb
            }



def test_evaluate_k_all(system_Fe_W90):
    result = wberri.evaluate_k(
                            system_Fe_W90,
                            k=k,
                            quantities=quantities,
                            formula=formulae,
                            param_formula={"morb":{"external_terms":False}},
                            iband=[4,5]
                        )
    result["kpoint"]=k
    np.savez_compressed(os.path.join(OUTPUT_DIR,"evaluate_k.npz"),**result)
    data_ref = np.load(os.path.join(REF_DIR,"evaluate_k.npz"))
    acc = 1e-8
    for key in result:
        assert (result[key] == approx(data_ref[key],abs=acc)
            ), "the result of evaluate_k for {key} is different from the reference data by {err} greater than the required accuracy {acc}".format(
                key=key, err=np.max(abs(result[key]-data_ref[key])),acc=acc)



def test_evaluate_k_1q(system_Fe_W90):
    data_ref = np.load(os.path.join(REF_DIR,"evaluate_k.npz"))
    for key in quantities:
        result = wberri.evaluate_k(
                            system_Fe_W90,
                            k=k,
                            quantities=[key],
                            param_formula={"morb":{"external_terms":False}},
                            iband=[4,5],
                            return_single_as_dict=False,
                        )
        acc = 1e-8
        assert (result == approx(data_ref[key],abs=acc)
            ), "the result of evaluate_k for {key} is different from the reference data by {err} greater than the required accuracy {acc}".format(
                key=key, err=np.max(abs(result[key]-data_ref[key])),acc=acc)


def test_evaluate_k_1f(system_Fe_W90):
    data_ref = np.load(os.path.join(REF_DIR,"evaluate_k.npz"))
    for key,form in formulae.items():
        result = wberri.evaluate_k(
                            system_Fe_W90,
                            k=k,
                            formula={key:form},
                            param_formula={"morb":{"external_terms":False}},
                            iband=[4,5],
                            return_single_as_dict=False,
                        )
        acc = 1e-8
        assert (result == approx(data_ref[key],abs=acc)
            ), "the result of evaluate_k for {key} is different from the reference data by {err} greater than the required accuracy {acc}".format(
                key=key, err=np.max(abs(result[key]-data_ref[key])),acc=acc)


def test_evaluate_k_hlp():
    result = wberri.evaluate_k()
    assert result is None
