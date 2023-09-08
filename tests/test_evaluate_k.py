"""Test wberri.evaluatre_k() function"""
import os

from pytest import approx

import wannierberri as wberri
from wannierberri import calculators as calc
import numpy as np
from common import OUTPUT_DIR, REF_DIR
from common_systems import Efermi_Fe


k=np.array([0.1,0.2,0.3])

quantities = ["energy","berry_curvature"]
formulae =  {   "ham" :wberri.formula.covariant.Hamiltonian,
                "vel" :wberri.formula.covariant.Velocity,
                "mass":wberri.formula.covariant.InvMass,
                "morb":wberri.formula.covariant.morb
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
    calculators = {k:cal(**param) for k,cal in calculators_Fe.items()}
    result = wberri.evaluate_k(
                            system_Fe_W90,
                            k=k,
                            quantities=quantities,
                            formula=formulae,
                            calculators=calculators,
                            param_formula={"morb":{"external_terms":False}},
                            iband=[4,5]
                        )
    result["kpoint"]=k

    result_ref = np.load(os.path.join(REF_DIR,"evaluate_k.npz"))
    acc = 1e-8
    for key,res in result.items():
        if isinstance(res,np.ndarray):
            data=res
            data_ref=result_ref[key]
        elif isinstance(res,wberri.result.Result):
            result[key]=res.as_dict()
            continue # uncomment to generate a new refernce file
            data=res.data
            data_ref=result_ref[key].data
        else:
            raise ValueError(f"Uncomparable type of result : {type(res)}")
        assert (data == approx(data_ref,abs=acc)
                ), "the result of evaluate_k for {key} is different from the reference data by {err} greater than the required accuracy {acc}".format(
                    key=key, err=np.max(abs(data-data_ref)),acc=acc)
    np.savez_compressed(os.path.join(OUTPUT_DIR,"evaluate_k.npz"), **result)




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


def test_evaluate_fail(system_Fe_W90):
    with pytest.raises(ValueError):
        wberri.evaluate_k(  system_Fe_W90,  k=k, quantities=["abracadabra"], )
    with pytest.raises(ValueError):
        wberri.evaluate_k(  system_Fe_W90,  k=k, quantities=["abracadabra"], formula={"abracadabra":None})
