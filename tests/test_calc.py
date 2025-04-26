import wannierberri as wberri
from wannierberri.calculators import static
from wannierberri.formula import covariant as frml
from wannierberri.result import EnergyResult, KBandResult
from .common import OUTPUT_DIR, REF_DIR
from .common_comparers import error_message
import numpy as np
import os
import pytest


from .common_systems import Efermi_Fe


@pytest.fixture
def check_calculator(compare_any_result):
    """

    """

    def _inner(system, calc, name, dK=[0.1, 0.2, 0.3], NKFFT=[3, 3, 3], param_K={},
               precision=-1e-8,
               compare_zero=False,
               do_not_compare=False,
               result_type=EnergyResult,
               factor=1
                ):
        grid = wberri.Grid(system, NKFFT=NKFFT, NKdiv=1)
        data_K = wberri.data_K.get_data_k(system, dK=dK, grid=grid, **param_K)
        result = calc(data_K) * factor

        filename = "calculator-" + name
        path_filename = os.path.join(OUTPUT_DIR, filename)
        result.save(path_filename)
        if do_not_compare:
            return result

        if compare_zero:
            result_ref = result * 0.
            path_filename_ref = "ZERO"
            assert precision > 0, "comparing with zero is possible only with absolute precision"
        else:
            path_filename_ref = os.path.join(REF_DIR, 'calculators', filename + ".npz")
            result_ref = result_type(file_npz=path_filename_ref)
        maxval = result_ref._maxval_raw
        if precision is None:
            precision = max(maxval / 1E12, 1E-11)
        elif precision < 0:
            precision = max(maxval * abs(precision), 1E-11)
        err = (result - result_ref)._maxval_raw
        assert err < precision, error_message(
            name, "", 0, err, path_filename, path_filename_ref, precision)
        return result
    return _inner



def test_calc_fder(system_Fe_W90, check_calculator):

    param = dict(Formula=frml.Identity, Efermi=Efermi_Fe, tetra=False)
    for fder in range(4):
        calc = static.StaticCalculator(**param, fder=fder)
        name = f"Fe-ident-fder={fder}"
        check_calculator(system_Fe_W90, calc, name)


def test_tabulator_mul(system_Fe_W90, check_calculator):
    calc = wberri.calculators.tabulate.Energy()
    name = "Fe-tab-energy"
    check_calculator(system_Fe_W90, calc, name, factor=5, result_type=KBandResult)


def test_tab_fit(system_Haldane_PythTB):
    system = system_Haldane_PythTB
    dK = [0.1, 0.2, 0.3]
    grid = wberri.Grid(system, NKFFT=[3, 3, 1], NKdiv=1)
    data_K = wberri.data_K.get_data_k(system, dK=dK, grid=grid)
    morb = wberri.calculators.tabulate.OrbitalMoment
    berry = wberri.calculators.tabulate.BerryCurvature
    noext = dict(kwargs_formula={"external_terms": False})
    calculators = [[cal(ibands=ib, **noext) for ib in ([0,], [0, 1])] for cal in (berry, morb)]
    results = [[cal[b](data_K) for b in range(2)] for cal in calculators]
    for ical1, ib1 in wberri.utility.iterate_nd((2, 2)):
        for ical2, ib2 in wberri.utility.iterate_nd((2, 2)):
            r1 = results[ical1][ib1]
            r2 = results[ical2][ib2]
            check = r1.fit(r2)
            assert check == ((ib1 == ib2)), f"{r1.nband}, {r2.nband}, {check}"

    for res in results:
        r1 = res[0]
        r2 = res[1].select_bands((0,))
        assert r1.fit(r2), f"{r1.nband}, {r2.nband}, {r1.fit(r2)}"
        assert r1.data == pytest.approx(r2.data)


def test_BD_trace(system_Haldane_PythTB):
    system = system_Haldane_PythTB
    dK = [0.1, 0.2, 0.3]
    grid = wberri.Grid(system, NKFFT=[3, 3, 1], NKdiv=1)
    data_K = wberri.data_K.get_data_k(system, dK=dK, grid=grid)
    noext = dict(kwargs_formula={"external_terms": False})
    bd = wberri.calculators.tabulate.DerBerryCurvature(**noext)
    result = bd(data_K)
    trace = result.get_component("trace")
    print(f"shape of trace {trace.shape}")
    assert (np.max(abs(trace))) < 1e-10


@pytest.fixture
def check_save_result():
    def _inner(system, calc, result_type, filename="dummy"):
        grid = wberri.Grid(system, NKFFT=3, NK=5)
        dK = np.random.random(3)
        data_K = wberri.data_K.get_data_k(system, dK=dK, grid=grid)
        result = calc(data_K)
        path_filename = os.path.join(OUTPUT_DIR, filename)
        result.save(path_filename)
        result_read = result_type(file_npz=path_filename + ".npz")
        assert result.data.shape == result_read.data.shape
        assert str(result.transformTR) == str(result_read.transformTR)
        assert str(result.transformInv) == str(result_read.transformInv)
    return _inner


def test_SDCT(system_random_load_bare, check_calculator):

    from .test_run import Efermi_GaAs, calculators_SDCT
    param = {'Efermi': Efermi_GaAs,
             'omega': np.linspace(0.0, 7, 8),
             'kBT': 0.05, 'smr_fixed_width': 0.1,
             }

    for key, calculator in calculators_SDCT.items():
        for term in ["M1", "E2", "V", "all"]:
            param_terms = {f"{t}_terms": (t == "all") for t in ["M1", "E2", "V"]}
            if term != "all":
                param_terms[f"{term}_terms"] = True
            name = f"random-{key}-{term}_terms"
            print(name)
            calc = calculator(**param_terms, **param)
            check_calculator(system_random_load_bare, calc, name, do_not_compare=False)




def test_save_KBandResult(system_Haldane_PythTB, check_save_result):
    calc = wberri.calculators.tabulate.Energy()
    check_save_result(system_Haldane_PythTB, calc, result_type=KBandResult)


def test_save_KBandResult_add(system_Haldane_PythTB, check_calculator):
    calc1 = wberri.calculators.tabulate.Energy()
    calc2 = wberri.calculators.tabulate.Velocity()
    calc3 = wberri.calculators.tabulate.BerryCurvature(kwargs_formula={"external_terms": False})
    res1 = check_calculator(system_Haldane_PythTB, calc1, "dummy", result_type=KBandResult, do_not_compare=True)
    res2 = check_calculator(system_Haldane_PythTB, calc2, "dummy", result_type=KBandResult, do_not_compare=True)
    res3 = check_calculator(system_Haldane_PythTB, calc3, "dummy", result_type=KBandResult, do_not_compare=True)
    assert res1.fit(res2) is False
    assert res2.fit(res3) is False
    assert res3.fit(res1) is False
    with pytest.raises(AssertionError):
        res1 + res2
    with pytest.raises(AssertionError):
        res1 - res2


def test_save_EnergyResult(system_Haldane_PythTB, check_save_result):
    param = dict(Formula=frml.Identity, Efermi=Efermi_Fe, tetra=False, fder=0)
    calc = static.StaticCalculator(**param)
    check_save_result(system_Haldane_PythTB, calc, result_type=EnergyResult)


def test_get_transform():
    from wannierberri.symmetry.point_symmetry import transform_from_dict
    assert transform_from_dict({"asdasd": "aasd"}, "transformTR") is None
    assert transform_from_dict({"transformTR": np.array("transform()", dtype=object)}, "transformTR") is None
    assert transform_from_dict({"transformTR": None}, "transformTR") is None
    with pytest.raises(TypeError):
        transform_from_dict({"transformTR": np.array({"x": 5}, dtype=object)}, "transformTR")
    with pytest.raises(ValueError):
        transform_from_dict({"transformTR": np.zeros(5)}, "transformTR")
    with pytest.raises(ValueError):
        transform_from_dict({"transformTR": np.array((1, 2, 3), dtype=object)}, "transformTR")
