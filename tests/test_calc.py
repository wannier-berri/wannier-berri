import wannierberri as wberri
from wannierberri.calculators import static
from wannierberri.formula import covariant as frml
from wannierberri.__factors import factor_ahc
from wannierberri.result import EnergyResult
from common import OUTPUT_DIR, REF_DIR
from common_comparers import error_message
import os, pytest


from common_systems import (
    Efermi_Fe,
    )

@pytest.fixture
def check_calculator(compare_any_result):

    def _inner(system,calc,name,dK=[0.1,0.2,0.3],NKFFT=[3,3,3], param_K={}, 
                precision=-1e-8,
                compare_zero=False,
                do_not_compare=False,
                result_type = EnergyResult
                ):
        grid = wberri.Grid(system, NKFFT=NKFFT, NK=1)
        data_K = wberri.data_K.get_data_k(system, dK=dK, grid=grid, **param_K)
        result = calc(data_K)
        filename = "calculator-"+name
        path_filename=os.path.join(OUTPUT_DIR,filename)
        result.save(path_filename)
        if do_not_compare:
            return result

        if compare_zero:
            result_ref = result * 0.
            path_filename_ref = "ZERO"
            assert precision > 0, "comparing with zero is possible only with absolute precision"
        else:
            path_filename_ref = os.path.join(REF_DIR, filename+".npz")
            result_ref = result_type(file_npz=path_filename_ref)
            maxval = result_ref._maxval_raw
            if precision is None:
                precision = max(maxval / 1E12, 1E-11)
            elif precision < 0:
                precision = max(maxval * abs(precision), 1E-11)
            err = (result - result_ref)._maxval_raw
            assert err < precision, error_message(
                fout_name, "", 0, err, path_filename, path_filename_ref, precision)
        return result
    return _inner



def test_AHC_fder(system_Fe_W90,check_calculator):

    class AHC1(static.StaticCalculator):
        def __init__(self,  **kwargs):
            self.Formula = frml.Omega
            self.fder = 1
            super().__init__(constant_factor=factor_ahc, **kwargs)

    class AHC2(static.StaticCalculator):
        def __init__(self, **kwargs):
            self.Formula = frml.Omega
            self.fder = 1
            super().__init__(constant_factor=factor_ahc, **kwargs)

    param = {"Efermi":Efermi_Fe}
    for name,calc in [("AHC0",static.AHC), ("AHC1", AHC1), ("AHC2", AHC2)]:
        check_calculator(system_Fe_W90,calc(**param),"Fe-"+name)