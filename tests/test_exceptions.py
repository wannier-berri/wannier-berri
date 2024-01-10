from wannierberri import __utility as util
import pytest
import numpy as np
from wannierberri import calculators as calc


def test_utility_str2bool():
    for v in "F", "f", "False", "false", "fAlSe", ".false.", " \n .False. \n":
        assert util.str2bool(v) is False
    for v in "True", "true", "TRuE", ".TrUe.", " True \n":
        assert util.str2bool(v) is True
    for v in ".true", "false.", "svas":
        with pytest.raises(ValueError, match=f"unrecognized value of bool parameter :`{v}`"):
            util.str2bool(v)


def test_utility_FFT():
    inp = np.random.random((5, 4, 3, 5)) * (1 + 2.j)
    axes = (0, 2)
    for fft in "fftw", "numpy", "FfTw", "nUmPY":
        util.FFT(inp, axes, fft=fft)
    for fft in "fft", "np", "idkwhat", "dummy":
        with pytest.raises(ValueError, match=f"unknown type of fft : {fft}"):
            util.FFT(inp, axes, fft=fft)


def test_utility_FFT_R_to_k():
    NKFFT = (4, 5, 6)
    iRvec = np.random.randint(100, size=(20, 3))
    num_wann = 5
    for lib in 'fftw', 'numpy', 'NuMpY', 'fftW':
        util.FFT_R_to_k(iRvec, NKFFT, num_wann, lib=lib)

    for lib in 'unknonw', 'fft', 'nump', 'np', "NP":
        with pytest.raises(AssertionError, match=f"fft lib '{lib.lower()}' is unknown/supported"):
            util.FFT_R_to_k(iRvec, NKFFT, num_wann, lib=lib)

    AAA_K = np.random.random((num_wann, num_wann) + NKFFT + (3, 3, 3))

    util.FFT_R_to_k(iRvec, NKFFT, num_wann, lib="numpy").transform(AAA_K)
    with pytest.raises(RuntimeError, match="FFT.transform should not be called for slow FT"):
        util.FFT_R_to_k(iRvec, NKFFT, num_wann, lib="slow").transform(AAA_K)


@pytest.mark.parametrize("ibands", [[5, 6], [4, 6, 7, 8]])
def test_TabulatorAll_fail(ibands):
    with pytest.raises(ValueError):
        calc.tabulate.TabulatorAll(
            {
                "Energy": calc.tabulate.Energy(),  # yes, in old implementation degen_thresh was applied to qunatities,
                # but not to energies
                "V": calc.tabulate.Velocity(ibands=ibands, print_comment=True),
            },
            ibands=[5, 6, 7, 8])


def test_Chiral_left_tab_static(check_run, system_Chiral_left):
    grid_param = {'NK': [10, 10, 4], 'NKFFT': [5, 5, 2]}
    param = dict(Efermi=np.array([-1, 0, 1]), tetra=False, kwargs_formula={"external_terms": False})
    system = system_Chiral_left

    calculators = {"AHC": calc.static.AHC(**param),
                "Morb": calc.static.Morb(**param)
                    }
    calculators["tabulate"] = calc.TabulatorAll(
        {
            "AHC": calc.static.AHC(**param, k_resolved=True),
            "Morb": calc.static.Morb(**param, k_resolved=True)
        },
        mode="path",
        ibands=(0, 1))

    with pytest.raises(ValueError):
        check_run(
                system,
                calculators,
                fout_name="berry_Chiral_static_tab",
                suffix="",
                grid_param=grid_param,
                parameters_K={
                    '_FF_antisym': True,
                    '_CCab_antisym': True
                },
                use_symmetry=False,
                do_not_compare=True
        )
