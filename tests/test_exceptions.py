from wannierberri import utility as util
import pytest
import numpy as np
import wannierberri as wberri
from .common_systems import create_files_tb
from wannierberri import calculators as calc
from wannierberri.fourier.fft import FFT_R_to_k, execute_fft


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
        execute_fft(inp, axes, fftlib=fft)
    for fft in "fftlib", "np", "idkwhat", "dummy":
        with pytest.raises(ValueError, match=f"unknown type of fftlib : {fft}"):
            execute_fft(inp, axes, fftlib=fft)


def test_utility_FFT_R_to_k():
    NKFFT = (4, 5, 6)
    iRvec = np.random.randint(100, size=(20, 3))
    num_wann = 5
    for lib in 'fftw', 'numpy', 'NuMpY', 'fftW':
        FFT_R_to_k(iRvec, NKFFT, num_wann, fftlib=lib)

    for lib in 'unknonw', 'fftlib', 'nump', 'np', "NP":
        with pytest.raises(AssertionError, match=f"fftlib '{lib.lower()}' is unknown/not supported"):
            FFT_R_to_k(iRvec, NKFFT, num_wann, fftlib=lib)

    shape = (num_wann, num_wann) + NKFFT + (3, 3, 3)
    AAA_K = np.random.random(shape) + 1j * np.random.random(shape)

    FFT_R_to_k(iRvec, NKFFT, num_wann, fftlib="numpy").transform(AAA_K)
    with pytest.raises(RuntimeError, match="FFT.transform should not be called for slow FT"):
        FFT_R_to_k(iRvec, NKFFT, num_wann, fftlib="slow").transform(AAA_K)


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
            use_symmetry=False,
            do_not_compare=True
        )


def test_TBmodels_fail():
    try:
        import tbmodels
        tbmodels  # just to avoid F401
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import tbmodels")
    with pytest.raises(ValueError):
        wberri.system.System_TBmodels(wberri.models.Haldane_tbm(delta=0.2, hop1=-1.0, hop2=0.15), spin=True)


def test_morb_fail():
    try:
        import tbmodels
        tbmodels  # just to avoid F401
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import tbmodels")
    with pytest.raises(ValueError):
        wberri.system.System_TBmodels(wberri.models.Haldane_tbm(delta=0.2, hop1=-1.0, hop2=0.15), spin=True)



def test_system_GaAs_tb_morb_fail():
    """Create system for GaAs using _tb.dat data"""

    seedname = create_files_tb(dir="GaAs_Wannier90", file="GaAs_tb.dat")
    for tag in 'spin', 'morb', 'SHCqiao', 'SHCryoo':
        with pytest.raises(ValueError):
            wberri.system.System_tb(seedname, **{tag: True})


def test_wrong_mat_fail():
    model_pythtb_Haldane = wberri.models.Haldane_ptb(delta=0.2, hop1=-1.0, hop2=0.15)
    system = wberri.system.System_PythTB(model_pythtb_Haldane)
    system.set_R_mat('abracadabra', system.get_R_mat('Ham') * 4)
    with pytest.raises(NotImplementedError, match="symmetrization of matrices"):
        system.symmetrize(
            positions=np.array([[1 / 3, 1. / 3, 0], [2. / 3., 2. / 3., 0]]),
            atom_name=['one', 'two'],
            proj=['one:s', 'two:s'],
            soc=False,)


def test_result_fit_fail_transform():
    from wannierberri.result.energyresult import EnergyResult
    from wannierberri.symmetry.point_symmetry import transform_ident, transform_odd, transform_odd_conj, transform_trans, transform_odd_trans_021, transform_odd_trans_102
    transform_list = [transform_ident, transform_odd, transform_odd_conj, transform_trans, transform_odd_trans_021, transform_odd_trans_102]
    data = np.random.random((10, 3, 3))
    Energies = [np.linspace(-1, 1, 10)]
    for i, trans1 in enumerate(transform_list):
        for j, trans2 in enumerate(transform_list):
            r1 = EnergyResult(data=data, Energies=Energies, transformTR=trans1, transformInv=transform_ident)
            r2 = EnergyResult(data=data, Energies=Energies, transformTR=trans2, transformInv=transform_ident)
            if i == j:
                r1 + r2  # should not raise
            else:
                with pytest.raises(RuntimeError, match="Results do not fit, cannot add them : TR transformations do not match"):
                    r1 + r2
            r1 = EnergyResult(data=data, Energies=Energies, transformInv=trans1, transformTR=transform_ident)
            r2 = EnergyResult(data=data, Energies=Energies, transformInv=trans2, transformTR=transform_ident)
            if i == j:
                r1 + r2  # should not raiseInv      else:
            else:
                with pytest.raises(RuntimeError, match="Results do not fit, cannot add them : Inv transformations do not match"):
                    r1 + r2


def test_result_fit_fail_rank():
    from wannierberri.result.energyresult import EnergyResult
    data1 = np.random.random((10, 3, 3))
    Energies = [np.linspace(-1, 1, 10)]
    data2 = np.random.random((10, 3, 3, 3))

    r1 = EnergyResult(data=data1, Energies=Energies)
    r2 = EnergyResult(data=data2, Energies=Energies)

    with pytest.raises(RuntimeError, match="Results do not fit, cannot add them : Ranks do not match"):
        r1 + r2


def test_result_fit_fail_energies():
    from wannierberri.result.energyresult import EnergyResult
    data1 = np.random.random((10, 3, 3))
    Energies1 = [np.linspace(-1, 1, 10)]
    data2 = np.random.random((10, 3, 3))
    Energies2 = [np.linspace(-1, 2, 10)]

    r1 = EnergyResult(data=data1, Energies=Energies1)
    r2 = EnergyResult(data=data2, Energies=Energies2)

    with pytest.raises(RuntimeError, match="Results do not fit, cannot add them : Energy levels do not match"):
        r1 + r2


def test_result_fit_fail_smoothers():
    from wannierberri.result.energyresult import EnergyResult
    from wannierberri.smoother import (VoidSmoother, FermiDiracSmoother, GaussianSmoother)
    Energies = [np.linspace(-1, 1, 10)]
    smoothers = [None, VoidSmoother(), FermiDiracSmoother(Energies[0], 1000.0), FermiDiracSmoother(Energies[0], 100.0),
                 GaussianSmoother(Energies[0], 0.1), GaussianSmoother(Energies[0], 0.2)]

    data1 = np.random.random((10, 3, 3))
    for i, smoother1 in enumerate(smoothers):
        for j, smoother2 in enumerate(smoothers):
            print(f"Testing smoothers: {i} {smoother1} vs {j} {smoother2}")
            r1 = EnergyResult(data=data1, Energies=Energies, smoothers=[smoother1])
            r2 = EnergyResult(data=data1, Energies=Energies, smoothers=[smoother2])
            if i == j or (i in [0, 1] and j in [0, 1]):  # None smoother is the same as VoidSmoother
                r1 + r2  # should not raise
            else:
                with pytest.raises(RuntimeError, match="Results do not fit, cannot add them : Smoothers do not match"):
                    r1 + r2


def test_deprecated_calculator():
    with pytest.raises(RuntimeError, match="This calculator is deprecated"):
        calc.static.AHC_Zeeman_spin()
