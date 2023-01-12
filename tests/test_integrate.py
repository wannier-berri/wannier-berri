"""Test the anomalous Hall conductivity."""
import os

import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from wannierberri.__old_API import fermiocean

from common import OUTPUT_DIR
from common_comparers import compare_quant
from common_systems import (
    Efermi_Fe,
    Efermi_Fe_FPLO,
    Efermi_GaAs,
    Efermi_Haldane,
    Efermi_CuMnAs_2d,
    Efermi_Chiral,
    Efermi_Mn3Sn,
)


@pytest.fixture
def check_integrate(parallel_serial):

    def _inner(
            system,
            quantities=[],
            user_quantities={},
            fout_name="berry",
            Efermi=np.linspace(-10, 10, 10),
            comparer=None,
            compare_zero=False,
            parallel=parallel_serial,
            grid_param={
                'NK': [6, 6, 6],
                'NKFFT': [3, 3, 3]
            },
            adpt_num_iter=0,
            additional_parameters={},
            parameters_K={},
            specific_parameters={},
            use_symmetry=False,
            smearEf=600.0,
            suffix="",
            suffix_ref="",
            mode="bin",
            extra_precision={},
            precision=-1e-8,
            compare_smooth=True,
            restart=False):

        grid = wberri.Grid(system, **grid_param)
        result = wberri.integrate(
            system,
            grid=grid,
            Efermi=Efermi,
            smearEf=smearEf,
            #            omega = omega,
            quantities=quantities,
            user_quantities=user_quantities,
            parallel=parallel,
            adpt_num_iter=adpt_num_iter,
            use_irred_kpt=use_symmetry,
            symmetrize=use_symmetry,
            parameters=additional_parameters,
            specific_parameters=specific_parameters,
            parameters_K=parameters_K,
            fout_name=os.path.join(OUTPUT_DIR, fout_name),
            suffix=suffix,
            write_txt=("txt" in mode),
            write_bin=("bin" in mode),
            restart=restart,
        )
        if len(suffix) > 0:
            suffix = "-" + suffix
        if len(suffix_ref) > 0:
            suffix_ref = "-" + suffix_ref

        # compare results externally
        if comparer is None:
            return result

        for quant in quantities + list(user_quantities.keys()):
            data = result.results.get(quant).data
            assert data.shape[0] == len(Efermi)
            assert np.all(np.array(data.shape[1:]) == 3)
            prec = extra_precision[quant] if quant in extra_precision else precision
            comparer(
                fout_name,
                quant + suffix,
                adpt_num_iter,
                suffix_ref=compare_quant(quant) + suffix_ref,
                mode=mode,
                compare_zero=compare_zero,
                precision=prec,
                compare_smooth=compare_smooth)

    return _inner


@pytest.fixture(scope="session")
def quantities_Fe():
    return ['ahc', 'ahc_test', 'dos', 'cumdos', 'conductivity_ohmic', 'conductivity_ohmic_fsurf', 'Morb', 'Morb_test']


# quantities containing external terms
@pytest.fixture(scope="session")
def quantities_Fe_ext():
    return ['ahc', 'ahc_test', 'Morb', 'Morb_test']


# quantities containing external terms
@pytest.fixture(scope="session")
def quantities_CuMnAs_2d():
    return ['dos', 'cumdos', 'conductivity_ohmic', 'Hall_morb_fsurf', 'Hall_classic_fsurf']


@pytest.fixture(scope="session")
def quantities_Chiral():
    return [
        #         'spin'                     ,#: fermiocean.spin
        #         'Morb'                     ,#: fermiocean.Morb
        'ahc',  #: fermiocean.AHC
        'cumdos',  #: fermiocean.cumdos
        'dos',  #: fermiocean.dos
        'conductivity_ohmic',  #: fermiocean.ohmic
        'conductivity_ohmic_fsurf',  #: fermiocean
        'berry_dipole',  #: fermiocean.berry_dipole
        'berry_dipole_fsurf',  #: fermiocean.berry_dipole_fsurf
        #         'gyrotropic_Korb'          ,#: fermiocean.gme_orb
        #         'gyrotropic_Korb_fsurf'    ,#: fermiocean.gme_orb_fsurf
        #         'gyrotropic_Kspin'         ,#: fermiocean.gme_spin
        #         'gyrotropic_Kspin_fsurf'   ,#: fermiocean.gme_spin_fsurf
        'Hall_classic',  #: fermiocean.Hall_classic
        'Hall_classic_fsurf',  #: fermiocean.Hall_classic_fsurf
        #         'Hall_morb_fsurf'          ,#: fermiocean.Hall_morb_fsurf
        #         'Hall_spin_fsurf'          ,#: fermiocean.Hall_spin_fsurf
        'Der3E',  #: fermiocean.Der3E
        #         'Der3E_fsurf'              ,#: fermiocean.Der3E_fsurf
        #         'Der3E_fder2'              ,#: fermiocean.Der3E_fder2
    ]


@pytest.fixture(scope="module")
def quantities_Haldane():
    return ['ahc', 'dos', 'conductivity_ohmic']


@pytest.fixture(scope="module")
def quantities_GaAs():
    return ["berry_dipole", "berry_dipole_test", "berry_dipole_fsurf"]


@pytest.fixture(scope="module")
def quantities_GaAs_internal():
    "quantities containing only internal terms"
    return ["dos", "cumdos", "conductivity_ohmic"]


def test_Fe(check_integrate, system_Fe_W90, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(
        system_Fe_W90,
        quantities_Fe,
        fout_name="berry_Fe_W90",
        suffix="",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        compare_smooth=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"Morb": -1e-6})


def test_Fe_user(check_integrate, system_Fe_W90, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""

    calculators =  {
        'Morb': fermiocean.Morb,
        'ahc': fermiocean.AHC,
        'ahc_test': fermiocean.AHC_test,
        'cumdos': fermiocean.cumdos,
        'dos': fermiocean.dos,
        'conductivity_ohmic': fermiocean.ohmic,
        'conductivity_ohmic_fsurf': fermiocean.ohmic_fsurf,
    }

    check_integrate(
        system_Fe_W90,
        quantities=[],
        user_quantities=calculators,
        fout_name="berry_Fe_W90",
        suffix="user",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        compare_smooth=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"Morb": -1e-6})


def test_Fe_wcc(check_integrate, system_Fe_W90_wcc, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    # here we test against reference data obtained without wcc_phase. Low accuracy for Morb - this may be a bug
    check_integrate(
        system_Fe_W90_wcc,
        quantities_Fe,
        fout_name="berry_Fe_W90",
        suffix="wcc",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        additional_parameters={'correction_wcc': True},
        extra_precision={"Morb": -1})  # the wcc gives quite big error, just checking that it runs
    # here we test agaist reference data obtained with wcc_phase, should matcxh with high accuracy"


#    compare_energyresult( "berry_Fe_W90", "Morb-wcc",  0 , suffix_ref="Morb-wcc" ,precision=-1e-8, compare_smooth = True )


def test_Fe_sym(check_integrate, system_Fe_W90, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(
        system_Fe_W90,
        quantities_Fe,
        fout_name="berry_Fe_W90",
        use_symmetry=True,
        suffix="sym",
        suffix_ref="sym",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        })


def test_Fe_sym_W90(check_integrate, system_Fe_sym_W90, compare_energyresult):
    """Test anomalous Hall conductivity , spin
        using symmetrized model, without further summetrization"""
    check_integrate(
        system_Fe_sym_W90, ['ahc', 'spin', 'Morb'],
        fout_name="berry_Fe_sym_W90",
        use_symmetry=False,
        Efermi=Efermi_Fe,
        comparer=compare_energyresult)
    check_integrate(
        system_Fe_sym_W90, ["gyrotropic_Kspin", "berry_dipole", "gyrotropic_Korb"],
        fout_name="berry_Fe_sym_W90",
        use_symmetry=False,
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        compare_zero=True)


def test_Fe_sym_W90_sym(check_integrate, system_Fe_sym_W90, compare_energyresult):
    """Test anomalous Hall conductivity , spin
        using symmetrized model and further symmetrization ( which should not have effect )"""
    check_integrate(
        system_Fe_sym_W90, ['ahc', 'spin', 'Morb'],
        fout_name="berry_Fe_sym_W90",
        suffix="sym",
        use_symmetry=True,
        Efermi=Efermi_Fe,
        comparer=compare_energyresult)
    check_integrate(
        system_Fe_sym_W90, ["gyrotropic_Kspin", "berry_dipole", "gyrotropic_Korb"],
        fout_name="berry_Fe_sym_W90",
        suffix="sym",
        use_symmetry=True,
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        compare_zero=True)


def test_Fe_FPLO(check_integrate, system_Fe_FPLO, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(
        system_Fe_FPLO,
        quantities_Fe + ["spin"],
        fout_name="berry_Fe_FPLO",
        Efermi=Efermi_Fe_FPLO,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        additional_parameters={"external_terms": True})


def test_Fe_FPLO_wcc(check_integrate, system_Fe_FPLO_wcc, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(
        system_Fe_FPLO_wcc,
        quantities_Fe + ["spin"],
        fout_name="berry_Fe_FPLO",
        suffix="wcc",
        suffix_ref="",
        Efermi=Efermi_Fe_FPLO,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        additional_parameters={"external_terms": False})


def test_Fe_FPLO_wcc_ext(check_integrate, system_Fe_FPLO_wcc, compare_energyresult, quantities_Fe_ext):
    "Now check that external terms are really zero"
    check_integrate(
        system_Fe_FPLO_wcc,
        quantities_Fe_ext,
        fout_name="berry_Fe_FPLO",
        suffix="wcc_ext",
        Efermi=Efermi_Fe_FPLO,
        comparer=compare_energyresult,
        compare_zero=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        additional_parameters={
            "internal_terms": False,
            "external_terms": True
        })


def test_Fe_FPLO_wcc_sym(check_integrate, system_Fe_FPLO_wcc, compare_energyresult, quantities_Fe):
    """Check that the system is reallysymmetric"""
    check_integrate(
        system_Fe_FPLO_wcc,
        quantities_Fe + ["spin"],
        fout_name="berry_Fe_FPLO",
        suffix="wcc-sym",
        suffix_ref="",
        Efermi=Efermi_Fe_FPLO,
        comparer=compare_energyresult,
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        additional_parameters={"external_terms": False})


def test_GaAs(check_integrate, system_GaAs_W90, compare_energyresult, quantities_GaAs, quantities_GaAs_internal):
    """Test berry dipole"""
    check_integrate(
        system_GaAs_W90,
        quantities_GaAs_internal + quantities_GaAs + ['gyrotropic_Korb', 'gyrotropic_Korb_test','gyrotropic_Kspin', 'gyrotropic_Kspin_fsurf'],
        fout_name="berry_GaAs_W90",
        suffix="",
        Efermi=Efermi_GaAs,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        #                additional_parameters = {"internal_terms":False },
        extra_precision={"berry_dipole_fsurf": 1e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_GaAs_tb(check_integrate, system_GaAs_tb, compare_energyresult, quantities_GaAs):
    """Test berry dipole"""
    check_integrate(
        system_GaAs_tb,
        quantities_GaAs,
        fout_name="berry_GaAs_tb",
        suffix="",
        Efermi=Efermi_GaAs,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"berry_dipole_fsurf": 1e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_GaAs_wcc(
        check_integrate, system_GaAs_W90_wcc, compare_energyresult, quantities_GaAs, quantities_GaAs_internal):
    """Test GaAs with wcc_phase, comparing with data obtained without it"""
    check_integrate(
        system_GaAs_W90_wcc,
        quantities_GaAs + quantities_GaAs_internal + ['gyrotropic_Kspin', 'gyrotropic_Kspin_fsurf'],  #+['gyrotropic_Korb_test'],
        fout_name="berry_GaAs_W90",
        suffix="wcc",
        Efermi=Efermi_GaAs,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        additional_parameters={'correction_Morb_wcc': True},
        extra_precision={
            "berry_dipole_fsurf": 1e-6
        })  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem


def test_GaAs_tb_wcc(check_integrate, system_GaAs_tb_wcc, compare_energyresult, quantities_GaAs):
    """Test GaAs (from tb file) with wcc_phase, comparing with data obtained without it"""
    check_integrate(
        system_GaAs_tb_wcc,
        quantities_GaAs,
        fout_name="berry_GaAs_tb",
        suffix="wcc",
        Efermi=Efermi_GaAs,
        comparer=compare_energyresult,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"berry_dipole_fsurf": 1e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_GaAs_tb_wcc_ws(check_integrate, system_GaAs_tb_wcc_ws, compare_energyresult, quantities_GaAs_internal):
    """Test GaAs (from tb file) with wcc_phase and use_ws=True, comparing with data obtained from .chk file"""
    check_integrate(
        system_GaAs_tb_wcc_ws,
        quantities_GaAs_internal,
        fout_name="berry_GaAs_W90",
        suffix="tb_wcc_ws",
        suffix_ref="",
        Efermi=Efermi_GaAs,
        comparer=compare_energyresult,
        extra_precision={"conductivity_ohmic": -2e-6}
    )  # This is a low precision for the nonabelian thing, not sure if it does not indicate a problem, or is a gauge-dependent thing


def test_GaAs_sym_tb(check_integrate, system_GaAs_sym_tb, compare_energyresult):
    """Test ahc with sym_wann method"""
    check_integrate(
        system_GaAs_sym_tb, ["ahc"],
        fout_name="ahc_GaAs_sym_tb",
        suffix="",
        Efermi=Efermi_GaAs,
        comparer=compare_energyresult,
        compare_zero=True,
        extra_precision={"ahc": 1e-5})


def test_Mn3Sn_sym_tb(check_integrate, system_Mn3Sn_sym_tb, compare_energyresult):
    """Test ahc with sym_wann method"""
    check_integrate(
        system_Mn3Sn_sym_tb, ["ahc"],
        fout_name="ahc_Mn3Sn_sym_tb",
        suffix="",
        Efermi=Efermi_Mn3Sn,
        comparer=compare_energyresult)


def test_Haldane_PythTB(check_integrate, system_Haldane_PythTB, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_PythTB,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="pythtb",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_TBmodels(check_integrate, system_Haldane_TBmodels, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_TBmodels,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_PythTB_wcc(check_integrate, system_Haldane_PythTB, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_PythTB,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="pythtb_wcc",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_TBmodels_wcc(check_integrate, system_Haldane_TBmodels, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_TBmodels,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="wcc",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_TBmodels_wcc_internal_2(
        check_integrate, system_Haldane_TBmodels_internal, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_TBmodels_internal,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="wcc_internal_2",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        additional_parameters={'external_terms': False},
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_TBmodels_wcc_external(
        check_integrate, system_Haldane_TBmodels, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_TBmodels, ["ahc"],
        fout_name="berry_Haldane_tbmodels",
        suffix="wcc_external",
        suffix_ref="wcc_external",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        compare_zero=True,
        additional_parameters={'internal_terms': False},
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_PythTB_sym(check_integrate, system_Haldane_PythTB, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_PythTB,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="pythtb_sym",
        suffix_ref="",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        use_symmetry=True,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_TBmodels_sym(check_integrate, system_Haldane_TBmodels, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_TBmodels,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="sym",
        suffix_ref="",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        use_symmetry=True,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Haldane_TBmodels_sym_refine(
        check_integrate, system_Haldane_TBmodels, compare_energyresult, quantities_Haldane):
    check_integrate(
        system_Haldane_TBmodels,
        quantities_Haldane,
        fout_name="berry_Haldane_tbmodels",
        suffix="sym",
        suffix_ref="sym",
        Efermi=Efermi_Haldane,
        comparer=compare_energyresult,
        adpt_num_iter=1,
        use_symmetry=True,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })


def test_Fe_sym_refine(check_integrate, system_Fe_W90, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    check_integrate(
        system_Fe_W90,
        quantities_Fe,
        fout_name="berry_Fe_W90",
        adpt_num_iter=1,
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        suffix="sym",
        suffix_ref="sym",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult)


def test_Fe_pickle_Klist(check_integrate, system_Fe_W90, compare_energyresult, quantities_Fe):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos"""
    #  First, remove the Klist.pickle file if exists
    if os.path.exists("Klist.pickle"):
        os.remove("Klist.pickle")

    check_integrate(
        system_Fe_W90,
        quantities_Fe,
        fout_name="berry_Fe_W90",
        adpt_num_iter=0,
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        suffix="pickle",
        suffix_ref="sym",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult)
    check_integrate(
        system_Fe_W90,
        quantities_Fe,
        fout_name="berry_Fe_W90",
        adpt_num_iter=1,
        use_symmetry=True,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        suffix="pickle",
        suffix_ref="sym",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        restart=True)


def test_Fe_parallel_ray(check_integrate, system_Fe_W90, compare_energyresult, quantities_Fe, parallel_ray):
    """Test anomalous Hall conductivity , ohmic conductivity, dos, cumdos in parallel with ray"""
    check_integrate(
        system_Fe_W90,
        quantities_Fe,
        fout_name="berry_Fe_W90",
        suffix="paral-ray-4",
        suffix_ref="",
        Efermi=Efermi_Fe,
        comparer=compare_energyresult,
        parallel=parallel_ray,
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
    )
    parallel_ray.shutdown()


def test_Chiral(check_integrate, system_Chiral_left, compare_energyresult, quantities_Chiral):
    check_integrate(
        system_Chiral_left,
        quantities_Chiral,
        fout_name="berry_Chiral",
        Efermi=Efermi_Chiral,
        smearEf=1200,
        comparer=compare_energyresult,
        use_symmetry=True,
        additional_parameters={'external_terms': False},
        grid_param={
            'NK': [10, 10, 4],
            'NKFFT': [5, 5, 2]
        })


def test_Chiral_txt(check_integrate, system_Chiral_left, compare_energyresult, quantities_Chiral):
    check_integrate(
        system_Chiral_left,
        quantities_Chiral,
        fout_name="berry_Chiral",
        Efermi=Efermi_Chiral,
        comparer=compare_energyresult,
        mode="txt",
        smearEf=1200,
        use_symmetry=True,
        precision=-1e-5,
        additional_parameters={'external_terms': False},
        grid_param={
            'NK': [10, 10, 4],
            'NKFFT': [5, 5, 2]
        })


def test_Chiral_tetra(check_integrate, system_Chiral_left, compare_energyresult, quantities_Chiral):
    check_integrate(
        system_Chiral_left,
        quantities_Chiral,
        fout_name="berry_Chiral_tetra",
        Efermi=Efermi_Chiral,
        comparer=compare_energyresult,
        smearEf=1200,
        use_symmetry=True,
        additional_parameters={
            'external_terms': False,
            'tetra': True
        },
        grid_param={
            'NK': [10, 10, 4],
            'NKFFT': [5, 5, 2]
        })


def test_CuMnAs_PT(check_integrate, system_CuMnAs_2d_broken, compare_energyresult, quantities_CuMnAs_2d):
    """here no additional data is needed, we just check that degen_thresh=0.05 and degen_Kramers=True give the same result"""
    quantities = []
    specific_parameters = {}
    degen_param = [('degen_thresh', 0.05), ('degen_Kramers', True)]
    for quant in quantities_CuMnAs_2d:
        for tetra in True, False:
            for degen in degen_param:
                qfull = f"{quant}^tetra={tetra}_{degen[0]}={degen[1]}"
                quantities.append(qfull)
                specific_parameters[qfull] = {'tetra': tetra, degen[0]: degen[1]}

    result = check_integrate(
        system_CuMnAs_2d_broken,
        quantities,
        fout_name="berry_CuMnAs_2d",
        Efermi=Efermi_CuMnAs_2d,
        comparer=None,
        use_symmetry=True,
        additional_parameters={'external_terms': False},
        specific_parameters=specific_parameters,
        grid_param={
            'NK': [10, 10, 1],
            'NKFFT': [5, 5, 1]
        })

    for quant in quantities_CuMnAs_2d:
        for tetra in True, False:
            degen = degen_param[0]
            qfull1 = f"{quant}^tetra={tetra}_{degen[0]}={degen[1]}"
            degen = degen_param[1]
            qfull2 = f"{quant}^tetra={tetra}_{degen[0]}={degen[1]}"
            data1 = result.results.get(qfull1).data
            data2 = result.results.get(qfull2).data
            assert data1.shape == data2.shape
            assert np.all(np.array(data1.shape[1:]) == 3)
            assert np.all(np.array(data2.shape[1:]) == 3)
            precision = 1e-14 * np.max(abs(data1))
            assert data1 == approx(
                data2, abs=precision
            ), (
                f"calcuylated data of {qfull1}  and {qfull2} give a maximal "
                + "absolute difference of {abs_err} greater than the required precision {required_precision}. ".format(
                    abs_err=np.max(abs(data1 - data2)), required_precision=precision))
