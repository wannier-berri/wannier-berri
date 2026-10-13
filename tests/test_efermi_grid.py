"""Fermi energies of the static calculators: increasing, and evenly spaced without the tetrahedron method"""
import numpy as np
import pytest
from wannierberri.calculators import static
from .test_formula import datak_Fe_Gamma as datak_Fe  # noqa: F401  (fixture)
from .common_systems import Efermi_Fe


calculators = [static.CumDOS, static.AHC, static.Ohmic_FermiSurf, static.BerryDipole_FermiSurf]


@pytest.mark.parametrize("calc_class", calculators)
@pytest.mark.parametrize("Efermi", [[17.0, 17.1, 17.3],      # uneven
                                    [17.1, 17.1, 17.2]])     # repeated
def test_not_evenly_spaced(calc_class, Efermi):
    with pytest.raises(ValueError, match="evenly spaced and increasing"):
        calc_class(Efermi=np.array(Efermi), tetra=False)
    calc_class(Efermi=np.array(Efermi), tetra=True)


@pytest.mark.parametrize("calc_class", calculators)
@pytest.mark.parametrize("tetra", [True, False])
@pytest.mark.parametrize("Efermi", [[17.3, 17.2, 17.1], [17.1, 17.3, 17.2]])
def test_not_increasing(calc_class, tetra, Efermi):
    with pytest.raises(ValueError, match="increasing order"):
        calc_class(Efermi=np.array(Efermi), tetra=tetra)


@pytest.mark.parametrize("calc_class", calculators)
def test_evenly_spaced(calc_class):
    calc_class(Efermi=Efermi_Fe, tetra=False)
    calc_class(Efermi=list(np.arange(17.0, 18.0, 0.1)), tetra=False)  # a list, with roundoff in the steps
    calc_class(Efermi=np.array([17.5]), tetra=False)


@pytest.mark.parametrize("calc_class", calculators)
def test_tetra_uneven(datak_Fe, calc_class):  # noqa: F811
    """with the tetrahedron method, uneven steps and repeated values give the values computed one by one"""
    Efermi = np.array([17.2, 17.25, 17.25, 17.6, 17.9])
    result = calc_class(Efermi=Efermi, tetra=True)(datak_Fe).data
    single = np.array([calc_class(Efermi=np.array([ef]), tetra=True)(datak_Fe).data[0] for ef in Efermi])
    assert np.abs(single).max() > 0
    assert np.allclose(result, single, rtol=0, atol=1e-8 * np.abs(single).max())
