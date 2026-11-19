
from wannierberri.system import System_R
from wannierberri.models import Haldane_ptb
from pytest import approx


def test_find_efermi():
    system = System_R.from_pythtb(Haldane_ptb())
    system._NKFFT_recommended = (12, 12, 1)
    assert system.find_efermi(0.5, prec=1e-3) == approx(-1.5212840824415266, abs=2e-3)
    assert system.find_efermi(1.000, prec=1e-3) == approx(0, abs=2e-3)
    assert system.find_efermi(1.005, prec=1e-3) == approx(0.611043159531992, abs=2e-3)
