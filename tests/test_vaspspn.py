import pytest
from wannierberri.utils.vaspspn import main as vaspspn_main
from .conftest import OUTPUT_DIR, REF_DIR, ROOT_DIR
from wannierberri.w90files import SPN


@pytest.mark.parametrize("norm", ["norm", "none"])
def test_vaspspn(norm):
    argv = [
        f"fin={ROOT_DIR}/data/WAVECAR",
        f"fout={OUTPUT_DIR}/wannier90-{norm}.spn",
        "IBstart=2",
        "NB=3",
        "norm={norm}",
    ]
    vaspspn_main(argv)
    spn = SPN.from_w90_file(f"{OUTPUT_DIR}/wannier90-{norm}")
    spn_ref = SPN.from_w90_file(f"{REF_DIR}/wannier90-{norm}")
    assert spn.NK == spn_ref.NK, f"NK mismatch: {spn.NK} vs {spn_ref.NK}"
    assert spn.NB == spn_ref.NB, f"NB mismatch: {spn.NB} vs {spn_ref.NB}"
    assert set(spn.data.keys()) == set(spn_ref.data.keys()), \
        f"Data keys mismatch: {set(spn.data.keys())} vs {set(spn_ref.data.keys())}"
    for ik, valref in spn_ref.data.items():
        val = spn.data[ik]
        assert val.shape == valref.shape, f"Shape mismatch for ik={ik}: {val.shape} vs {valref.shape}"
        assert val == pytest.approx(valref, abs=1e-6), f"Value mismatch for ik={ik}"
