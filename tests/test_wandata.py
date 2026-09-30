from types import SimpleNamespace

import numpy as np

from wannierberri.w90files.wandata import BKVectors, CheckPoint, WIN, WannierData


def _checkpoint():
    return SimpleNamespace(
        real_lattice=np.eye(3),
        recip_lattice=np.eye(3),
        mp_grid=np.ones(3, dtype=int),
        kpt_red=np.zeros((1, 3)),
        num_bands=1,
        num_kpts=1,
    )


def _patch_w90_readers(monkeypatch, calls):
    checkpoint = _checkpoint()
    from_chk = object()
    from_nnkp = object()

    monkeypatch.setattr(
        CheckPoint,
        "from_w90_file",
        classmethod(lambda cls, seedname, **kwargs: checkpoint),
    )
    monkeypatch.setattr(
        CheckPoint,
        "from_win",
        classmethod(lambda cls, win: checkpoint),
    )
    monkeypatch.setattr(
        WIN,
        "from_w90_file",
        classmethod(lambda cls, seedname: SimpleNamespace()),
    )
    monkeypatch.setattr(
        BKVectors,
        "from_kpoints",
        classmethod(lambda cls, **kwargs: calls.append("chk") or from_chk),
    )

    def read_nnkp(cls, filename, **kwargs):
        calls.append(("nnkp", kwargs.get("real_lattice")))
        return from_nnkp

    monkeypatch.setattr(BKVectors, "from_nnkp", classmethod(read_nnkp))
    return from_chk, from_nnkp


def test_from_w90_files_explicit_readnnkp_overrides_auto(monkeypatch, tmp_path):
    calls = []
    _, from_nnkp = _patch_w90_readers(monkeypatch, calls)
    seedname = str(tmp_path / "wannier90")
    (tmp_path / "wannier90.nnkp").touch()

    wandata = WannierData.from_w90_files(
        seedname=seedname,
        files=["chk"],
        readnnkp=True,
    )

    assert wandata.bkvec is from_nnkp
    assert len(calls) == 1
    assert calls[0][0] == "nnkp"
    assert np.array_equal(calls[0][1], np.eye(3))


def test_from_w90_files_explicit_skip_nnkp(monkeypatch, tmp_path):
    calls = []
    from_chk, _ = _patch_w90_readers(monkeypatch, calls)
    seedname = str(tmp_path / "wannier90")
    (tmp_path / "wannier90.nnkp").touch()

    wandata = WannierData.from_w90_files(
        seedname=seedname,
        files=["win"],
        readnnkp=False,
    )

    assert wandata.bkvec is from_chk
    assert calls == ["chk"]
