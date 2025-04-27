"""Test reading formatted Wannier90 files."""

import os
import numpy as np
import pytest

import wannierberri as wberri
from wannierberri.w90files import UHU, UIU, SHU, SIU, SPN


@pytest.fixture(scope="module")
def generate_formatted_files(create_files_GaAs_W90):
    """Create formatted files for Fe using utils.mmn2uHu"""

    data_dir = create_files_GaAs_W90

    print(f"data_dir={str(data_dir)}")

    # Create sIu, sHu files using mmn2uHu utility if they do not exist
    tags_needed = ["uHu", "uIu", "sHu", "sIu", "spn"]
    tags_compute = []

    # Compute tags only if the corresponding files do not exist
    for tag in tags_needed:
        if not os.path.isfile(os.path.join(data_dir, f"GaAs_formatted.{tag}")):
            tags_compute.append(tag)

    if len(tags_compute) > 0:
        kwargs = []
        kwargs.append(f"input={str(data_dir)}")
        kwargs.append(f"output={str(data_dir)}/reduced_formatted")
        kwargs.append("targets=" + ",".join(tags_compute))
        tags_formatted = []
        for tag in tags_compute:
            if tag == "spn":
                tags_formatted.append('spn_out')
            else:
                tags_formatted.append(tag)
        kwargs.append("formatted=" + ",".join(tags_formatted))
        kwargs.append("IBstart=1")
        kwargs.append("IBstartSum=1")
        kwargs.append("NBsum=16,100")
        print("kwargs = ", kwargs)
        nb_out_list = wberri.utils.mmn2uHu.main(["GaAs"] + kwargs)

        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, f"reduced_formatted_NB={nb_out}")
            if tag == "spn":
                os.rename(
                    os.path.join(result_dir, f"GaAs.{tag}"),
                    os.path.join(data_dir, f"GaAs_formatted.{tag}"))
            else:
                os.rename(
                    os.path.join(result_dir, f"GaAs_nbs={nb_out}.{tag}"),
                    os.path.join(data_dir, f"GaAs_formatted.{tag}"))

    return data_dir


def test_formatted_uXu(generate_formatted_files):
    data_dir = generate_formatted_files
    uHu_unformatted = UHU(os.path.join(data_dir, "GaAs"), autoread=True)
    uHu_formatted = UHU(os.path.join(data_dir, "GaAs_formatted"), formatted=True, autoread=True)
    assert np.allclose(uHu_unformatted.data, uHu_formatted.data)

    uIu_unformatted = UIU(os.path.join(data_dir, "GaAs"), autoread=True)
    uIu_formatted = UIU(os.path.join(data_dir, "GaAs_formatted"), formatted=True, autoread=True)
    assert np.allclose(uIu_unformatted.data, uIu_formatted.data)


def test_formatted_spn(generate_formatted_files):
    data_dir = generate_formatted_files
    spn_unformatted = SPN(os.path.join(data_dir, "GaAs"), autoread=True)
    spn_formatted = SPN(os.path.join(data_dir, "GaAs_formatted"), formatted=True, autoread=True)
    assert np.allclose(spn_unformatted.data, spn_formatted.data)


def test_formatted_sXu(generate_formatted_files):
    data_dir = generate_formatted_files
    sHu_unformatted = SHU(os.path.join(data_dir, "GaAs"), autoread=True)
    sHu_formatted = SHU(os.path.join(data_dir, "GaAs_formatted"), formatted=True, autoread=True)
    assert np.allclose(sHu_unformatted.data, sHu_formatted.data)

    sIu_unformatted = SIU(os.path.join(data_dir, "GaAs"), autoread=True)
    sIu_formatted = SIU(os.path.join(data_dir, "GaAs_formatted"), formatted=True, autoread=True)
    assert np.allclose(sIu_unformatted.data, sIu_formatted.data)
