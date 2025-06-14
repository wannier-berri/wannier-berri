"""Test reading formatted Wannier90 files."""

import os
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
    uHu_unformatted = UHU.from_w90_file(os.path.join(data_dir, "GaAs"))
    uHu_formatted = UHU.from_w90_file(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    eql, msg = uHu_unformatted.equals(uHu_formatted)
    assert eql, msg

    uIu_unformatted = UIU.from_w90_file(os.path.join(data_dir, "GaAs"))
    uIu_formatted = UIU.from_w90_file(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    eql, msg = uIu_unformatted.equals(uIu_formatted)
    assert eql, msg


def test_formatted_spn(generate_formatted_files):
    data_dir = generate_formatted_files
    spn_unformatted = SPN.from_w90_file(os.path.join(data_dir, "GaAs"))
    spn_formatted = SPN.from_w90_file(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    eql, msg = spn_unformatted.equals(spn_formatted)
    assert eql, msg


def test_formatted_sXu(generate_formatted_files):
    data_dir = generate_formatted_files
    sHu_unformatted = SHU.from_w90_file(os.path.join(data_dir, "GaAs"))
    sHu_formatted = SHU.from_w90_file(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    eql, msg = sHu_unformatted.equals(sHu_formatted)
    assert eql, msg

    sIu_unformatted = SIU.from_w90_file(os.path.join(data_dir, "GaAs"))
    sIu_formatted = SIU.from_w90_file(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    eql, msg = sIu_unformatted.equals(sIu_formatted)
    assert eql, msg
