"""Test reading formatted Wannier90 files."""

import os
import numpy as np
import pytest

import wannierberri as wberri
from wannierberri.system.__w90_files import UHU, UIU, SHU, SIU, SPN


@pytest.fixture(scope="module")
def generate_formatted_files(create_files_GaAs_W90):
    """Create formatted files for Fe using utils.mmn2uHu"""

    data_dir = create_files_GaAs_W90

    # Create sIu, sHu files using mmn2uHu utility if they do not exist
    tags_needed = ["uHu", "uIu", "sHu", "sIu", "spn"]
    tags_compute = []

    # Compute tags only if the corresponding files do not exist
    for tag in tags_needed:
        if not os.path.isfile(os.path.join(data_dir, "GaAs_formatted.{}".format(tag))):
            tags_compute.append(tag)

    if len(tags_compute) > 0:
        kwargs = {}
        for tag in tags_compute:
            kwargs["write" + tag.upper()] = True
            if tag == "spn":
                kwargs[tag + "_formatted_in"] = False
                kwargs[tag + "_formatted_out"] = True
            else:
                kwargs[tag + "_formatted"] = True

        nb_out_list = wberri.utils.mmn2uHu.run_mmn2uHu(
            "GaAs", INPUTDIR=data_dir, OUTDIR=str(data_dir) + "/reduced_formatted", **kwargs)

        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, "reduced_formatted_NB={0}".format(nb_out))
            if tag == "spn":
                os.rename(
                    os.path.join(result_dir, "GaAs.{0}".format(tag)),
                    os.path.join(data_dir, "GaAs_formatted.{0}".format(tag)))
            else:
                os.rename(
                    os.path.join(result_dir, "GaAs_nbs={0}.{1}".format(nb_out, tag)),
                    os.path.join(data_dir, "GaAs_formatted.{0}".format(tag)))

    return data_dir


def test_formatted_uXu(generate_formatted_files):
    data_dir = generate_formatted_files
    uHu_unformatted = UHU(os.path.join(data_dir, "GaAs"))
    uHu_formatted = UHU(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    assert np.allclose(uHu_unformatted.data, uHu_formatted.data)

    uIu_unformatted = UIU(os.path.join(data_dir, "GaAs"))
    uIu_formatted = UIU(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    assert np.allclose(uIu_unformatted.data, uIu_formatted.data)


def test_formatted_spn(generate_formatted_files):
    data_dir = generate_formatted_files
    spn_unformatted = SPN(os.path.join(data_dir, "GaAs"))
    spn_formatted = SPN(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    assert np.allclose(spn_unformatted.data, spn_formatted.data)


def test_formatted_sXu(generate_formatted_files):
    data_dir = generate_formatted_files
    sHu_unformatted = SHU(os.path.join(data_dir, "GaAs"))
    sHu_formatted = SHU(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    assert np.allclose(sHu_unformatted.data, sHu_formatted.data)

    sIu_unformatted = SIU(os.path.join(data_dir, "GaAs"))
    sIu_formatted = SIU(os.path.join(data_dir, "GaAs_formatted"), formatted=True)
    assert np.allclose(sIu_unformatted.data, sIu_formatted.data)
