"""Test reading formatted Wannier90 files."""

import os
import numpy as np
import pytest
from pytest import approx

import wannierberri as wberri
from wannierberri.__w90_files import UHU, UIU, SHU, SIU, SPN

from create_system import create_files_GaAs_W90

@pytest.fixture(scope="module")
def generate_formatted_files(rootdir, create_files_GaAs_W90):
    """Create formatted files for Fe using mmn2uHu"""

    data_dir = create_files_GaAs_W90

    # Create sIu, sHu files using mmn2uHu utility if they do not exist
    tags_needed = ["uHu", "uIu", "sHu", "sIu"]
    tags_compute = []

    # Compute tags only if the corresponding files do not exist
    for tag in tags_needed:
        if not os.path.isfile(os.path.join(data_dir, "GaAs_formatted.{}".format(tag))):
            tags_compute.append(tag)

    if len(tags_compute) > 0:
        kwargs = {}
        for tag in tags_compute:
            kwargs["write" + tag.upper()] = True
            kwargs[tag + "_formatted"] = True

        nb_out_list = wberri.mmn2uHu.run_mmn2uHu("GaAs", INPUTDIR=data_dir,
            OUTDIR=str(data_dir)+"/reduced_formatted", **kwargs)

        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, "reduced_formatted_NB={0}".format(nb_out))
            os.rename(os.path.join(result_dir, "GaAs_nbs={0}.{1}".format(nb_out, tag)),
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
