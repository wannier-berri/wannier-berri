"""Create system objects."""

import os
import tarfile
import shutil

import pytest

import wannierberri as wberri

@pytest.fixture(scope="session")
def system_Fe_W90(rootdir):
    """Create system for Fe using Wannier90 data"""

    data_dir = os.path.join(rootdir, "data", "Fe_Wannier90")

    # Extract files if is not already done
    for tag in ["mmn", "amn"]:
        if not os.path.isfile(os.path.join(data_dir, "Fe.{}".format(tag))):
            tar = tarfile.open(os.path.join(data_dir, "Fe.{}.tar.gz".format(tag)))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)

    # Create sIu, sHu files using mmn2uHu utility if they do not exist
    tags_needed = ["sHu", "sIu"]
    tags_compute = []

    # Compute tags only if the corresponding files do not exist
    for tag in tags_needed:
        if not os.path.isfile(os.path.join(data_dir, "Fe.{}".format(tag))):
            tags_compute.append(tag)

    if len(tags_compute) > 0:
        kwargs = {}
        for tag in tags_compute:
            kwargs["write" + tag.upper()] = True

        nb_out_list = wberri.mmn2uHu.run_mmn2uHu("Fe", INPUTDIR=data_dir,
            OUTDIR=str(data_dir)+"/reduced", **kwargs)
        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, "reduced_NB={0}".format(nb_out))
            os.rename(os.path.join(result_dir, "Fe_nbs={0}.{1}".format(nb_out, tag)),
                      os.path.join(data_dir, "Fe.{0}".format(tag)))

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.System_w90(seedname, berry=True, SHCqiao=True, SHCryoo=True,
        transl_inv=False)

    return system
