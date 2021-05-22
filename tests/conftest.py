"""pytest configuration file for WannierBerri tests."""

import os

import numpy
import pytest

@pytest.fixture(scope="session")
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session", autouse=True)
def output_dir(rootdir):
    from pathlib import Path
    directory = os.path.join(rootdir, "_dat_files")
    Path(directory).mkdir(exist_ok=True)
    return directory
