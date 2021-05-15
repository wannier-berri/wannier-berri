"""pytest configuration file for WannierBerri tests."""

import os

import numpy
import pytest

@pytest.fixture(scope="session")
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session", autouse=True)
def create_tmpdir(rootdir):
    from pathlib import Path
    Path(os.path.join(rootdir, "_dat_files")).mkdir(exist_ok=True)