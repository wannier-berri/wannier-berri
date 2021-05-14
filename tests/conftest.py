"""pytest configuration file for WannierBerri tests."""

import os

import numpy
import pytest

@pytest.fixture(scope="session")
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def Efermi_Fe():
    return os.path.dirname(os.path.abspath(__file__))

