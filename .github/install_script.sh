#!/bin/bash

# Be verbose, and stop with error as soon there's one
set -ev
# for gpaw
# sudo apt-get install libxc-dev

pip install codecov
pip install -U pip setuptools wheel 

# install optional dependencies
pip install pythtb # tbmodels spglib
# pip install ase
# pip install gpaw
# gpaw info
# gpaw install-data --register ~/gpaw-data
pip install -U .[all]
# sudo apt-get install wannier90

