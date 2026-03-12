#!/bin/bash

# Be verbose, and stop with error as soon there's one
set -ev
# for gpaw
# sudo apt-get install libxc-dev

pip install codecov
pip install -U pip setuptools wheel 

# install optional dependencies
pip install pythtb tbmodels
pip install gpaw
pip install -U .[all]

