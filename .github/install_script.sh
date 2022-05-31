#!/bin/bash

# Author: Dominik Gresch <greschd@gmx.ch>
# Copied from Z2Pack https://github.com/Z2PackDev/Z2Pack

# Be verbose, and stop with error as soon there's one
set -ev
sudo apt-get install libxc-dev
pip install codecov
pip install -U pip setuptools wheel 
pip install ray

# install optional dependencies
pip install tbmodels pythtb spglib
pip install ase
pip install gpaw
gpaw info
gpaw install-data --register ~/gpaw-data

case "$INSTALL_TYPE" in
    dev)
        pip install -e .
        ;;
    dev_sdist)
    #     python setup.py sdist
    #     ls -1 dist/ | xargs -I % pip install dist/%[dev]
    #     ;;
    # dev_bdist_wheel)
    #     python setup.py bdist_wheel
    #     ls -1 dist/ | xargs -I % pip install dist/%[dev]
    #     ;;
esac
