#!/bin/bash

# Be verbose, and stop with error as soon there's one
set -ev

pip install -U pip setuptools wheel 

pip install -U .[default,tests,symmetry,parallel,fftw]

