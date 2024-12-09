"""Common constants shared by all tests"""
import warnings
import sys
import os

try:
    import pyfftw
    print(f"pyfftw version : {pyfftw.__version__}")  # this is only to avoid lint error
except ImportError as err:
    warnings.warn(f"PyFFT was not imported:{err}")

import wannierberri as wberri

print(f"sys.version: {sys.version}")
print(f"sys.path: {sys.path}")


print("sys.path: ", sys.path)
print(f"imported wberri from {wberri.__file__} version {wberri.__version__}")

# Root folder containing test scripts
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing output dat files of tests
OUTPUT_DIR = os.path.join(ROOT_DIR, "_output")
OUTPUT_DIR_RUN = os.path.join(OUTPUT_DIR, 'integrate_files')
REF_DIR = os.path.join(ROOT_DIR, 'reference')
REF_DIR_INTEGRATE = os.path.join(REF_DIR, 'integrate_files')
TMP_DATA_DIR = os.path.join(ROOT_DIR, "_tmp_data_postw90")
