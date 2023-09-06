"""Common constants shared by all tests"""
import pyfftw
print (f"pyfftw version : {pyfftw.__version__}") # this is only to avoid lint error
import os

# Root folder containing test scripts
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing output dat files of tests
OUTPUT_DIR = os.path.join(ROOT_DIR, "_dat_files")
REF_DIR = os.path.join(ROOT_DIR, 'reference')
TMP_DATA_DIR = os.path.join(ROOT_DIR, "_tmp_data_postw90")
