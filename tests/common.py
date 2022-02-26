"""Common constants shared by all tests"""

import os
import pytest

# Root folder containing test scripts
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing output dat files of tests
OUTPUT_DIR = os.path.join(ROOT_DIR, "_dat_files")
REF_DIR    = os.path.join(ROOT_DIR, 'reference')
