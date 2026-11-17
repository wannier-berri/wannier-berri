"""
wannierberri - a module for Wannier Functions and Wannier interpolation
"""
from . import utils
from .welcome_message import welcome
from .w90files import WannierData, WannierDataSOC
from .wannierisation.wannierise import wannierise
from .system import System_R, SystemKP, SystemSOC
from .grid import Grid, Path
from .evaluate_k import evaluate_k, evaluate_k_path
from .run_grid import run
from .logging import configure_logging
configure_logging()


try:
    from irrep import logging as irrep_logging
    irrep_logging.configure_logging(console_level=irrep_logging.WARNING)
except:
    pass

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"
