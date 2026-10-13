from .wandata_soc import WannierDataSOC
import logging
logger = logging.getLogger(__name__)



class Wannier90dataSOC(WannierDataSOC):
    """Class to handle Wannier90 data with spin-orbit coupling as perturbation - deprecated, use WannierData instead."""

    def __init__(self, *args, **kwargs):
        logger.warning("DeprecationWarning: Wannier90dataSOC is deprecated, use WannierDataSOC instead.")
        super().__init__(*args, **kwargs)
