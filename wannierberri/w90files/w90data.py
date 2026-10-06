from .wandata import WannierData
import logging
logger = logging.getLogger(__name__)



class Wannier90data(WannierData):
    """Class to handle Wannier90 data. - deprecated, use WannierData instead."""

    def __init__(self, *args, **kwargs):
        logger.warning("Wannier90data is deprecated, use WannierData instead.")
        super().__init__(*args, **kwargs)
