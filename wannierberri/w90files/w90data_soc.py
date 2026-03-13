from .wandata_soc import WannierDataSOC


class Wannier90dataSOC(WannierDataSOC):
    """Class to handle Wannier90 data with spin-orbit coupling as perturbation - deprecated, use WannierData instead."""

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("Wannier90dataSOC is deprecated, use WannierDataSOC instead.", DeprecationWarning)
        super().__init__(*args, **kwargs)
