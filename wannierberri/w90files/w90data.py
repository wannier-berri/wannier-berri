from .wandata import WannierData


class WannierData(WannierData):
    """Class to handle Wannier90 data. - deprecated, use WannierData instead."""

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("Wannier90data is deprecated, use WannierData instead.", DeprecationWarning)
        super().__init__(*args, **kwargs)
