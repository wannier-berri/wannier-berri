from .classes import StaticCalculator
from wannierberri import covariant_formulak as frml
from wannierberri import fermiocean

#######################################
#                                     #
#         integration (Efermi-only)   #
#                                     #
#######################################

#  TODO: Ideally, a docstring of every calculator should contain the equation that it implements
#        and references (with urls) to the relevant papers




class _DOS(StaticCalculator):

    def __init__(self, fder,**kwargs):
        self.Formula = frml.Identity
        self.factor = 1
        self.fder = fder
        super().__init__(**kwargs)

    def __call__(self, data_K):
        return super().__call__(data_K) * data_K.cell_volume

class DOS(_DOS):

    def __init__(self, **kwargs):
        super().__init__(fder=1, **kwargs)

class CumDOS(_DOS):

    def __init__(self, **kwargs):
        super().__init__(fder=0, **kwargs)


class AHC(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.Omega
        self.factor = fermiocean.fac_ahc
        self.fder = 0
        super().__init__(**kwargs)


class Ohmic(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.InvMass
        self.factor = fermiocean.factor_ohmic
        self.fder = 0
        super().__init__(**kwargs)


class BerryDipole_FermiSurf(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.VelOmega
        self.factor = 1
        self.fder = 1
        super().__init__(**kwargs)


class BerryDipole_FermiSea(StaticCalculator):

    def __init__(self, **kwargs):
        self.Formula = frml.DerOmega
        self.factor = 1
        self.fder = 0
        super().__init__(**kwargs)

    def __call__(self, data_K):
        res = super().__call__(data_K)
        # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        res.data = res.data.swapaxes(1, 2)
        return res
