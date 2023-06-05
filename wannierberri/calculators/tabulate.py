from .classes import Tabulator
from .. import covariant_formulak as frml

class BerryCurvature(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Omega,**kwargs)

class Energy(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Eavln,**kwargs)

