from .classes import Tabulator
from wannierberri import covariant_formulak as frml
from wannierberri import covariant_formulak_basic as frml_bas

class BerryCurvature(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Omega,**kwargs)

class Energy(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Eavln,**kwargs)

class Velocity(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Velocity,**kwargs)

class Spin(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.Spin,**kwargs)

class Der_Spin(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.DerSpin,**kwargs)

class morb(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml.morb_bohr,**kwargs)

class Der_morb(Tabulator):
    def __init__(self,**kwargs):
        super().__init__(frml_bas.Der_morb_bohr,**kwargs)


