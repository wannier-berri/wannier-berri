import numpy as np
import untangle
from .system_w90 import System_w90
from .system_phonon_qe import System_Phonon_QE
from ..__utility import FFT,real_recip_lattice,FortranFileR
import os


class System_EPW(System_w90):

    """Class to represent electron-phonon matrix elements, wannierized by EPW   

    includes a System_phonons_qe and System_w90


    Parameters
    ----------
    asr : bool
        imposes a simple acoustic sum rule (equivalent to `zasr = 'simple'` in QuantumEspresson

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System`

    """

    def __init__(self,seedname,
            parameters_phonon={},
            parameters_electron={},
            **parameters):

        self.set_parameters(**parameters)
        path = tuple(seedname.split("/")[:-1])
        path = os.path.join(*path) if len(path)>0 else ""
        f  = open(os.path.join(path,"epwdata.fmt"),"r")
        ef = float(f.readline())
        nbndsub, nrr_k, nmodes, nrr_q, nrr_g = (int(x) for x in f.readline().split())
#        zstar, epsi = (float(x) for x in f.readline().split())
        f.close()
        with open(seedname+".epmatwp","rb") as f:
            epmatwp = np.fromfile(f,dtype=np.complex128).reshape( (nbndsub, nbndsub, nrr_k, nmodes,nrr_g), order='F' )

        self.system_ph=System_Phonon_QE(seedname,**parameters_phonon)
        self.system_el=System_w90(seedname,**parameters_electron)



    @property
    def is_electron_phonon(self):
        return True

