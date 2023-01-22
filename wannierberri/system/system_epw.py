import numpy as np
import untangle
from .system_w90 import System_w90
from .system_phonon_qe import System_Phonon_QE
from ..__utility import FFT,real_recip_lattice,FortranFileR
import os

# Note : it will probably not work if use_ws=True in EPW

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
            system_electron=None,
            system_phonon=None,
            **parameters):

        self.set_parameters(**parameters)
        path = tuple(seedname.split("/")[:-1])
        path = os.path.join(*path) if len(path)>0 else ""
        read_crystal(os.path.join(path,"crystal.fmt"))
        (nbndsub, nmodes, nrr_k, nrr_q, nrr_g, Ham_R, dyn_mat, irvec_k, irvec_q, irvec_g,
            ndegen_k,ndegen_g) = read_epwdata(os.path.join(path,"epwdata.fmt"))
        with open(seedname+".epmatwp","rb") as f:
            epmatwp = np.fromfile(f,dtype=np.complex128).reshape( (nbndsub, nbndsub, nrr_k, nmodes,nrr_g), order='F' )
        epmatwp/=ndegen_k[None,None,:,None,None]
        epmatwp/=ndegen_g[None,None,None,None,:]
        print (nbndsub,nmodes,nrr_k,nrr_q,nrr_g)

#        self.system_ph=System_Phonon_QE(seedname,**parameters_phonon)
#        self.system_el=System_w90(seedname,**parameters_electron)

# os.path.join(path,"epwdata.fmt"),"r")

    @property
    def is_electron_phonon(self):
        return True



class FormattedFile():

    def __init__(self,filename):
        self.fl = open(filename,"r")

    def read_dtype(self,dtype):
        return [dtype(x) for x in self.fl.readline().split()]

    def readint(self):
        return self.read_dtype(int)

    def readcomplex(self):
        return np.complex(*np.safe_eval(self.fl.readline().strip()))

    def readfloat(self):
        return self.read_dtype(float)

    def readline(self):
        return self.fl.readline()

def read_epwdata(filename):
    
    f  = FormattedFile(filename)
    # so far just read lines, data will be extracted when needed
    f.readfloat() # WRITE(epwdata,*) ef
    nbndsub, nrr_k, nmodes, nrr_q, nrr_g = f.readint() # WRITE(epwdata,*) nbndsub, nrr_k, nmodes, nrr_q, nrr_g
    f.readline() # WRITE(epwdata,*) zstar, epsi
    # TODO : optimize reading (try sdsolutions from here : https://stackoverflow.com/questions/34882397/read-fortran-complex-number-into-python
    Ham_R = np.zeros( (nbndsub,nbndsub,nrr_k), dtype=complex)
    for ibnd in range(nbndsub):
        for jbnd in range(nbndsub):
            for irk in range(nrr_k):
                Ham_R[ibnd,jbnd,irk] = f.readcomplex() # WRITE (epwdata,*) chw(ibnd, jbnd, irk)
    dyn_mat = np.zeros( (nmodes,nmodes,nrr_q), dtype=complex)
    for imode in range(nmodes):
        for jmode in range(nmodes):
            for irq in range(nrr_q):
                dyn_mat[imode,jmode,irq] = f.readcomplex()  # WRITE(epwdata,*) rdw(imode, jmode, irq)
    irvec_k = np.zeros( (nrr_k,3),dtype = int)
    ndegen_k = np.zeros( nrr_k,dtype = int) 
    for irk in range(nrr_k):
        _ = f.readint()        #    WRITE(epwdata,*) irvec_k(:,irk), ndegen_k(irk)
        irvec_k[irk],ndegen_k[irk] = _[:3],_[3]
    irvec_q = np.zeros( (nrr_q,3),dtype = int)
    ndegen_q = np.zeros( nrr_q,dtype = int) 
    for irq in range(nrr_q):
        _ = f.readint()        #    WRITE(epwdata,*) irvec_q(:,irq), ndegen_q(irq)
        irvec_q[irq],ndegen_q[irq] = _[:3],_[3]
    irvec_g = np.zeros( (nrr_g,3),dtype = int)
    ndegen_g = np.zeros( nrr_g,dtype = int) 
    for irg in range(nrr_g):
        _ = f.readint()        #    WRITE(epwdata,*) irvec_g(:,irg), ndegen_g(irg)
        irvec_g[irg],ndegen_g[irg] = _[:3],_[3]
    assert np.all(ndegen_k>0)
    assert np.all(ndegen_q>0)
    assert np.all(ndegen_g>0)
    Ham_R = Ham_R/ndegen_k[None,None,:]
    dyn_mat = dyn_mat/ndegen_q[None,None,:]
    return (nbndsub, nmodes, nrr_k, nrr_q, nrr_g, 
                Ham_R, dyn_mat,
                irvec_k, irvec_q, irvec_g, ndegen_k, ndegen_g)


def read_crystal(filename):
    f  = open(filename,"r")
    f.readline() #  WRITE(crystal,*) nat
    f.readline() #  WRITE(crystal,*) nmodes
    f.readline() #  WRITE(crystal,*) nelec
    f.readline() #  WRITE(crystal,*) at
    f.readline() #  WRITE(crystal,*) bg
    f.readline() #  WRITE(crystal,*) omega
    f.readline() #  WRITE(crystal,*) alat
    f.readline() #  WRITE(crystal,*) tau
    f.readline() #  WRITE(crystal,*) amass
    f.readline() #  WRITE(crystal,*) ityp
    f.readline() #  WRITE(crystal,*) noncolin
    f.readline() #  WRITE(crystal,*) w_centers
    return None
