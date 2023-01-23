import numpy as np
import untangle
from .system_w90 import System_w90
from .system_phonon import System_Phonon
from ..__utility import real_recip_lattice
from ..__factors import bohr # Bohr radius in Angstroms
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
            asr = True
            **parameters):

        self.set_parameters(**parameters)
        path = tuple(seedname.split("/")[:-1])
        path = os.path.join(*path) if len(path)>0 else ""

        (real_lattice,masses,atom_positions_red,wannier_centers_red
                ) = read_crystal(os.path.join(path,"crystal.fmt"))

        (nbndsub, nmodes, nrr_k, nrr_q, nrr_g, Ham_R, dyn_mat, irvec_k, irvec_q, irvec_g,
            ndegen_k,ndegen_q,ndegen_g) = read_epwdata(os.path.join(path,"epwdata.fmt"))

        self.system_ph=System_Phonon(real_lattice=real_lattice,
                masses=masses,
                iRvec=irvec_q,ndegen=None,
                mp_grid=None,
                Ham_R=None,dyn_mat_R=dyn_mat,
                atom_positions_cart=None,atom_positions_red=atom_positions_red, # speciy one of them
                asr=asr,
                )

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
        return np.array([dtype(x) for x in self.fl.readline().split()])

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
                irvec_k, irvec_q, irvec_g, ndegen_k, ndegen_q, ndegen_g)


def read_crystal(filename):
    f  = FormattedFile(filename)
    nat = f.readint()[0] #  WRITE(crystal,*) nat
    nmodes = f.readint()[0] #  WRITE(crystal,*) nmodes
    assert 3*nat==nmodes
    nelec = f.readfloat()[0] #  WRITE(crystal,*) nelec
    at = f.readfloat().reshape((3,3),order='C')  #  WRITE(crystal,*) at
    bg = f.readfloat().reshape((3,3),order='C') #  WRITE(crystal,*) bg
    assert (np.linalg.norm(at.dot(bg.T)-np.eye(3))<1e-8)
    omega = f.readfloat()[0] #  WRITE(crystal,*) omega
    alat = f.readfloat()[0] #  WRITE(crystal,*) alat
    assert abs(omega-np.linalg.det(at*alat))<1e-8
    real_lattice=alat*at*bohr
    atom_positions_red = f.readfloat().reshape(nat,3,order='C').dot(bg.T)#  WRITE(crystal,*) tau
    print (atom_positions_red)
    amass = f.readfloat() #  WRITE(crystal,*) amass
    ityp = f.readint() #  WRITE(crystal,*) ityp
    masses = np.array([amass[it-1] for it in ityp])
    f.readline() #  WRITE(crystal,*) noncolin
    wannier_centers_red = f.readfloat().reshape((-1,3),order='C').dot(bg.T)#  WRITE(crystal,*) w_centers
    print (real_lattice)
    print (wannier_centers_red)
    return (real_lattice,masses,atom_positions_red,wannier_centers_red)
