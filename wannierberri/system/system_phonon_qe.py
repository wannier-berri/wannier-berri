import numpy as np
import untangle
import multiprocessing
from wannierberri.__utility import real_recip_lattice
from wannierberri.system import System_w90
from wannierberri.__utility import FFT
from scipy import constants as const
from ..__factors import Ry_eV
from .system_phonon import System_Phonon
#from scipy.constants import elementary_charge, hbar, electron_mass, physical_constants, angstrom  #, Boltzmann

#print ( 1/(hbar*2*np.pi*1e12/elementary_charge) )

# These constants are taken from QE
AMU_RY = const.m_u/const.m_e/2

def _str2array(s,dtype=float):
    if dtype == float:
        return np.array([_.split() for _ in s.strip().split("\n")],dtype=float)
    elif dtype == complex:
        s = s.replace(","," ")
        s = _str2array(s)
        return s[:,0]+1j*s[:,1]
    else:
        raise ValueError(f"dtype = '{dtype}' is not supported by _str2array")


class System_Phonon_QE(System_Phonon,System_w90):

    """Class to represent dynamical matrices from QuantumEspresso

    reads the '*.dyn*.xml' files and
    allows to interpolate the dynamical matrix and get the phonon frequencies
    so far onl;y DOS, cumDOS and tabulation is tested.
    Other calculators, in principle, may be applied on your own risk.

    Parameters
    ----------
    asr : bool
        imposes a simple acoustic sum rule (equivalent to `zasr = 'simple'` in QuantumEspresson

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System`

    """

    def __init__(self,seedname,
            fft='fftw',
            npar=multiprocessing.cpu_count(),
            asr=True,
            **parameters):

        self.set_parameters(**parameters)
        with open(seedname+".dyn0","r") as f:
            mp_grid =  np.array(f.readline().split(),dtype=int)
            nqirr = int(f.readline().strip())
#        print (self.mp_grid,nqirr)
        q_points = []
        dynamical_mat = []
        cnt = 0
        for ifile in range(1,nqirr+1):
            fname = f"{seedname}.dyn{ifile}.xml"
            data = untangle.parse(open(fname).read().lower()).root
            geometry = data.geometry_info
            if ifile == 1:
                number_of_types = int(geometry.number_of_types.cdata)
                masses_tp = np.array([float(geometry.__getattr__(f'mass_{i+1}').cdata) for i in range(number_of_types)])
                real_lattice = _str2array(geometry.at.cdata)
                self.real_lattice, self.recip_lattice = real_recip_lattice(real_lattice=real_lattice)
                number_of_atoms = int(geometry.number_of_atoms.cdata)
                num_wann = 3*number_of_atoms
                atom_positions_cart = np.array([geometry.__getattr__(f'atom_{i+1}')['tau'].split()
                                for i in range(number_of_atoms)],dtype=float)
                types = np.array([geometry.__getattr__(f'atom_{i+1}')['index'] for i in range(number_of_atoms)],dtype=int)-1
                masses = masses_tp[types]
            number_of_q = int(geometry.number_of_q.cdata)
            for iq in range(number_of_q):
                dynamical = data.__getattr__(f"dynamical_mat__{iq+1}")
                q = _str2array(dynamical.q_point.cdata).reshape(3)
                q = self.real_lattice.dot(q) # converting from cartisean(2pi/alatt) to reduced coordinates
                dyn_mat = np.zeros( (3*number_of_atoms,3*number_of_atoms), dtype=complex)
                for i in range(number_of_atoms):
                    for j in range(number_of_atoms):
                        phi = _str2array(dynamical.__getattr__(f'phi_{i+1}_{j+1}').cdata,dtype=complex
                                            ).reshape(3,3,order='F')
                        dyn_mat[i*3:i*3+3,j*3:j*3+3] = phi
                dyn_mat = 0.5*(dyn_mat+dyn_mat.T.conj())
                dynamical_mat.append( dyn_mat )
                q_points.append(q)
                cnt += 1
        qpoints_found = np.zeros( mp_grid,dtype=float)
        dynamical_matrix_q = np.zeros( tuple(mp_grid)+(num_wann,)*2,dtype=complex)
        agrid = np.array(mp_grid)
        for q,dyn_mat in zip(q_points,dynamical_mat):
            iq = tuple(np.array(np.round(q*agrid), dtype=int) % agrid)
            dynamical_matrix_q[iq] = dyn_mat
            qpoints_found[iq] = True
        assert np.all(qpoints_found),('some qpoints were not found in the files:\n'+'\n'.join(str(x/agrid))
               for x in np.where(np.logical_not(qpoints_found)))
        dyn_mat_R = FFT(dynamical_matrix_q, axes=(0, 1, 2), numthreads=npar, fft=fft, destroy=False)
        iRvec, Ndegen = self.wigner_seitz(mp_grid)
        dyn_mat_R= np.array([dyn_mat_R[tuple(iR % mp_grid)] / nd for iR, nd in zip(iRvec, Ndegen)]) / np.prod(mp_grid)
        dyn_mat_R = dyn_mat_R.transpose((1, 2, 0))


        super().__init__(
                real_lattice=real_lattice,
                masses=masses,
                iRvec=iRvec,ndegen=None,
                mp_grid=mp_grid,
                Ham_R=None,dyn_mat_R=dyn_mat_R,
                atom_positions_cart=atom_positions_cart,atom_positions_red=None, # speciy one of them
                asr=True,
                )


