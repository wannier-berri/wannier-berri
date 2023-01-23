import numpy as np
import untangle
import multiprocessing
from wannierberri.__utility import real_recip_lattice
from .system import System
from wannierberri.__utility import FFT
from scipy import constants as const
from ..__factors import Ry_eV

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


class System_Phonon(System):

    """Class to represent dynamical matrices from anything"""

    def __init__(self,
                real_lattice,
                masses,
                iRvec,ndegen=None,
                mp_grid=None,
                Ham_R=None,dyn_mat_R=None,
                atom_positions_cart=None,atom_positions_red=None, # speciy one of them
                asr=True,
                **parameters
                ):

        self.set_parameters(**parameters)
        self.mp_grid = mp_grid
        self.real_lattice, self.recip_lattice = real_recip_lattice(real_lattice=real_lattice)
        assert (atom_positions_red is None)!=(atom_positions_cart is None), "need to specify either atom_positions_red or atom_positions_cart, but never both) "
        if atom_positions_red is not None:
            self.atom_positions_red = atom_positions_red
        else:
            self.atom_positions_red = atom_positions_cart.dot(np.linalg.inv(self.real_lattice))
        self.number_of_atoms = self.atom_positions_red.shape[0]
        self.num_wann = 3*self.number_of_atoms
        self.wannier_centers_red = np.array( [atom for atom in self.atom_positions_red for i in range(3)])
        self.wannier_centers_cart = self.wannier_centers_red.dot(self.real_lattice)
        self.iRvec=iRvec
        self.masses = masses
        assert (Ham_R is None)!=(dyn_mat_R is None), "need to specify either dyn_mat_R or Ham_R, but never both) "
        if Ham_R is not None:
            self.Ham_R = Ham_R
        else:
            self.Ham_R = dyn_mat_R*(Ry_eV**2) # now the units are eV**2, to be "kind of consistent" with electronic systems
            for i in range(self.number_of_atoms):
                for j in range(self.number_of_atoms):
                    self.Ham_R[3*i:3*i+3, 3*j:3*j+3, :] /= np.sqrt(masses[i]*masses[j]) * AMU_RY
        if ndegen is not None:
            assert np.all(ndegen>0)
            self.Ham_R/=ndegen[None,None,:]
        if asr :
            self.acoustic_sum_rule()
        self.do_at_end_of_init()



    def acoustic_sum_rule(self,apply_masses = True):
        iR0 = self.iR0
        for i in range(self.number_of_atoms):
            for j in range(self.number_of_atoms):
                self.Ham_R[3*i:3*i+3, 3*j:3*j+3, :] *= np.sqrt(self.masses[i]*self.masses[j]) 
        for i in range(3):
            for j in range(3):
                for a in range(self.number_of_atoms):
                    self.Ham_R[3*a+i,3*a+j,iR0]-= self.Ham_R[3*a+i,j::3,:].sum()
        for i in range(self.number_of_atoms):
            for j in range(self.number_of_atoms):
                self.Ham_R[3*i:3*i+3, 3*j:3*j+3, :] /= np.sqrt(self.masses[i]*self.masses[j]) 

    @property
    def is_phonon(self):
        return True

