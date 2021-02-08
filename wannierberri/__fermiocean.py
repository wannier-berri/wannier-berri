import numpy as np
from functools import partial
from scipy import constants
from collections import Iterable, defaultdict
import inspect, sys
from .__utility import print_my_name_start, print_my_name_end, TAU_UNIT, alpha_A, beta_A, warning
from . import __result as result
from scipy.constants import Boltzmann, elementary_charge, hbar, electron_mass, physical_constants, angstrom
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom
fac_ahc = -100000000.0 * elementary_charge ** 2 / hbar

def LFAHE_spin_fsea(data, Efermi, Bdirection=None):
    factor = hbar / (2 * electron_mass) * fac_ahc
    ocean  = data.perturbation_derOmegaTr_ZeemanSpin_2( Bdirection=Bdirection, Emin=Efermi[0], Emax=Efermi[-1] ), 
    return result.EnergyResult(Efermi,ocean(Efermi,data), TRodd=False, Iodd=False)


def AHC(data,Efermi):
    fac_ahc  = -1.0e8*elementary_charge**2/hbar
    return Omega_tot(data,Efermi)*fac_ahc

def Omega_tot(data,Efermi):
    ocean  = data.Omega_ocean( Emin=Efermi[0], Emax=Efermi[-1] )
    return result.EnergyResult(Efermi,ocean(Efermi,data), TRodd=True, Iodd=False)


##################################
### The private part goes here  ##
##################################

class FermiOcean:

    def __init__(self, data_list):
        """initialize from a list of dictionaries E:value """
        self.data = data_list
        self.shape = self.data[0][(-10000000000.0)].shape
        self.dtype = self.data[0][(-10000000000.0)].dtype

    def __call__(self, Efermi,data_k=None):
        result = np.zeros(Efermi.shape + self.shape, self.dtype)
        for datak in self.data:
            resk = np.zeros(Efermi.shape + self.shape, self.dtype)
            for E in sorted(datak.keys()):
                resk[Efermi >= E] = datak[E][None]
            result += resk
        if data_k is not None: 
            result/= (data_k.NKFFT_tot * data_k.cell_volume)
        return result


def sea_from_matrix_product(mat_list, EK, Emin, Emax, ndim, dtype):
    """  
        mat_list  list/tuple of type ( ('nl',A, [ ('ln',B1) , ('lpn',B2,C2) , ('lmn',B3,C3),...] )
                               or  ( ('nm',A) )  or (('n',A) )
        EK-  energies of bands
        Emin,Emax  - minimal and maximal energies of interest
        cartesian dimensiopns of A,B and C should match
        returns an array of energies and values
        """
    EK = np.copy(EK)
    assert Emax >= Emin
#    print (mat_list)
    nk, nb = EK.shape
    bandmin = np.full(nk, 0)
    sel = EK < Emin
    bandmin[sel[:, 0]] = [max(np.where(s)[0]) for s in sel[sel[:, 0]]]
    EK[sel] = Emin - 1e-06
    bandmax = np.full(nk, 0)
    sel = EK < Emax
    bandmax[sel[:, 0]] = [max(np.where(s)[0])+1 for s in sel[sel[:, 0]]]
    lambdadic= lambda: np.zeros(((3, ) * ndim), dtype=dtype)
    values = [defaultdict(lambdadic ) for ik in range(nk)]
    for ABC in mat_list:
        Aind = ABC[0]
        A = ABC[1]
        if Aind == 'nl':
    #        for BC in ABC[2]:
    #            print(Aind,BC[0],BC[1].sum(),A.sum())
            Ashape = A.shape[3:]
            if len(ABC[2]) == 0:
                raise ValueError("with 'nl' for A at least one B matrix shopuld be provided")
            for ik in range(nk):
                for n in range(bandmin[ik], bandmax[ik]):
                    a = A[ik, :n + 1, n + 1:]
                    bc = 0
                    for BC in ABC[2]:
#                        print(Aind,BC[0])
                        if BC[0] == 'ln':
                            bc += BC[1][ik, n + 1:, :n + 1]
                        elif BC[0] == 'lpn':
                            bc += np.einsum('lp...,pn...->ln...', BC[1][ik, n + 1:, n + 1:], BC[2][ik, n + 1:, :n + 1])
                        elif BC[0] == 'lmn':
                            bc += np.einsum('lm...,mn...->ln...', BC[1][ik, n + 1:, :n + 1], BC[2][ik, :n + 1, :n + 1])
                        else:
                            raise ValueError('Wrong index for B,C : {}'.format(BC[0]))
                    values[ik][n] += np.einsum('nl...,ln...->...', a, bc).real
        elif Aind == 'mn':
            if len(ABC[0] > 2):
                warning("only one matrix should be given for 'mn'")
            else:
                for ik in range(nk):
                    for n in range(bandmin[ik], bandmax[ik]):
                        values[ik][n] += A[ik, :n + 1, :n + 1].sum(axis=(0,1))
        elif Aind == 'n':
            if len(ABC) > 2:
                warning("only one matrix should be given for 'n'")
            else:
                for ik in range(nk):
                    for n in range(bandmin[ik], bandmax[ik]):
                        values[ik][n] += A[ik, :n + 1].sum(axis=0)
        else:
            raise RuntimeError('Wrong indexing for array A : {}'.format(Aind))
    # here we need to check if there are degenerate states.
    # if so - include only the upper band
    # the new array will have energies as keys instead of band indices
    for ik in range(nk):
        val_new=defaultdict(lambdadic)
        for ib in sorted(values[ik] ):
            take=True
            if ib+1 in values[ik]:
                if abs(EK[ik,ib+1]-EK[ik,ib])<1e-5:
                    take=False
            if take:
                val_new[ EK[ik,ib] ] = values[ik][ib]
        values[ik]=val_new

    return values
