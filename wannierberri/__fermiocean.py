import numpy as np
from scipy import constants
from collections import defaultdict
from .__utility import  warning
from . import __result as result
from . import __trace_formula as trF
from scipy.constants import Boltzmann, elementary_charge, hbar, electron_mass, physical_constants, angstrom
from math import ceil
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom
fac_ahc = -100000000.0 * elementary_charge ** 2 / hbar


def AHC(data_K,Efermi,kpart=None):
    fac_ahc  = -1.0e8*elementary_charge**2/hbar
    return Omega_tot(data_K,Efermi,kpart=kpart)*fac_ahc

def berry_dipole(data_K,Efermi,kpart=None):
    res =  iterate_kpart(trF.derOmega,data_K,Efermi,kpart)
    res.data= np.swapaxes(res.data,1,2)  # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
    return res

def Omega_tot(data_K,Efermi,kpart=None):    return iterate_kpart(trF.Omega,data_K,Efermi,kpart)


##################################
### The private part goes here  ##
##################################

def iterate_kpart(formula_fun,data_K,Efermi,kpart=None,dtype=float,**parameters): 
    """ formula_fun should be callable eturning a TraceFormula object 
    with first three parameters as data_K,op,ed, and the rest
    and the rest will be arbitrary keyword arguments."""
    if kpart is None : 
        kpart=data_K.NKFFT_tot
    n=data_K.NKFFT_tot//kpart
    if n>0 :
        kpart += ceil( (data_K.NKFFT_tot % kpart)/n )
    borders=list(range(0,data_K.NKFFT_tot,kpart))+[data_K.NKFFT_tot]
#    print ("processing the {} FFT grid points in {} portions of {},  last portion is {}".format(
#               data_K.NKFFT_tot,len(borders)-1,kpart,data_K.NKFFT_tot%kpart))
    Emin,Emax=Efermi[0],Efermi[-1]
    f0=formula_fun(data_K,0,0,**parameters)  # just to get the basic properties
    res= sum ( 
        FermiOcean( formula_fun(data_K,op,ed,**parameters),
                    data_K.E_K[op:ed],
                    Emin, Emax,
                    ndim=f0.ndim,
                    dtype=dtype)(Efermi)
                   for op,ed in zip(borders,borders[1:]) 
               ) / (data_K.NKFFT_tot * data_K.cell_volume)
    return result.EnergyResult(Efermi,res, TRodd=f0.TRodd, Iodd=f0.Iodd )



class FermiOcean:

    def __call__(self, Efermi) :
        result = np.zeros(Efermi.shape + self.shape, self.dtype)
        for datak in self.data:
            resk = np.zeros(Efermi.shape + self.shape, self.dtype)
            for E in sorted(datak.keys()):
                resk[Efermi >= E] = datak[E][None]
            result += resk
        return result


    def __init__(self , formula , EK, Emin, Emax, ndim, dtype):
        """  
        mat_list  list/tuple of type ( ('nl',A, [ ('ln',B1) , ('lpn',B2,C2) , ('lmn',B3,C3),...] )
                               or  ( ('nm',A) )  or (('n',A) )
        EK-  energies of bands
        Emin,Emax  - minimal and maximal energies of interest
        cartesian dimensiopns of A,B and C should match
        returns an array (by k-point) of dictionaries {E:value}  
        wheree E is the energy of a state, value - the returned array (result)
        """

        formula.group_terms()
        mat_list=formula.trace_list
        self.shape = (3,)*ndim if ndim>0 else (1,)
        self.dtype = dtype 
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
                                bc += np.einsum('lp...,pn...->ln...', BC[1][ik, n + 1:, n + 1:], BC[2][ik, n + 1:, :n + 1],optimize=True)
                            elif BC[0] == 'lmn':
                                bc += np.einsum('lm...,mn...->ln...', BC[1][ik, n + 1:, :n + 1], BC[2][ik, :n + 1, :n + 1],optimize=True)
                            else:
                                raise ValueError('Wrong index for B,C : {}'.format(BC[0]))
                        values[ik][n] += np.einsum('nl...,ln...->...', a, bc,optimize=True).real
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

        self.data = values
