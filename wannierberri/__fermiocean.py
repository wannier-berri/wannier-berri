import numpy as np
from scipy import constants
from collections import defaultdict
from .__utility import  warning
from .__tetrahedron import weights_parallelepiped  as weights_tetra  
from . import __result as result
from . import __trace_formula as trF
from scipy.constants import Boltzmann, elementary_charge, hbar, electron_mass, physical_constants, angstrom
from math import ceil
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom
fac_ahc = -1e8 * elementary_charge ** 2 / hbar

degen_thresh=1e-5

def AHC(data_K,Efermi,kpart=None,tetra=False):
    fac_ahc  = -1.0e8*elementary_charge**2/hbar
    return Omega_tot(data_K,Efermi,kpart=kpart,tetra=tetra)*fac_ahc

def cumdos(data_K,Efermi,kpart=None,tetra=False):
    return iterate_kpart(trF.identity,data_K,Efermi,kpart,tetra)*data_K.cell_volume


def berry_dipole(data_K,Efermi,kpart=None,tetra=False):
    return iterate_kpart(trF.derOmega,data_K,Efermi,kpart,tetra)

def Omega_tot(data_K,Efermi,kpart=None,tetra=False):
    return iterate_kpart(trF.Omega,data_K,Efermi,kpart,tetra)


##################################
### The private part goes here  ##
##################################

def iterate_kpart(formula_fun,data_K,Efermi,kpart=None,tetra=False,dtype=float,**parameters): 
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
                    data_K,op,ed,
                    Efermi,
                    ndim=f0.ndim,
                    dtype=f0.dtype,
                    tetra=tetra ) ()
                   for op,ed in zip(borders,borders[1:]) 
               ) / (data_K.NKFFT_tot * data_K.cell_volume)

    return result.EnergyResult(Efermi,res, TRodd=f0.TRodd, Iodd=f0.Iodd )


## Note - there is probalby no point to create an object and use it only once
## it is done for visual separation of "preparation" and "evaluation"


class  FermiOcean():

    def __init__(self , formula , data_K, op, ed, Efermi, ndim, dtype,tetra):
        """  
        mat_list  list/tuple of type ( ('nl',A, [ ('ln',B1) , ('lpn',B2,C2) , ('lmn',B3,C3),...] )
                               or  ( ('nm',A) )  or (('n',A) )
        EK-  energies of bands
        Emin,Emax  - minimal and maximal energies of interest
        cartesian dimensiopns of A,B and C should match
        returns an array (by k-point) of dictionaries {E:value}  
        wheree E is the energy of a state, value - the returned array (result)
        """

        Emin=Efermi[ 0]
        Emax=Efermi[-1]
        self.Efermi=Efermi
        self.tetra=tetra
        self.nk=ed-op
        # get a list [{ib:W} for ik in op:ed]  
        if self.tetra:
            self.weights=data_K.tetraWeights.weights_allbands(Efermi,der=0)   # here W is array of shape Efermi
        else:
            self.weights=data_K.get_bands_in_range(Emin,Emax,op,ed) # here W is energy

        self.__evaluate_traces(formula,[list(sorted(w.keys())) for w in self.weights], ndim, dtype)

        # here we need to check if there are degenerate states.
        # if so - include only the upper band
        # the new array will have energies as keys instead of band indices
#        for ik in range(nk):
#            val_new=defaultdict(lambdadic)
#            for ib in sorted(values[ik] ):
#                take=True
#                if ib+1 in values[ik]:
#                    if abs(EK[ik,ib+1]-EK[ik,ib])<degen_thresh:
#                        take=False
#                if take:
#                    val_new[ EK[ik,ib] ] = values[ik][ib]
#            values[ik]=val_new
#
#        self.data = values


    def __evaluate_traces(self,formula,bands, ndim, dtype):
        """formula  - TraceFormula to evaluate 
           bands = a list of lists of k-points for every 
        """

        formula.group_terms()
        mat_list=formula.term_list
        self.shape = (3,)*ndim
        self.dtype = dtype 
        lambdadic= lambda: np.zeros(((3, ) * ndim), dtype=dtype)
        self.values = [defaultdict(lambdadic ) for ik in range(self.nk)]
        for ABC in formula.term_list:
            Aind = ABC[0]
            A = ABC[1]
            if Aind == 'nl':
        #        for BC in ABC[2]:
        #            print(Aind,BC[0],BC[1].sum(),A.sum())
                Ashape = A.shape[3:]
                if len(ABC[2]) == 0:
                    raise ValueError("with 'nl' for A at least one B matrix shopuld be provided")
                for ik,bnd in enumerate(bands):
                    if self.tetra and bnd[0]>0:
                        bnd=[bnd[-1]]+list(bnd)
                    for n in bnd :
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
                        self.values[ik][n] += np.einsum('nl...,ln...->...', a, bc,optimize=True).real
            elif Aind == 'mn':
                if len(ABC[0] > 2):
                    warning("only one matrix should be given for 'mn'")
                else:
                    for ik,bnd in enumerate(bands):
                        for n in bnd :
                            self.values[ik][n] += A[ik, :n + 1, :n + 1].sum(axis=(0,1))
            elif Aind == 'n':
                if len(ABC) > 2:
                    warning("only one matrix should be given for 'n'")
                else:
                    for ik,bnd in enumerate(bands):
                        for n in bnd :
                            self.values[ik][n] += A[ik, :n + 1].sum(axis=0)
            else:
                raise RuntimeError('Wrong indexing for array A : {}'.format(Aind))


    def __call__(self) :
        result = np.zeros(self.Efermi.shape + self.shape, self.dtype)
        for ik,weights in enumerate(self.weights):
            resk = np.zeros(self.Efermi.shape + self.shape, self.dtype)
            values = self.values[ik]
            if self.tetra:
                ibndsrt=sorted(weights.keys())
                if len(ibndsrt)>0:
                    ib0=ibndsrt[0]
                    ibm=ibndsrt[-1]
                    if ib0>0:
                        resk+=np.einsum("e,...->e...",values[ib0-1],1.-weights[ib0])
                    resk+=np.einsum("e,...->e...",values[ibm],weights[ibm])
                    for ib in sorted(ibsrt[:-1]):
                        resk+=np.einsum("e,...->e...",values[ib],weights[ib+1]-weight[ib])
            else:
                resk = np.zeros(self.Efermi.shape + self.shape, self.dtype)
                for ib in sorted(weights):
                    resk[Efermi >= weights[ib]] = self.values[ik][ib]
            result += resk
        return result

""" nonabelian implementation: 

    if tetra:
        if mode == 'fermi-surface':
            weights=data.tetraWeights.weights_allbands(Efermi,der=1)
            for ik in range(data.NKFFT_tot):
                for ib,ibrange in  enumerate(data.degen[ik]):
                    if all([ibr in weights[ik] for ibr in range(*ibrange) ]):
                        weight=sum( weights[ik][ibr] for ibr in range(*ibrange) )/(ibrange[1]-ibrange[0])
                        me=np.einsum(einline,*(m[ik][ib] for m in M)).real
                        res+=np.einsum('e,...->e...',weight,me)
"""



