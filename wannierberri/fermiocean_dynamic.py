import numpy as np
from .__utility import  alpha_A,beta_A, TAU_UNIT
from collections import defaultdict
from . import __result as result
from math import ceil
from . import covariant_formulak as frml
from .formula import FormulaProduct,FormulaProduct_2,ProductDelta
from . import covariant_formulak_basic as frml_basic
from itertools import permutations
from scipy.constants import Boltzmann, elementary_charge, hbar, electron_mass, physical_constants, angstrom
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom
from .__kubo import Gaussian, Lorentzian
from . import calculators as calc
import abc
import functools
#def occunocc(E1,E2,Efermi,omega):
#    return (E1<Efermi.max()) and (E2>Efermi.min()) and (<E2-E1<omega.max()+

def FermiDirac(E, mu, kBT):
    "here E is a number, mu is an array"
    if kBT == 0:
        return 1.0*(E <= mu)
    else:
        res = np.zeros_like(mu)
        res[mu>E+30*kBT] = 1.0
        res[mu<E-30*kBT] = 0.0
        sel = abs(mu-E)<=30*kBT
        res[sel]=1.0/(np.exp((E-mu[sel])/kBT) + 1)
        return res



class Optical2(calc.Optical,abc.ABC):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.formula_kwargs = {}
        self.Formula  = None 
        self.final_factor = 1.
        self.dtype = complex
        self.EFmin=self.Efermi.min()
        self.EFmax=self.Efermi.max()
        self.omegamin=self.omega.min()
        self.omegamax=self.omega.max()
        
        if self.smr_type == 'Lorentzian':
            self.smear = functools.partial(Lorentzian,width = self.smr_fixed_width)
        elif self.smr_type == 'Gaussian':
            self.smear = functools.partial(Gaussian,width = self.smr_fixed_width,adpt_smr = False)
        else:
            cprint("Invalid smearing type. Fallback to Lorentzian", 'red')
        self.FermiDirac = functools.partial(FermiDirac,mu = self.Efermi,kBT = self.kBT) 
        


    @abc.abstractmethod
    def energy_factor(self,E1,E2):
        pass



    def nonzero(self,E1,E2):
        """determines if 2 energies give nonzero contribution.
        may be re-defined for better efficiency """
        return  True # np.any(abs(self.energy_factor(E1,E2) ) > 1e-10)

    def __call__(self,data_K):
        formula  = self.Formula(data_K,**self.formula_kwargs)
        restot_shape = (len(self.Efermi),len(self.omega))+(3,)*formula.ndim
        restot  = np.zeros(restot_shape,self.dtype)
    
        for ik in range(data_K.nk):
            degen_groups = data_K.get_bands_in_range_groups_ik(ik,-np.Inf,np.Inf,degen_thresh=self.degen_thresh,degen_Kramers=self.degen_Kramers)
            #now find needed pairs:
            # as a dictionary {((ibm1,ibm2),(ibn1,ibn2)):(Em,En)} 
            degen_group_pairs= { (ibm,ibn):(Em,En) 
                                     for ibm,Em in degen_groups.items()
                                         for ibn,En in degen_groups.items()
                                             if self.nonzero(Em,En) }
#        matrix_elements = {(inn1,inn2):self.formula.trace_ln(ik,inn1,inn2) for (inn1,inn2) in self.energy_factor().keys()}
            for pair,EE in degen_group_pairs.items():
                factor = self.energy_factor(EE[0],EE[1])
                matrix_element = formula.trace_ln(ik,np.arange(*pair[0]),np.arange(*pair[1]))
#                restot+=np.einsum( "ew,...->ew...",factor,matrix_element )
                restot+=factor.reshape(factor.shape+(1,)*formula.ndim)*matrix_element[None,None]
        restot *= self.final_factor / (data_K.nk*data_K.cell_volume)
        return result.EnergyResult([self.Efermi,self.omega],restot, TRodd=formula.TRodd, Iodd=formula.Iodd )


##################################
###              JDOS           ##
##################################

class Formula_dyn_ident():

    def __init__(self,data_K):
        self.TRodd = False
        self.Iodd =False
        self.ndim = 0
        
    def trace_ln(self,ik,inn1,inn2):
        return len(inn1)*len(inn2)


class JDOS(Optical2):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sigma = self.smr_fixed_width
        self.Formula = Formula_dyn_ident
        self.dtype = float
    
    def nonzero(self,E1,E2):
        return (E1<self.Efermi.max()) and (E2>self.Efermi.min()) and (self.omega.min()-5*self.smr_fixed_width<E2-E1<self.omega.max()+5*self.smr_fixed_width)

    def energy_factor(self,E1,E2):
        res = np.zeros((len(self.Efermi),len(self.omega)))
        gauss = self.smear(E2-E1-self.omega,self.smr_fixed_width)
        res[(E1<self.Efermi)*(self.Efermi<E2)] = gauss[None,:]
        return res



##################################
###          SHC                ##
##################################
from .covariant_formulak import SpinVelocity

class Formula_SHC():

    def __init__(self,data_K,SHC_type='ryoo',shc_abc = None):
        A =  SpinVelocity(data_K, SHC_type).matrix
        B = - 1j*data_K.A_H
#        self.imAB = np.imag(np.einsum('knmac,kmnb->knmabc',A,B))
        self.imAB = np.imag( A[:,:,:,:,None,:]* B.swapaxes(1,2)[:,:,:,None,:,None])
        self.ndim = 3
        if shc_abc is not None:
            assert len(shc_abc)==3
            a,b,c = (x-1 for x in shc_abc)
            self.imAB = self.imAB[:,:,:,a,b,c]
            self.ndim = 0
        self.TRodd = False
        self.Iodd = False
        
    def trace_ln(self,ik,inn1,inn2):
        return self.imAB[ik,inn1].sum(axis=0)[inn2].sum(axis=0)

class SHC(Optical2):

    def __init__(self,SHC_type="ryoo",shc_abc=None,**kwargs):
        super().__init__(**kwargs)
        self.formula_kwargs = dict(SHC_type=SHC_type,shc_abc=shc_abc)
        self.Formula = Formula_SHC
        self.final_factor = elementary_charge**2/(100.0 * hbar * angstrom)

    
#    def nonzero(self,E1,E2):    
#        return (E1<self.Efermi.max()) and (E2>self.Efermi.min()) and (self.omega.min()-5*self.smr_fixed_width<E2-E1<self.omega.max()+5*self.smr_fixed_width)

    def energy_factor(self,E1,E2):
        delta_arg = E2 - E1 - self.omega # argument of delta function [iw, n, m]
        delta_minus = self.smear(E2 - E1 - self.omega)
        delta_plus  = self.smear(E1 - E2 - self.omega)
        cfac2 = delta_plus - delta_minus   # TODO : for Lorentzian do the real and imaginary parts together
        cfac1 = np.real( (E1-E2)/((E1-E2)**2-(self.omega+1j*self.smr_fixed_width)**2) )
        cfac = (2*cfac1 + 1j*np.pi*cfac2)/4.
        dfE = self.FermiDirac(E2)-self.FermiDirac(E1)  # [n, m]
        return dfE[:,None]*cfac[None,:]

    def nonzero(self,E1,E2):
        emin = self.EFmin-30*self.kBT
        if E1<emin and E2<emin:
            return False
        emax = self.EFmax+30*self.kBT
        if E1>emax and E2>emax:
            return False
        return True