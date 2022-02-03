import numpy as np
from scipy.constants import  elementary_charge, hbar, electron_mass, physical_constants, angstrom
bohr_magneton = elementary_charge * hbar / (2 * electron_mass)
bohr = physical_constants['Bohr radius'][0] / angstrom
eV_au = physical_constants['electron volt-hartree relationship'][0]
Ang_SI = angstrom

from .classes import DynamicCalculator

##################################
###              JDOS           ##
##################################

class Formula_dyn_ident():

    def __init__(self,data_K):
        self.TRodd = False
        self.Iodd =False
        self.TRtrans = False
        self.ndim = 0
        
    def trace_ln(self,ik,inn1,inn2):
        return len(inn1)*len(inn2)


class JDOS(DynamicCalculator):

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
##    Optical Conductivity      ##
##################################


class Formula_OptCond():

    def __init__(self,data_K):
        A =  data_K.A_H
        self.AA = 1j*A[:,:,:,:,None]* A.swapaxes(1,2)[:,:,:,None,:]
        self.ndim = 2
        self.TRodd = False
        self.Iodd = False
        self.TRtrans = True
        
    def trace_ln(self,ik,inn1,inn2):
        return self.AA[ik,inn1].sum(axis=0)[inn2].sum(axis=0)

class  OpticalConductivity(DynamicCalculator):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.Formula = Formula_OptCond
        self.final_factor = elementary_charge**2/(100.0 * hbar * angstrom)

    
    def energy_factor(self,E1,E2):
        delta_arg_12 = E2 - E1 - self.omega # argument of delta function [iw, n, m]
        cfac = 1./(delta_arg_12-1j*self.smr_fixed_width) 
        if self.smr_type!='Lorentzian':
            cfac.imag = np.pi*self.smear(delta_arg_12)
        dfE = self.FermiDirac(E2)-self.FermiDirac(E1)  # [n, m]
        return dfE[:,None]*(E2-E1)*cfac[None,:]


##################################
###          SHC                ##
##################################
from wannierberri.covariant_formulak import SpinVelocity

class Formula_SHC():

    def __init__(self,data_K,SHC_type='ryoo',shc_abc = None):
        A =  SpinVelocity(data_K, SHC_type).matrix
        B = - 1j*data_K.A_H
        self.imAB = np.imag( A[:,:,:,:,None,:]* B.swapaxes(1,2)[:,:,:,None,:,None])
        self.ndim = 3
        if shc_abc is not None:
            assert len(shc_abc)==3
            a,b,c = (x-1 for x in shc_abc)
            self.imAB = self.imAB[:,:,:,a,b,c]
            self.ndim = 0
        self.TRodd = False
        self.Iodd = False
        self.TRtrans = False
        
    def trace_ln(self,ik,inn1,inn2):
        return self.imAB[ik,inn1].sum(axis=0)[inn2].sum(axis=0)

class _SHC(DynamicCalculator):

    def __init__(self,SHC_type="ryoo",shc_abc=None,**kwargs):
        super().__init__(**kwargs)
        self.formula_kwargs = dict(SHC_type=SHC_type,shc_abc=shc_abc)
        self.Formula = Formula_SHC
        self.final_factor = elementary_charge**2/(100.0 * hbar * angstrom)


    def energy_factor(self,E1,E2):
        delta_minus = self.smear(E2 - E1 - self.omega)
        delta_plus  = self.smear(E1 - E2 - self.omega)
        cfac2 = delta_plus - delta_minus   # TODO : for Lorentzian do the real and imaginary parts together
        cfac1 = np.real( (E1-E2)/((E1-E2)**2-(self.omega+1j*self.smr_fixed_width)**2) )
        cfac = (2*cfac1 + 1j*np.pi*cfac2)/4.
        dfE = self.FermiDirac(E2)-self.FermiDirac(E1)  # [n, m]
        return dfE[:,None]*cfac[None,:]



class SHC(_SHC):
    "a more laconic implementation of the energy factor"

    def energy_factor(self,E1,E2):
        delta_arg_12 = E1 - E2 - self.omega # argument of delta function [iw, n, m]
        cfac = 1./(delta_arg_12-1j*self.smr_fixed_width) 
        if self.smr_type!='Lorentzian':
            cfac.imag = np.pi*self.smear(delta_arg_12)
        dfE = self.FermiDirac(E2)-self.FermiDirac(E1)  # [n, m]
        return dfE[:,None]*cfac[None,:]/2
