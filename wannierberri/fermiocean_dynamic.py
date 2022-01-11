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


def occunocc(E1,E2,Efermi,omega):
    return (E1<Efermi.max()) and (E2>Efermi.min()) and (<E2-E1<omega.max()+



class Formula_dyn_ident():
    def __inti__(self):
        self.TRodd = False
        self.Iodd =False
        self.ndim = 0

def jdos(Efermi,omegasigma=0.1)
    def nonzero(E1,E2,Efermi,omega):
        return (E1<Efermi.max()) and (E2>Efermi.min()) and (omega.min()-5*sigma<E2-E1<omega.max()+5*sigma)
    def energy_factor(E1,E2,Efermi,omega):
        res = np.zeros((len(Efermi),len(omega))
        gauss = Gaussian(E2-E1-omega,sigma,adpt_smr=False)
        res[E1<Efermi<E2] = gauss[None,:]
        return res
    return FermiOcean_dynamic


##################################
### The private part goes here  ##
##################################




def FermiOcean_dynamic(self , formula , data_K,  Efermi, omega, energy_factor_fun, nonzero_fun=None, tetra=False,weightfun = None, degen_thresh=1e-4,degen_Kramers=False):
    """ 
    So far make it a separate class, later make the static as particular case of this.
    formula should have a trace_ln(ik,inn_l,inn_n) method 
    fder derivative of fermi distribution . 0: fermi-sea, 1: fermi-surface 2: f''
    """

    if tetra : 
        raise NotImplementedError("tetrahedron method is not implemented for dynamical variables")

    result_shape = (len(Efermi),len(omega))+(3,)*formula.ndim
    result  = np.zeros(result_shape,formula.dtype)
    
    for ik in range(data_K.NKFFTtot):
        energy_factor = EnergyFactor(self.data_K,ik,self.Efermi,self.omega,energy_factor_fun,nonzero_fun=None,degen_thresh,degen_Kramers):
#        matrix_elements = {(inn1,inn2):self.formula.trace_ln(ik,inn1,inn2) for (inn1,inn2) in self.energy_factor().keys()}
        for band_pair,EE in energy_factor.degen_group_pairs:
            factor = energy_factor.get_value(EE[0],EE[1])
            matrix_element = formula.trace_ln(inn1,inn2)
            restot+=np.einsum( "ew,...->ew...",factor,matrix_element )
            
    restot *= self.final_factor
    return result.EnergyResult([Efermi,omega],restot, TRodd=formula.TRodd, Iodd=formula.Iodd )




class EnergyFactor():

    """
    energy_factor(E1,E2,Efermi,omega) - gives the energy factor
    nonzero(E1,E2,Efermi,omega) - gives True if the energy factor is nonzero for at least one Fermi level and frequency
    """
    
    def __init__(self,ik,data_K,Efermi,omega,energy_factor_fun,nonzero_fun=None,degen_thresh=1e-4,degen_Kramers=False):

        self.Efermi = Efermi
        self.omega = omega
        self.get_value = functools.partial(energy_factor_fun,Efermi = self.Efermi, omega = self.omega) 
        if nonzero is None:
            nonzero_loc = lambda E1,E2 : no.any(abs(self.energy_factor(E1,E2) ) > 1e-10)
        else:
            nonzero_loc = functools.partial(nonzero_fun,Efermi = self.Efermi, omega = self.omega) 
        # first find degenerate groups
        # as a dictionary {(ibm1,ibm2):Em} 
        degen_groups = data_K.get_bands_in_range_groups_ik(ik,-np.Inf,np.Inf,degen_thresh=degen_thresh,degen_Kramers=degen_Kramers)
        #now find needed pairs:
        # as a dictionary {((ibm1,ibm2),(ibn1,ibn2)):(Em,En)} 
        self.degen_group_pairs= { (ibm,ibn):(Em,En) 
                                     for ibm,Em in degen_groups.items()
                                         for ibn,En in degen_groups.items()
                                             if nonzero(Em,En) }
                                             
        
        
