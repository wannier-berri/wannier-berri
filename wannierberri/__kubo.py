#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
#------------------------------------------------------------
#                                                            #
# author of this file: Patrick M. Lenggenhager               #
#                                                            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable
import functools
from termcolor import cprint 

from . import __result as result

# constants
pi = constants.pi
e = constants.e
hbar = constants.hbar

# smearing functions
def Lorentzian(x, width):
    return 1.0/(pi*width) * width**2/(x**2 + width**2)
def Gaussian(x, width):
    arg = np.minimum(200.0, (x/width)**2)
    return 1.0/(np.sqrt(pi) * width) * np.exp(-1*arg)
    
# Fermi-Dirac distribution
def FermiDirac(E, mu, kBT):
    if kBT == 0:
        return 1.0*(E <= mu)
    else:
        arg = np.maximum(np.minimum((E-mu)/kBT, 700.0), -700.0)
        return 1.0/(np.exp(arg) + 1)


def opt_conductivity(data, omega=0, mu=0, kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', adpt_smr=False,
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=0.1, adpt_smr_min=1e-15, conductivity_type='AHC'):
    '''
    Calculates the optical conductivity according to the Kubo-Greenwood formula.
    
    Arguments:
        data            instance of __data_dk.Data_dk representing a single point in the BZ
        omega           value or list of frequencies in units of eV/hbar
        mu              chemical potential in units of eV/hbar
        kBT             temperature in units of eV/kB
        smr_fixed_width smearing paramters in units of eV
        smr_type        analytical form of broadened delta function ('Gaussian' or 'Lorentzian')
        adpt_smr        specifies whether to use an adaptive smearing parameter (for each pair of states)
        adpt_smr_fac    prefactor for the adaptive smearing parameter
        adpt_smr_max    maximal value of the adaptive smearing parameter
        adpt_smr_min    minimal value of the adaptive smearing parameter
        
    Returns:    a list of (complex) optical conductivity 3 x 3 tensors (one for each frequency value).
                The result is given in S/cm.
    '''
    
    # data gives results in terms of
    # ik = index enumerating the k points in Data_dk object
    # m,n = indices enumerating the eigenstates/eigenvalues of H(k)
    # a,b = cartesian coordinate
    # additionally the result will include
    # iw = index enumerating the frequency values
    # ri = index for real and imaginary parts (0 -> real, 1 -> imaginary)
    
    # TODO: optimize for T = 0? take only necessary elements
    

    
    # frequency
    if not isinstance(omega, Iterable):
        omega = np.array([omega])

    if conductivity_type == 'AHC':    
        sigma_H = np.zeros((omega.shape[0], 3, 3), dtype=np.dtype('complex128'))
        sigma_AH = np.zeros((omega.shape[0], 3, 3), dtype=np.dtype('complex128'))
        rank=2
    elif conductivity_type == 'SHC':
        sigma_H = np.zeros((omega.shape[0], 3, 3, 3), dtype=np.dtype('complex128'))
        sigma_AH = np.zeros((omega.shape[0], 3, 3, 3), dtype=np.dtype('complex128'))
        rank=3

    # prefactor for correct units of the result (S/cm)
    pre_fac = e**2/(100.0 * hbar * data.NKFFT_tot * data.cell_volume * constants.angstrom)

    # iterate over ik, simple summation
    for ik in range(data.NKFFT_tot):
        # energy
        E = data.E_K[ik] # energies [n] in eV
        dE = E[np.newaxis,:] - E[:, np.newaxis] # E_m(k) - E_n(k) [n, m]

         # occupation
        fE = FermiDirac(E, mu, kBT) # f(E_m(k)) - f(E_n(k)) [n]
        dfE = fE[np.newaxis,:] - fE[:, np.newaxis] # [n, m]
        
        fEw= FermiDirac(E[np.newaxis,:],omega[:,np.newaxis],kBT) #[iw,n] for shifted Fermi energies
        dfEw= fEw[:,np.newaxis,:] - fEw[:,:,np.newaxis] #[iw,n,m]
        
        if conductivity_type == 'AHC':
            # generalized Berry connection matrix
            A = data.A_H[ik] # [n, m, a] in angstrom
            B = data.A_H[ik]
        elif conductivity_type == 'SHC':
            delH = data.V_H[ik] # [n,m,a]
            SS = data.S_H[ik]   # [n,m,b]
            SA = data.SA_H[ik]  # [n,m,a,b]
            SHA = data.SHA_H[ik]# [n,m,a,b]
            A = np.einsum('nla,lmb->nmab',delH,SS)+np.einsum('nlb,lma->nmab',SS,delH)
            #A = np.zeros((data.num_wann,data.num_wann,3,3),dtype=np.dtype('complex128'))
            A += -1j * (E[np.newaxis,:,np.newaxis,np.newaxis]*SA - SHA)
            SA_adj = np.conjugate(np.swapaxes(SA,0,1))
            SHA_adj = np.conjugate(np.swapaxes(SHA,0,1))
            A += 1j *  (E[:,np.newaxis,np.newaxis,np.newaxis]*SA_adj - SHA_adj)
            A /= 2.0
            
            B = - 1j*data.A_H[ik]
        else:
            raise ValueError("The current available types of optical conductivities are AHC and SHC.")
       
        # E - omega
        delta_arg = dE[np.newaxis,:,:] - omega[:,np.newaxis,np.newaxis] # argument of delta function [iw, n, m]

        # smearing
        if adpt_smr: # [iw, n, m]
            eta = smr_fixed_width
            delE = data.delE_K[ik] # energy derivatives [n, a] in eV*angstrom
            ddelE = delE[np.newaxis,:] - delE[:, np.newaxis] # delE_m(k) - delE_n(k) [n, m, a]
            eta = np.maximum(adpt_smr_min, np.minimum(adpt_smr_max,
                adpt_smr_fac * np.linalg.norm(ddelE, axis=2) * np.max(data.Kpoint.dK_fullBZ)))[np.newaxis, :, :]
        else:
            eta = smr_fixed_width # number
        
        # Hermitian part of the conductivity tensor
        # broadened delta function [iw, n, m]
        if smr_type == 'Lorentzian':
            delta = Lorentzian(delta_arg, eta)
        elif smr_type == 'Gaussian':
            delta = Gaussian(delta_arg, eta)
        else:
            cprint("Invalid smearing type. Fallback to Lorentzian", 'orange')
            delta = Lorentzian(delta_arg, eta)
        

        if conductivity_type == 'AHC':
            # Hermitian part of the conductivity tensor
            sigma_H += -1 * pi * pre_fac * np.einsum('nm,nm,nma,mnb,wnm->wab', dfE, dE, A, B, delta) # [iw, a, b]
            # free memory
            del delta
            # anti-Hermitian part of the conductivity tensor
            re_efrac = delta_arg/(delta_arg**2 + eta**2) # real part of energy fraction [iw, n, m]
            sigma_AH += 1j * pre_fac * np.einsum('nm,nm,wnm,nma,mnb->wab', dfE, dE, re_efrac, A, B) # [iw, a, b]
        elif conductivity_type == 'SHC':
    #         # Hermitian part of the conductivity tensor
    #         sigma_H += -1j * pi * pre_fac * np.einsum('nm,nmac,mnb,wnm->wabc', dfE, A, B, delta) # [iw, a, b, c]
    #         # free memory
    #         del delta
    #         # anti-Hermitian part of the conductivity tensor
    #         #re_efrac = delta_arg/(delta_arg**2 + eta**2) # real part of energy fraction [iw, n, m]
    #         re_efrac = 1/delta_arg
    #         sigma_AH += -pre_fac * np.einsum('nm,wnm,nmac,mnb->wabc', dfE, re_efrac, A, B) # [iw, a, b, c]
            cfac2=delta
            cfac1=np.real(0.5/(delta_arg-1j*eta))
            delta_arg = dE[np.newaxis,:,:] + omega[:,np.newaxis,np.newaxis]
            if smr_type == 'Lorentzian':
                delta = Lorentzian(delta_arg, eta)
            elif smr_type == 'Gaussian':
                delta = Gaussian(delta_arg, eta)
            else:
                cprint("Invalid smearing type. Fallback to Lorentzian", 'orange')
                delta = Lorentzian(delta_arg, eta)
            cfac2+=-delta 
            cfac1+=np.real(0.5/(delta_arg+1j*eta))

    #        cfac1=np.real(dE[np.newaxis,:,:]/(dE[np.newaxis,:,:]**2-(omega[:,np.newaxis,np.newaxis]+1j*eta)**2))
            imAB=np.imag(A[:,:,:,np.newaxis,:]*np.swapaxes(B,0,1)[:,:,np.newaxis,:,np.newaxis])
    #        imAB=np.ones((data.num_wann,data.num_wann,3,3,3),dtype=np.dtype('complex128'))

            sigma_H += 1j * pi * pre_fac * np.einsum('mn,wmn,nmabc->wabc',dfE,cfac2,imAB) / 4.0
            sigma_AH += pre_fac * np.einsum('mn,wmn,nmabc->wabc',dfE,cfac1,imAB) / 2.0
            del delta
        else:
            raise ValueError("The current available types of optical conductivities are AHC and SHC.")
        

        # free memory
        #del re_efrac
        del delta_arg
        del dfE
        del dE
        
    # TODO: optimize by just storing independent components or leave it like that?
    # 3x3 tensors [iw, a, b] or [iw,a,b,c]
    sigma_sym = np.real(sigma_H) + 1j * np.imag(sigma_AH) # symmetric (TR-even, I-even)
    sigma_asym = np.real(sigma_AH) + 1j * np.imag(sigma_H) # ansymmetric (TR-odd, I-even)
    
    # return result dictionary
    return result.EnergyResultDict({
        'sym':  result.EnergyResult(omega, sigma_sym, TRodd=False, Iodd=False, rank=rank),
        'asym': result.EnergyResult(omega, sigma_asym, TRodd=True, Iodd=False, rank=rank)
    }) # the proper smoother is set later for both elements


def opt_SHC(data, omega=0, mu=0, kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', adpt_smr=False,
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=0.1, adpt_smr_min=1e-15):
    return opt_conductivity(data, omega, mu, kBT, smr_fixed_width, smr_type, adpt_smr,
                adpt_smr_fac, adpt_smr_max, adpt_smr_min, conductivity_type='SHC')