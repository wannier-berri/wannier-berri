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
    return 1.0/(np.sqrt(2*pi) * width) * np.exp(-0.5 * (x/width)**2)
    
# Fermi-Dirac distribution
def FermiDirac(E, mu, kBT):
    if kBT == 0:
        return 1.0*(E <= mu)
    else:
        return 1.0/(np.exp((E-mu)/kBT) + 1)


def opt_conductivity(data, hbaromega=0, mu=0, kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', adpt_smr=False,
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=1.0):
    '''
    Calculates the optical conductivity according to the Kubo-Greenwood formula.
    
    Arguments:
        data            instance of __data_dk.Data_dk representing a single point in the BZ
        hbaromega       value or list of frequencies in units of eV/hbar
        mu              chemical potential in units of eV/hbar
        kBT             temperature in units of eV/kB
        smr_fixed_width smearing paramters in units of eV
        smr_type        analytical form of broadened delta function ('Gaussian' or 'Lorentzian')
        adpt_smr        specifies whether to use an adaptive smearing parameter (for each pair of states)
        
    Returns:    a list of (complex) optical conductivity 3 x 3 tensors (one for each frequency value).
                The result is given in SI units.
    '''
    
    # data gives results in terms of
    # ik = index enumerating the k points in Data_dk object
    # m,n = indices enumerating the eigenstates/eigenvalues of H(k)
    # a,b = cartesian coordinate
    # additionally the result will include
    # iw = index enumerating the frequency values
    # ri = index for real and imaginary parts (0 -> real, 1 -> imaginary)

    # prefactor
    pre_fac = e**2/(hbar * data.NKFFT_tot * data.cell_volume * constants.angstrom)

    # energy
    E = data.E_K # [ik, n] in eV
    delE = data.delE_K # [ik, n, a] in eV*angstrom
    
    # generalized Berry connection matrix
    A = data.A_H # [ik, n, m, a] in angstrom
    
    # frequency
    if not isinstance(hbaromega, Iterable):
        hbaromega = np.array([hbaromega])
    
    
    # E_m(k) - E_n(k)
    dE = E[:,np.newaxis,:] - E[:, :, np.newaxis] # [ik, n, m]
    
    # delE_m(k) - delE_n(k)
    ddelE = delE[:,np.newaxis,:] - delE[:, :, np.newaxis] # [ik, n, m]
    
    # f(E_m(k)) - f(E_n(k))
    fE = FermiDirac(E, mu, kBT) # [ik, n, m]
    dfE = fE[:,np.newaxis,:] - fE[:, :, np.newaxis] # [ik, n, m]
    # TODO: optimize for T = 0? take only necessary elements
    
    # argument of delta function
    delta_arg = dE - hbaromega[:, np.newaxis, np.newaxis, np.newaxis] # [iw, ik, n, m]
    
    # smearing
    if adpt_smr: # [iw, ik, n, m]
        eta = smr_fixed_width # TODO: implementation missing
        cprint("Not implemented. Fallback to fixed smearing parameter.", 'orange')
        #eta = np.minimum(adpt_smr_max, adpt_smr_fac * np.linalg.norm(ddelE, axis=3) * )[np.newaxis,:]
    else:
        eta = smr_fixed_width # number
    
    # broadened delta function [iw, ik, n, m]
    if smr_type == 'Lorentzian':
        delta = Lorentzian(delta_arg, eta)
    elif smr_type == 'Gaussian':
        delta = Gaussian(delta_arg, eta)
    else:
        cprint("Invalid smearing type. Fallback to Lorentzian", 'orange')
        delta = Lorentzian(delta_arg, eta)
    
    # real part of energy fraction
    re_efrac = delta_arg/(delta_arg**2 + eta**2) # [iw, ik, n, m]

    # Hermitian part of the conductivity tensor
    sigma_H = -1 * pi * pre_fac * np.einsum('knm,knm,knma,kmnb,wknm->wab', dfE, dE, A, A, delta) # [iw, a, b]
    
    # anit-Hermitian part of the conductivity tensor
    sigma_AH = 1j * pre_fac * np.einsum('knm,knm,wknm,knma,kmnb->wab', dfE, dE, re_efrac, A, A) # [iw, a, b]
    
    # TODO: optimize by just storing independent components or leave it like that?
    # 3x3 tensors [iw, a, b]
    sigma_sym = np.real(sigma_H) + 1j * np.imag(sigma_AH) # symmetric (TR-even, I-even)
    sigma_asym = np.real(sigma_AH) + 1j * np.imag(sigma_H) # ansymmetric (TR-odd, I-even)
    
    # return result dictionary
    return EnergyResultDict({
        'sym':  EnergyResult(hbaromega, sigma_sym, TRodd=False, Iodd=False, rank=2),
        'asym': EnergyResult(hbaromega, sigma_asym, TRodd=True, Iodd=False, rank=2)
    }) # the proper smoother is set later for both elements