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
        return 1 if E <= mu else 0
    else:
        return 1.0/(np.exp((E-mu)/kBT) + 1)


def calcKubo(data, hbaromega=0, mu=0, kBT=0, eta=0.1, smearing_type='Lorentzian', adpt_eta=False,
                adpt_eta_fac=np.sqrt(2), adpt_eta_max=1.0):
    '''
    Calculates the optical conductivity according to the Kubo-Greenwood formula.
    
    Arguments:
        data            instance of __data_dk.Data_dk representing a single point in the BZ
        hbaromega       value or list of frequencies in units of eV/hbar
        mu              chemical potential in units of eV/hbar
        kBT             temperature in units of eV/kB
        eta             smearing paramters in units of eV
        smearing_type   analytical form of broadened delta function ('Gaussian' or 'Lorentzian')
        adaptive_eta    specifies whether to use an adaptive value for eta (for each pair of states)
        
    Returns:    a list of (complex) optical conductivity 3 x 3 tensors (one for each frequency value) with
                each entry in the tensor being an array of two elements, the real and imaginary part.
                The result is given in SI units.
    '''
    
    # data gives results in terms of
    # ik = index enumerating the k points in Data_dk object
    # m,n = indices enumerating the eigenstates/eigenvalues of H(k)
    # a,b = cartesian coordinate
    # additionally the result will include
    # iw = index enumerating the frequency values
    # ri = index for real and imaginary parts (0 -> real, 1 -> imaginary)

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
    # TODO: optimize for T = 0?
    
    # argument of delta function
    delta_arg = dE - hbaromega[:, np.newaxis, np.newaxis, np.newaxis] # [iw, ik, n, m]
    
    # adaptive eta
    if adpt_eta: # [iw, ik, n, m]
        adeta = eta # TODO: implementation missing
        #adeta = np.minimum(adpt_eta_max, adpt_eta_fac * np.linalg.norm(ddelE, axis=3) * )[np.newaxis,:]
    else:
        adeta = eta # number
    
    # broadened delta function [iw, ik, n, m]
    if smearing_type == 'Lorentzian':
        delta = Lorentzian(delta_arg, adeta)
    elif smearing_type == 'Gaussian':
        delta = Gaussian(delta_arg, adeta)
    else:
        cprint("Invalid smearing type. Fallback to Lorentzian", 'orange')
        delta = Lorentzian(delta_arg, adeta)
    
    # real part of energy fraction
    re_efrac = delta_arg/(delta_arg**2 + adeta**2) # [iw, ik, n, m]

    # Hermitian part of the conductivity tensor
    sigma_H = -1 * pi * np.einsum('knm,knm,knma,kmnb,wknm->wab', dfE, dE, A, A, delta) # [iw, a, b]
    
    # anit-Hermitian part of the conductivity tensor
    sigma_AH = 1j * np.einsum('knm,knm,wknm,knma,kmnb->wab', dfE, dE, re_efrac, A, A) # [iw, a, b]
    
    # 3x3 tensor result
    sigma = e**2/(hbar * data.NKFFT_tot * data.cell_volume * constants.angstrom) * (sigma_H + sigma_AH) # [iw, a, b]
    
    # get real and imaginary parts [iw, a, b, reim]
    data = np.stack((np.real(sigma), np.imag(sigma)), axis=-1)
    
    # return result
    return EnergyResult(hbaromega, data, TRodd=False, Iodd=True, rank=3) # the proper smoother is set later
    
    # alternative -> symmetric and antisymmetric components