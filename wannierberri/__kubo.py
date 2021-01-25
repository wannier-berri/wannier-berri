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
from .__utility import alpha_A,beta_A
from . import __result as result

# constants
pi = constants.pi
e = constants.e
hbar = constants.hbar

# smearing functions
def Lorentzian(x, width):
    return 1.0/(pi*width) * width**2/(x**2 + width**2)

def Gaussian(x, width, adpt_smr):
    '''
    Compute 1 / (np.sqrt(pi) * width) * exp(-(x / width) ** 2)
    If the exponent is less than -200, return 0.
    An unoptimized version is the following.
        def Gaussian(x, width, adpt_smr):
            return 1 / (np.sqrt(pi) * width) * np.exp(-np.minimum(200.0, (x / width) ** 2))
    '''
    inds = abs(x) < width * np.sqrt(200.0)
    output = np.zeros_like(x)
    if adpt_smr:
        # width is array
        width_tile = np.tile(width, (x.shape[0], 1, 1))
        output[inds] = 1.0 / (np.sqrt(pi) * width_tile[inds]) * np.exp(-(x[inds] / width_tile[inds])**2)
    else:
        # width is number
        output[inds] = 1.0 / (np.sqrt(pi) * width) * np.exp(-(x[inds] / width)**2)
    return output



# Fermi-Dirac distribution
def FermiDirac(E, mu, kBT):
    if kBT == 0:
        return 1.0*(E <= mu)
    else:
        arg = np.maximum(np.minimum((E-mu)/kBT, 700.0), -700.0)
        return 1.0/(np.exp(arg) + 1)



def fermiSurf(EF, E, kBT):   # returns arra [iF, n ]
    arg=abs(EF[:,None]-E[None,:])
    if kBT <= 0:
        if len(EF)<2: 
            raise RuntimeError('cannot evaluate kBT=0 with single Fermi level')
        dE=EF[1]-EF[0]
        return 1.0*( arg<dE/2 )/dE
    else:
        arg/=2*kBT
        sel= (arg<20)
        res=np.zeros( EF.shape+E.shape )
        res[sel]=1./(4*kBT*np.cosh(arg[sel])**2)
        return res



def kubo_sum_elements(x, y, num_wann):
    # Compute np.einsum('mnab(c),wnm->wab(c)', x, y).
    # This implementation is much faster than calling np.einsum.
    assert y.shape[1] == num_wann
    assert y.shape[2] == num_wann
    y_reshape = y.reshape((-1, num_wann**2))

    assert x.shape == (num_wann, num_wann, 3, 3) or x.shape == (num_wann, num_wann, 3, 3, 3)
    if x.shape == (num_wann, num_wann, 3, 3):
        x_reshape = x.reshape((num_wann**2, 3 * 3))
        return (y_reshape @ x_reshape).reshape((-1, 3, 3))
    else:
        x_reshape = x.reshape((num_wann**2, 3 * 3 * 3))
        return (y_reshape @ x_reshape).reshape((-1, 3, 3, 3))

def opt_conductivity(data, Efermi,omega=None,  kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', adpt_smr=False,
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=0.1, adpt_smr_min=1e-15, conductivity_type='kubo', SHC_type='ryoo',
               Hermitian = True, 
               antiHermitian = True):
    '''
    Calculates the optical conductivity according to the Kubo-Greenwood formula.

    Arguments:
        data            instance of :class:~wannierberri.__Data_K.Data_K representing a single FFT grid in the BZ
        Efermi          list of chemical potentials in units of eV
        omega           list of frequencies in units of eV/hbar
        kBT             temperature in units of eV/kB
        smr_fixed_width smearing paramters in units of eV
        smr_type        analytical form of broadened delta function ('Gaussian' or 'Lorentzian')
        adpt_smr        specifies whether to use an adaptive smearing parameter (for each pair of states)
        adpt_smr_fac    prefactor for the adaptive smearing parameter
        adpt_smr_max    maximal value of the adaptive smearing parameter
        adpt_smr_min    minimal value of the adaptive smearing parameter
        conductivity_type type of optical conductivity ('kubo', 'SHC'(spin Hall conductivity), 'tildeD' (finite-frequency Berry curvature dipole) )
        SHC_type        'ryoo': PRB RPS19, 'qiao': PRB QZYZ18
        Hermitian       evaluate the Hermitian part
        antiHermitian   evaluate anti-Hermitian part 

    Returns:    a list of (complex) optical conductivity 3 x 3 (x 3) tensors (one for each frequency value).
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


    if conductivity_type == 'kubo' :
        sigma_H  = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3), dtype=np.dtype('complex128'))
        sigma_AH = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3), dtype=np.dtype('complex128'))
        rank=2
    elif conductivity_type == 'SHC':
        sigma_H  = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3, 3), dtype=np.dtype('complex128'))
        sigma_AH = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3, 3), dtype=np.dtype('complex128'))
        rank=3
    elif conductivity_type == 'tildeD' :
        tildeD  = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3), dtype=float)
        rank=2


    # prefactor for correct units of the result (S/cm)
    pre_fac = e**2/(100.0 * hbar * data.NKFFT_tot * data.cell_volume * constants.angstrom)

    if adpt_smr: 
        cprint("WARNING: Adaptive smearing is an experimental feature and has not been extensively tested.", 'red')


    # iterate over ik, simple summation
    for ik in range(data.NKFFT_tot):
        # E - omega
        E  = data.E_K[ik] # energies [n] in eV
        dE = E[None,:] - E[:, None] # E_m(k) - E_n(k) [n, m]
        delta_arg = dE[None,:,:] - omega[:,None,None] # argument of delta function [iw, n, m]

        # smearing
        if adpt_smr: # [iw, n, m]
            eta = smr_fixed_width
            delE = data.delE_K[ik] # energy derivatives [n, a] in eV*angstrom
            ddelE = delE[None,:] - delE[:, None] # delE_m(k) - delE_n(k) [n, m, a]
            # Stepan :
            eta = np.maximum(adpt_smr_min, np.minimum(adpt_smr_max,
                adpt_smr_fac * np.abs(ddelE.dot(data.Kpoint.dK_fullBZ_cart.T)).max(axis=-1) ))[None, :, :]
            # Patrick's version: 
            # eta = np.maximum(adpt_smr_min, np.minimum(adpt_smr_max,
            #     adpt_smr_fac * np.linalg.norm(ddelE, axis=2) * np.max(data.Kpoint.dK_fullBZ)))[None, :, :]
        else:
            eta = smr_fixed_width # number

        # broadened delta function [iw, n, m]
        if smr_type == 'Lorentzian':
            delta = Lorentzian(delta_arg, eta)
        elif smr_type == 'Gaussian':
            delta = Gaussian(delta_arg, eta, adpt_smr)
        else:
            cprint("Invalid smearing type. Fallback to Lorentzian", 'red')
            delta = Lorentzian(delta_arg, eta)


        if conductivity_type == 'kubo':
            # generalized Berry connection matrix
            A = data.A_H[ik] # [n, m, a] in angstrom
#            B = data.A_H[ik]
        elif conductivity_type == 'SHC':
            B = - 1j*data.A_H[ik]
            if SHC_type == 'qiao':
                A = 0.5 * (data.shc_B_H[ik] + data.shc_B_H[ik].transpose(1,0,2,3).conj())
            elif SHC_type == 'ryoo':
                # PRB RPS19 Eqs. (21) and (26), j=(1/2)(VV*SS - i(E*SA - SHA) + adj. part)
                VV = data.V_H[ik] # [n,m,a]
                SS = data.S_H[ik]   # [n,m,b]
                SA = data.SA_H[ik]  # [n,m,a,b]
                SHA = data.SHA_H[ik]# [n,m,a,b]
                A = (np.matmul(VV.transpose(2,0,1)[:,None,:,:],SS.transpose(2,0,1)[None,:,:,:])
                    + np.matmul(SS.transpose(2,0,1)[None,:,:,:],VV.transpose(2,0,1)[:,None,:,:])).transpose(2,3,0,1)
                A += -1j * (E[None,:,None,None]*SA - SHA)
                SA_adj = SA.transpose(1,0,2,3).conj()
                SHA_adj = SHA.transpose(1,0,2,3).conj()
                A += 1j *  (E[:,None,None,None]*SA_adj - SHA_adj)
                A /= 2.0
            else:
                print("Invalid SHC type. ryoo or qiao.")
        elif  conductivity_type == 'tildeD':
            rfac=dE[None,:,:]/(dE[None,:,:]+omega[:,None,None]+1j*eta)
            rfac=(rfac+rfac.transpose(0,2,1).conj()).real/2
            A = data.A_H[ik]
            AA =A[:,:,:,None]*A.transpose(1,0,2)[:,:,None,:]
            imAA=np.imag(AA[:,:,alpha_A,beta_A] - AA[:,:,beta_A,alpha_A] )
            degen=np.zeros(E.shape,dtype=bool)
            degen[:-1][(E[1:]-E[:-1])<data.degen_thresh]=True
            degen[np.where(degen[:-1])[0]+1]=True
            tildeOmega= ( -rfac[:,:,:,None]*imAA[None,:,:,:]).sum(axis=2)    # [iw,n,c]
            tildeOmega[:,degen,:]=0

        if conductivity_type == 'tildeD':
            V =  data.delE_K[ik] 
            fs=fermiSurf(Efermi, E, kBT)
#            print("shapes",fs.shape,V.shape,tildeOmega.shape)
#            print("shapes",fermiSurf(Efermi, E, kBT)[:,None,:,None,None].shape,V [None,None,:,:,None].shape,tildeOmega[None,:,:,None,:].shape,tildeD.shape)
#            print("shapes",(fermiSurf(Efermi, E, kBT)[:,None,:,None,None]*V [None,None,:,:,None]*tildeOmega[None,:,:,None,:]).shape,tildeD.shape)
#        degen= ( abs(dE)<=data.degen_thresh )
            
            tildeD+= np.sum(  fermiSurf(Efermi, E, kBT)[:,None,:,None,None]
                                *V [None,None,:,:,None]
                                  *tildeOmega[None,:,:,None,:], axis=2)

            delta_arg = dE[None,:,:] + omega[:,None,None]

        else:
       # iterate over Fermi levels, TODO: think if it is optimizable
          for iEF,EF in enumerate(Efermi):
             # occupation
            fE = FermiDirac(E, EF, kBT) # f(E_m(k)) - f(E_n(k)) [n]
            dfE = fE[None,:] - fE[:, None] # [n, m]
    
            if conductivity_type == 'kubo':
                # real part of energy fraction [iw, n, m]
                re_efrac = delta_arg/(delta_arg**2 + eta**2)
                # temporary variables for computing conductivity tensor
                tmp1 = dfE * dE
                tmp2 = np.einsum('nma,mnb->nmab', A, A)
                tmp3 = tmp1[:, :, None, None] * tmp2
                # Hermitian part of the conductivity tensor
                sigma_H [iEF] += -1 * pi * pre_fac * kubo_sum_elements(tmp3, delta, data.num_wann)
                # anti-Hermitian part of the conductivity tensor
                sigma_AH[iEF]+= 1j * pre_fac * kubo_sum_elements(tmp3, re_efrac, data.num_wann)
    
            elif conductivity_type == 'SHC':
                delta_minus = delta
                delta_plus  = delta.transpose( (0,2,1) )
                cfac2 = delta_minus - delta_plus
                cfac1 = np.real(-dE[None,:,:]/(dE[None,:,:]**2-(omega[:,None,None]+1j*eta)**2))
                temp1 = dfE[None,:,:]*cfac1
                temp2 = dfE[None,:,:]*cfac2
                imAB = np.imag(np.einsum('nmac,mnb->nmabc',A,B))
                sigma_H [iEF] += 1j * pi * pre_fac * kubo_sum_elements(imAB, temp2, data.num_wann) / 4.0
                sigma_AH[iEF] += pre_fac * kubo_sum_elements(imAB, temp1, data.num_wann) / 2.0

                

    if conductivity_type == 'kubo':
        # TODO: optimize by just storing independent components or leave it like that?
        # 3x3 tensors [iw, a, b]
        sigma_sym = np.real(sigma_H) + 1j * np.imag(sigma_AH) # symmetric (TR-even, I-even)
        sigma_asym = np.real(sigma_AH) + 1j * np.imag(sigma_H) # ansymmetric (TR-odd, I-even)

        # return result dictionary
        return result.EnergyResultDict({
            'sym':  result.EnergyResult([Efermi,omega], sigma_sym, TRodd=False, Iodd=False, rank=rank),
            'asym': result.EnergyResult([Efermi,omega], sigma_asym, TRodd=True, Iodd=False, rank=rank)
        }) # the proper smoother is set later for both elements

    elif conductivity_type == 'SHC':
        sigma_SHC = np.real(sigma_AH) + 1j * np.imag(sigma_H)
        return result.EnergyResult([Efermi,omega], sigma_SHC, TRodd=False, Iodd=False, rank=rank)

    elif conductivity_type == 'tildeD':
        pre_fac = 1./ (data.NKFFT_tot * data.cell_volume )
        return result.EnergyResult([Efermi,omega], tildeD*pre_fac, TRodd=False, Iodd=True, rank=rank)


def opt_SHCqiao(data, Efermi, omega=0, kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', adpt_smr=False,
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=0.1, adpt_smr_min=1e-15):
    return opt_conductivity(data, Efermi, omega, kBT, smr_fixed_width, smr_type, adpt_smr,
                adpt_smr_fac, adpt_smr_max, adpt_smr_min, conductivity_type='SHC', SHC_type='qiao')

def opt_SHCryoo(data, Efermi, omega=0,  kBT=0, smr_fixed_width=0.1, smr_type='Lorentzian', adpt_smr=False,
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=0.1, adpt_smr_min=1e-15):
    return opt_conductivity(data, Efermi, omega,  kBT, smr_fixed_width, smr_type, adpt_smr,
                adpt_smr_fac, adpt_smr_max, adpt_smr_min, conductivity_type='SHC', SHC_type='ryoo')


def tildeD(data, Efermi, omega=0,  **parameters ):
    return opt_conductivity(data, Efermi, omega,   conductivity_type='tildeD', **parameters )


