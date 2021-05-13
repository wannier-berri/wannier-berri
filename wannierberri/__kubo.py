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
from collections.abc import Iterable
import functools
from termcolor import cprint
from .__utility import alpha_A,beta_A
from . import __result as result

# constants
pi = constants.pi
e = constants.e
hbar = constants.hbar
eV_seconds = 6.582119e-16

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
    # Compute np.einsum('nmab(c),wnm->wab(c)', x, y).
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
                adpt_smr_fac=np.sqrt(2), adpt_smr_max=0.1, adpt_smr_min=1e-15, shc_alpha=1, shc_beta=2, shc_gamma=3,
                shc_specification=False, conductivity_type='kubo', SHC_type='ryoo', sc_eta=0.04,
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
        shc_alpha       direction of spin current (1, 2, or 3)
        shc_beta        direction of applied electric field (1, 2, or 3)
        shc_gamma       direction of spin polarization (1, 2, or 3)
        shc_specification whether only a single component of SHC is calculated or not
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


    # prefactor for correct units of the result (S/cm)
    pre_fac = e**2/(100.0 * hbar * data.NKFFT_tot * data.cell_volume * constants.angstrom)

    if conductivity_type == 'kubo' :
        sigma_H  = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3), dtype=np.dtype('complex128'))
        sigma_AH = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3), dtype=np.dtype('complex128'))
        rank=2
    elif conductivity_type == 'SHC':
        if shc_specification:
            sigma_H = np.zeros((Efermi.shape[0],omega.shape[0]), dtype=np.dtype('complex128'))
            sigma_AH = np.zeros((Efermi.shape[0],omega.shape[0]), dtype=np.dtype('complex128'))
            rank=0
            shc_alpha = shc_alpha - 1 #In python, indices start from 0.
            shc_beta = shc_beta - 1
            shc_gamma = shc_gamma - 1
        else:
            sigma_H = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3, 3), dtype=np.dtype('complex128'))
            sigma_AH = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3, 3), dtype=np.dtype('complex128'))
            rank=3
    elif conductivity_type == 'tildeD' :
        tildeD  = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3), dtype=float)
        rank=2
    elif conductivity_type == 'shiftcurrent':
        sigma_abc = np.zeros((Efermi.shape[0],omega.shape[0], 3, 3, 3), dtype=np.dtype('complex128'))
        rank=3
        # prefactor for the shift current
        pre_fac = -1.j*eV_seconds*pi*e**3/(4.0 * hbar**(2) * data.NKFFT_tot * data.cell_volume)


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
        elif conductivity_type == 'shiftcurrent':
            B = data.A_H[ik]

            A_Hbar = data.A_Hbar[ik]
            D_H = data.D_H[ik]
            V_H = data.V_H[ik]
            A_Hbar_der = data.A_Hbar_der[ik]
            del2E_H = data.del2E_H[ik]
            dEig_inv = data.dEig_inv[ik].transpose(1,0)

            # define D using broadening parameter
            E_K  = data.E_K[ik] 
            dEig = E_K[:,None]-E_K[None,:]
            dEig_inv_Pval = dEig/(dEig**(2)+sc_eta**(2))
            D_H_Pval = -V_H*dEig_inv_Pval[:,:,None]

            # commutators
            # ** the spatial index of D_H_Pval corresponds to generalized derivative direction 
            # ** --> stored in the fourth column of output variables  
            sum_AD =  (np.einsum('nlc,lma->nmca', A_Hbar, D_H_Pval) - np.einsum('nnc,nma->nmca', A_Hbar, D_H_Pval))  \
                     -(np.einsum('nla,lmc->nmca', D_H_Pval, A_Hbar) - np.einsum('nma,mmc->nmca', D_H_Pval, A_Hbar))
            sum_HD =  (np.einsum('nlc,lma->nmca', V_H, D_H_Pval) - np.einsum('nnc,nma->nmca', V_H, D_H_Pval))  \
                     -(np.einsum('nla,lmc->nmca', D_H_Pval, V_H) - np.einsum('nma,mmc->nmca', D_H_Pval, V_H))

            # ** the spatial index of A_Hbar with diagonal terms corresponds to generalized derivative direction 
            # ** --> stored in the fourth column of output variables  
            AD_bit =     np.einsum('nnc,nma->nmac' , A_Hbar, D_H) - np.einsum('mmc,nma->nmac' , A_Hbar, D_H) \
                       + np.einsum('nna,nmc->nmac' , A_Hbar, D_H) - np.einsum('mma,nmc->nmac' , A_Hbar, D_H)
            AA_bit =     np.einsum('nnb,nma->nmab' , A_Hbar, A_Hbar) - np.einsum('mmb,nma->nmab' , A_Hbar, A_Hbar) 
            # ** this one is invariant under a<-->c
            DV_bit =     np.einsum('nmc,nna->nmca' , D_H, V_H) - np.einsum('nmc,mma->nmca' , D_H, V_H) \
                       + np.einsum('nma,nnc->nmca' , D_H, V_H) - np.einsum('nma,mmc->nmca' , D_H, V_H) 

            # generalized derivative        
            A =    A_Hbar_der + \
                   + AD_bit - 1j*AA_bit \
                   + sum_AD \
                   + 1j*(  del2E_H + sum_HD + DV_bit \
                        )*dEig_inv[:,:, np.newaxis, np.newaxis]

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
                cfac2 = delta_plus - delta_minus
                cfac1 = np.real(-dE[np.newaxis,:,:]/(dE[np.newaxis,:,:]**2-(omega[:,np.newaxis,np.newaxis]+1j*eta)**2))
                temp1 = dfE[np.newaxis,:,:]*cfac1
                temp2 = dfE[np.newaxis,:,:]*cfac2
                if shc_specification:
                    imAB = np.imag(np.einsum('nmac,mnb->nmabc',A,B))[:, :, shc_alpha, shc_beta, shc_gamma]
                    sigma_H [iEF] += 1j * pi * pre_fac * np.sum(imAB[:, :, np.newaxis] * temp2.transpose(1, 2, 0), axis = (0, 1)) / 4.0
                    sigma_AH[iEF] += pre_fac * np.sum(imAB[:, :, np.newaxis] * temp1.transpose(1, 2, 0), axis = (0, 1)) / 2.0
                else:
                    imAB = np.imag(np.einsum('nmac,mnb->nmabc',A,B))
                    sigma_H [iEF] += 1j * pi * pre_fac * kubo_sum_elements(imAB, temp2, data.num_wann) / 4.0
                    sigma_AH[iEF] += pre_fac * kubo_sum_elements(imAB, temp1, data.num_wann) / 2.0
                
            elif conductivity_type == 'shiftcurrent':
                delta_mn = np.copy(delta)
                dE2 = E[:,np.newaxis] - E[np.newaxis, :] # E_n(k) - E_m(k) [n, m]
                delta_arg = dE2[np.newaxis,:,:] - omega[:,np.newaxis,np.newaxis]
                if smr_type == 'Lorentzian':
                    delta = Lorentzian(delta_arg, eta)
                elif smr_type == 'Gaussian':
                    delta = Gaussian(delta_arg, eta, adpt_smr)
                else:
                    cprint("Invalid smearing type. Fallback to Lorentzian", 'red')
                    delta = Lorentzian(delta_arg, eta)
    
                delta_nm = np.copy(delta)
                cfac = delta_mn + delta_nm
                temp = -dfE[np.newaxis,:,:]*cfac
    
                # generalized derivative is fourth index of A, we put it into third index of Imn
                Imn = np.einsum('nmca,mnb->nmabc',A,B) + np.einsum('nmba,mnc->nmabc',A,B) 
                sigma_abc[iEF] +=  pre_fac * kubo_sum_elements(Imn, temp, data.num_wann) 




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

    elif conductivity_type == 'shiftcurrent':
        sigma_shift = sigma_abc
        return result.EnergyResult([Efermi,omega], sigma_shift, TRodd=False, Iodd=False, rank=rank)


def opt_SHCqiao(data, Efermi, omega=0, **parameters):
    return opt_conductivity(data, Efermi, omega, conductivity_type='SHC', SHC_type='qiao', **parameters )

def opt_SHCryoo(data, Efermi, omega=0,  **parameters):
    return opt_conductivity(data, Efermi, omega, conductivity_type='SHC', SHC_type='ryoo', **parameters )

def tildeD(data, Efermi, omega=0,  **parameters ):
    return opt_conductivity(data, Efermi, omega,   conductivity_type='tildeD', **parameters )

def opt_shiftcurrent(data, Efermi, omega=0, **parameters):
    return opt_conductivity(data, Efermi, omega, conductivity_type='shiftcurrent',  **parameters )


