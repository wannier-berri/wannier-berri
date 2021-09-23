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


## TODO : maybe to make some lazy_property's not so lazy to save some memory
import numpy as np
import lazy_property
from .__parallel import pool
from collections import defaultdict
from .__system import System
import time
from .__utility import  print_my_name_start,print_my_name_end, FFT_R_to_k, alpha_A,beta_A
import gc
import os
from .__tetrahedron import TetraWeights,get_bands_in_range,get_bands_below_range

def _rotate_matrix(X):
    return X[1].T.conj().dot(X[0]).dot(X[1])

   
class Data_K():
    default_parameters =  {
                    'frozen_max': -np.Inf,
                    'delta_fz':0.1,
                    'Emin': -np.Inf ,
                    'Emax': np.Inf ,
                    'use_wcc_phase':False,
                    'fftlib' : 'fftw',
                    'npar_k' : 1 ,
                    'random_gauge':False,
                    'degen_thresh_random_gauge':1e-4 ,
                       }

    __doc__ = """
    class to store data of the FFT grid. Is destroyed after  everything is evaluated for the FFT grid

    Parameters
    -----------
    frozen_max : float
        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary. 
        If not specified, attempts to read this value from system. Othewise set to  ``{frozen_max}``
    random_gauge : bool
        applies random unitary rotations to degenerate states. Needed only for testing, to make sure that gauge covariance is preserved. Default: ``{random_gauge}``
    degen_thresh_random_gauge : float
        threshold to consider bands as degenerate for random_gauge Default: ``{degen_thresh_random_gauge}``
    delta_fz:float
        size of smearing for B matrix with frozen window, from frozen_max-delta_fz to frozen_max. Default: ``{delta_fz}``
    """ .format(**default_parameters)



    def __init__(self,system,dK,grid,Kpoint=None,**parameters):
#        self.spinors=system.spinors
        self.system=system
        self.set_parameters(**parameters)
        self.NKFFT=grid.FFT
        self.select_K=np.ones(self.NKFFT_tot,dtype=bool)
        self.findif=grid.findif
        self.cell_volume=self.system.cell_volume
        self.num_wann=self.system.num_wann
        self.Kpoint=Kpoint
        self.nkptot = self.NKFFT[0]*self.NKFFT[1]*self.NKFFT[2]

#        self.use_wcc_phase=system.use_wcc_phase
        if self.use_wcc_phase:
            self.wannier_centres_reduced=system.wannier_centres_reduced
            self.wannier_centres_cart=system.wannier_centres_cart
        else:
            self.wannier_centres_reduced=np.zeros((self.num_wann,3))
            self.wannier_centres_cart=np.zeros((self.num_wann,3))


        self.fft_R_to_k=FFT_R_to_k(self.system.iRvec,self.NKFFT,self.system.num_wann,
               self.system.wannier_centres_reduced,self.system.real_lattice,
                numthreads=self.npar_k if self.npar_k>0 else 1,lib=self.fftlib,use_wcc_phase=self.use_wcc_phase)
        self.poolmap=pool(self.npar_k)[0]


        
        if self.use_wcc_phase:
            self.cRvec_wcc=self.system.cRvec_wcc
            w_centres_diff = np.array([[j-i for j in self.system.wannier_centres_reduced] for i in self.system.wannier_centres_reduced])
            self.expdK=np.exp(2j*np.pi*(self.system.iRvec[None,None,:,:] +w_centres_diff[:,:,None,:]).dot(dK))
        else:
            self.cRvec_wcc=self.system.cRvec[None,None,:,:]
            self.expdK=np.exp(2j*np.pi*self.system.iRvec.dot(dK))[None,None,:]
        self.HH_R=system.HH_R*self.expdK
        self.dK=dK

    def set_parameters(self,**parameters):
        for param in self.default_parameters:
            if param in parameters:
                vars(self)[param]=parameters[param]
            else: 
                vars(self)[param]=self.default_parameters[param]
        if 'frozen_max' not in parameters:
            try : 
                self.frozen_max= self.system.frozen_max
            except:
                pass 



###########################################
###   Now the **_R objects are evaluated only on demand 
### - as Lazy_property (if used more than once) 
###   as property   - iif used only once
###   let's write them explicitly, for better code readability
###########################

    @lazy_property.LazyProperty
    def AA_R(self):
        return self.system.AA_R*self.expdK[:,:,:,None]

    @lazy_property.LazyProperty
    def BB_R(self):
        return self.system.BB_R*self.expdK[:,:,:,None]

    @lazy_property.LazyProperty
    def CC_R(self):
        return self.system.CC_R*self.expdK[:,:,:,None]

    @lazy_property.LazyProperty
    def SS_R(self):
        return self.system.SS_R*self.expdK[:,:,:,None]

    @property
    def SH_R(self):
        return self.system.SH_R*self.expdK[:,:,:,None]

    @property
    def SR_R(self):
        return self.system.SR_R*self.expdK[:,:,:,None,None]

    @property
    def SA_R(self):
        return self.system.SA_R*self.expdK[:,:,:,None,None]

    @property
    def SHA_R(self):
        return self.system.SHA_R*self.expdK[:,:,:,None,None]

    @property
    def SHR_R(self):
        return self.system.SHR_R*self.expdK[:,:,:,None,None]

###############################################################

###########
#  TOOLS  #
###########

    def _rotate(self,mat):
        print_my_name_start()
        assert mat.ndim>2
        if mat.ndim==3:
            return  np.array(self.poolmap( _rotate_matrix , zip(mat,self.UU_K)))
        else:
            for i in range(mat.shape[-1]):
                mat[...,i]=self._rotate(mat[...,i])
            return mat

    @lazy_property.LazyProperty
    def diag_w_centres(self):
        '''
        After rotate. U^+ \tau U
        diagnal matrix of wannier centres delta_ij*tau_i (Cartesian)
        '''
        return np.sum(self.UU_K.conj()[:,:,:,None,None] *self.UU_K[:,:,None,:,None]*self.system.wannier_centres_cart[None,:,None,None,:] , axis=1)
        

    def _R_to_k_H(self,XX_R,der=0,hermitian=True,asym_before=False,asym_after=False,flag=None):
        """ converts from real-space matrix elements in Wannier gauge to 
            k-space quantities in k-space. 
            der [=0] - defines the order of comma-derivative 
            hermitian [=True] - consoder the matrix hermitian
            asym_before = True -  takes the antisymmetrc part over the first two cartesian indices before differentiation
            asym_after = True  - asymmetrize after  differentiation
            flag: is a flag indicates if we need additional terms in self.fft_R_to_k under use_wcc_phase. 
                'None' means no adiditional terms.
                'AA' means, under use_wcc_phase, FFT of AA_R have an additional term -tau_i.
                'BB' means, under use_wcc_phase, FFT of AA_R have an additional term -tau_i*HH_k.
                'CC' means, under use_wcc_phase, FFT of AA_R have an additional term epsilon_{abc} ((tau_j^a - tau_i^a)*B_Hbar_fz^b + 1j*V_H^a*tau_j^b )
            WARNING: the input matrix is destroyed, use np.copy to preserve it"""
        
        def asymmetrize(X,asym):
            """auxilary function"""
            if asym  :
                assert len(X.shape)>=5 , "cannot antisymmetrize less then 2 indices"
                return X[:,:,:,alpha_A,beta_A]-X[:,:,:,beta_A,alpha_A]
            else:
                return X
        XX_R=asymmetrize(XX_R, asym_before)
        for i in range(der):
            shape_cR = np.shape(self.cRvec_wcc)
            XX_R=1j*XX_R.reshape( (XX_R.shape)+(1,) ) * self.cRvec_wcc.reshape((shape_cR[0],shape_cR[1],self.system.nRvec)+(1,)*len(XX_R.shape[3:])+(3,))
        XX_R=asymmetrize(XX_R, asym_after)
        
        add_term = 0.0 # additional term under use_wcc_phase=True
        if self.use_wcc_phase:
            if flag=='AA':
                add_term = - self.diag_w_centres
            elif flag=='BB':
                add_term = - self.diag_w_centres*self.HH_K[:,:,:,None]
            elif flag=='CC':
                add_term = np.sum(
                        self.B_Hbar_fz[:,:,:,None,beta_A]*self.diag_w_centres[:,None,:,:,alpha_A] 
                        - self.B_Hbar_fz[:,None,:,:,beta_A]*self.diag_w_centres[:,:,:,None,alpha_A] 
                        + 1j*self.V_H[:,:,:,None,alpha_A]*self.diag_w_centres[:,None,:,:,beta_A]
                        ,axis=2) - np.sum(
                        self.B_Hbar_fz[:,:,:,None,alpha_A]*self.diag_w_centres[:,None,:,:,beta_A] 
                        - self.B_Hbar_fz[:,None,:,:,alpha_A]*self.diag_w_centres[:,:,:,None,beta_A] 
                        + 1j*self.V_H[:,:,:,None,beta_A]*self.diag_w_centres[:,None,:,:,alpha_A]
                        ,axis=2)
        res = self._rotate((self.fft_R_to_k( XX_R,hermitian=hermitian))[self.select_K]  )
        res = res + add_term
        return res

#####################
#  Basis variables  #
#####################
    @lazy_property.LazyProperty
    def nbands(self):
        return self.HH_R.shape[0]

    @lazy_property.LazyProperty
    def kpoints_all(self):
        dkx,dky,dkz=1./self.NKFFT
        return np.array([self.dK+np.array([ix*dkx,iy*dky,iz*dkz]) 
          for ix in range(self.NKFFT[0])
              for iy in range(self.NKFFT[1])
                  for  iz in range(self.NKFFT[2])])%1

    @lazy_property.LazyProperty
    def NKFFT_tot(self):
        return np.prod(self.NKFFT)

    @lazy_property.LazyProperty
    def tetraWeights(self):
        return TetraWeights(self.E_K,self.E_K_corners)

    def get_bands_in_range(self,emin,emax,op=0,ed=None):
        if ed is None: ed=self.NKFFT_tot
        select = [ np.where((self.E_K[ik]>=emin)*(self.E_K[ik]<=emax))[0] for ik in range(op,ed) ]
        return  [ {ib:self.E_K[ik+op,ib]  for ib in sel } for ik,sel in enumerate(select) ]

    def get_bands_below_range(self,emin,emax,op=0,ed=None):
        if ed is None: ed=self.NKFFT_tot
        res=[np.where((self.E_K[ik]<emin))[0] for ik in range(op,ed)]
        return [{a.max():self.E_K[ik+op,a.max()]} if len(a)>0 else [] for ik,a in enumerate(res)]

    def get_bands_in_range_sea(self,emin,emax,op=0,ed=None):
        if ed is None: ed=self.NKFFT_tot
        res=self.get_bands_in_range(emin,emax,op,ed)
        for ik in range(op,ed):
           add=np.where((self.E_K[ik]<emin))[0]
#           print ("add : ",add," / ",self.E_K[ik])
           if len(add)>0:
               res[ik-op][add.max()]=self.E_K[ik,add.max()]
        return res


    def get_bands_in_range_groups(self,emin,emax,op=0,ed=None,degen_thresh=-1,sea=False):
#        get_bands_in_range(emin,emax,Eband,degen_thresh=-1,Ebandmin=None,Ebandmax=None)
        if ed is None: ed=self.NKFFT_tot
        res=[]
        for ik in range(op,ed):
            bands_in_range=get_bands_in_range(emin,emax,self.E_K[ik],degen_thresh=degen_thresh)
            weights= { (ib1,ib2):self.E_K[ik,ib1:ib2].mean() 
                          for ib1,ib2 in bands_in_range  
                     }
            if sea :
                bandmax=get_bands_below_range(emin,self.E_K[ik])
#                print ("bandmax=",bandmax)
                if len(bands_in_range)>0 :
                    bandmax=min(bandmax, bands_in_range[0][0])
#                print ("now : bandmax=",bandmax ,self.E_K[ik][bandmax] )
                if bandmax>0:
                    weights[(0,bandmax)]=-np.Inf
            res.append( weights )
        return res

###################################################
#  Basis variables and their standard derivatives #
###################################################
    @lazy_property.LazyProperty
    def E_K(self):
        print_my_name_start()
        EUU=self.poolmap(np.linalg.eigh,self.HH_K)
        E_K=np.array([euu[0] for euu in EUU])
        select=(E_K>self.Emin)*(E_K<self.Emax)
        self.select_K=np.all(select,axis=1)
        self.select_B=np.all(select,axis=0)
        self.nk_selected=self.select_K.sum()
        self.nb_selected=self.select_B.sum()
        self._UU=np.array([euu[1] for euu in EUU])[self.select_K,:][:,self.select_B]
        print_my_name_end()
        return E_K[self.select_K,:][:,self.select_B]
    
    # evaluate the energies in the corners of the parallelepiped, in order to use tetrahedron method
    @lazy_property.LazyProperty
    def E_K_corners(self):
        dK2=self.Kpoint.dK_fullBZ/2
        expdK=np.exp(2j*np.pi*self.iRvec*dK2[None,:])
        expdK=np.array([1./expdK,expdK])
        Ecorners=np.zeros((self.nk_selected,2,2,2,self.nb_selected),dtype=float)
        for ix in 0,1:
            for iy in 0,1:
               for iz in 0,1:
                   _expdK=expdK[ix,:,0]*expdK[iy,:,1]*expdK[iz,:,2]
                   _HH_R=self.HH_R[:,:,:]*_expdK[None,None,:]
                   _HH_K=self.fft_R_to_k( _HH_R, hermitian=True)
                   E=np.array(self.poolmap(np.linalg.eigvalsh,_HH_K))
                   Ecorners[:,ix,iy,iz,:]=E[self.select_K,:][:,self.select_B]
        print_my_name_end()
        return Ecorners

    @property
    def HH_K(self):
        return self.fft_R_to_k( self.HH_R, hermitian=True) 

    @lazy_property.LazyProperty
    def delE_K(self):
        print_my_name_start()
        delE_K = np.einsum("klla->kla",self.V_H)
        check=np.abs(delE_K).imag.max()
        if check>1e-10: raiseruntimeError ( "The band derivatives have considerable imaginary part: {0}".format(check) )
        return delE_K.real

    @lazy_property.LazyProperty
    def del2E_H(self):
        return self._R_to_k_H( self.HH_R, der=2 )

    @lazy_property.LazyProperty
    def del3E_H(self):
        return self._R_to_k_H( self.HH_R, der=3 )

    @property
    def del2E_H_diag(self):
        return np.einsum("knnab->knab",self.del2E_H).real

    @lazy_property.LazyProperty
    def dEig_inv(self):
        dEig_threshold=1.e-7
        dEig=self.E_K[:,:,None]-self.E_K[:,None,:]
        select=abs(dEig)<dEig_threshold
        dEig[select]=dEig_threshold
        dEig=1./dEig
        dEig[select]=0.
        return dEig

#    defining sets of degenerate states - needed only for testing with random_gauge
    @lazy_property.LazyProperty
    def degen(self):
            A=[np.where(E[1:]-E[:-1]>self.degen_thresh_random_gauge)[0]+1 for E in self.E_K ]
            A=[ [0,]+list(a)+[len(E)] for a,E in zip(A,self.E_K) ]
            return [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:]) if ib2-ib1>1 ]    for a in A ]

    @lazy_property.LazyProperty
    def UU_K(self):
        print_my_name_start()
        self.E_K
        # the following is needed only for testing : 
        if self.random_gauge:
            from scipy.stats import unitary_group
            cnt=0
            s=0
            for ik,deg in enumerate(self.true):
                for ib1,ib2 in deg:
                    self._UU[ik,:,ib1:ib2]=self._UU[ik,:,ib1:ib2].dot( unitary_group.rvs(ib2-ib1) )
                    cnt+=1
                    s+=ib2-ib1
#            print ("applied random rotations {} times, average degeneracy is {}-fold".format(cnt,s/max(cnt,1)))
        print_my_name_end()
        return self._UU

    @lazy_property.LazyProperty
    def V_H(self):
        self.E_K
        return self._R_to_k_H( self.HH_R.copy(), der=1)

    @lazy_property.LazyProperty
    def D_H(self):
        return -self.V_H*self.dEig_inv[:, :,:,None]

    @lazy_property.LazyProperty
    def A_Hbar(self):
        return self._R_to_k_H(self.AA_R.copy(),flag='AA')

    @lazy_property.LazyProperty
    def A_H(self):
        '''Generalized Berry connection matrix, A^(H) as defined in eqn. (25) of 10.1103/PhysRevB.74.195118.'''
        return self.A_Hbar + 1j*self.D_H

    @lazy_property.LazyProperty
    def A_Hbar_der(self):
        return  self._R_to_k_H(self.AA_R.copy(), der=1) 

    @lazy_property.LazyProperty
    def B_Hbar(self):
        print_my_name_start()
        _BB_K=self._R_to_k_H( self.BB_R.copy(),hermitian=False,flag='BB')
        select=(self.E_K<=self.frozen_max)
        _BB_K[select]=self.E_K[select][:,None,None]*self.A_Hbar[select]
        return _BB_K
    
    @lazy_property.LazyProperty
    def B_Hbar_der(self):
        _BB_K=self._R_to_k_H( self.BB_R.copy(), der=1,hermitian=False)
        return _BB_K

    @lazy_property.LazyProperty
    def B_Hbarbar(self):
        print_my_name_start()
        B= self.B_Hbar-self.A_Hbar[:,:,:,:]*self.E_K[:,None,:,None]
        print_my_name_end()
        return B

    @lazy_property.LazyProperty
    def Omega_Hbar(self):
        print_my_name_start()
        return  -self._R_to_k_H( self.AA_R, der=1, asym_after=True)

    @lazy_property.LazyProperty
    def Omega_bar_der(self):
        print_my_name_start()
        _OOmega_K =  self.fft_R_to_k( (
                        self.AA_R[:,:,:,alpha_A]*self.cRvec_wcc[:,:,:,beta_A ] -     
                        self.AA_R[:,:,:,beta_A ]*self.cRvec_wcc[:,:,:,alpha_A])[:,:,:,:,None]*self.cRvec_wcc[:,:,:,None,:]   , hermitian=True)
        return self._rotate(_OOmega_K)

    @lazy_property.LazyProperty
    def Morb_Hbar(self):
        return self._R_to_k_H( self.CC_R.copy(),flag='CC')

    @lazy_property.LazyProperty
    def Morb_Hbar_der(self):
        return self._R_to_k_H( self.CC_R, der=1 )

    @lazy_property.LazyProperty
    def S_H(self):
        return  self._R_to_k_H( self.SS_R.copy() )

    @lazy_property.LazyProperty
    def delS_H(self):
        """d_b S_a """
        return self._R_to_k_H( self.SS_R.copy(), der=1,hermitian=True )

#PRB RPS19, Ryoo's way to calculate SHC
    @lazy_property.LazyProperty
    def SA_H(self):
        return self._R_to_k_H(self.SA_R, hermitian=False)
    
    @lazy_property.LazyProperty
    def SHA_H(self):
        return self._R_to_k_H(self.SHA_R, hermitian=False)
#PRB QZYZ18, Qiao's way to calculate SHC

    def _shc_B_H_einsum_opt(self, C, A, B):
        # Optimized version of C += np.einsum('knlc,klma->knmac', A, B). Used in shc_B_H.
        nw = self.system.num_wann
        for ik in range(self.nkptot):
            # Performing C[ik] += np.einsum('nlc,lma->nmac', A[ik], B[ik])
            tmp_a = np.swapaxes(A[ik], 1, 2) # nlc -> ncl
            tmp_a = np.reshape(tmp_a, (nw*3, nw)) # ncl -> (nc)l
            tmp_b = np.reshape(B[ik], (nw, nw*3)) # lma -> l(ma)
            tmp_c = tmp_a @ tmp_b # (nc)l, l(ma) -> (nc)(ma)
            tmp_c = np.reshape(tmp_c, (nw, 3, nw, 3)) # (nc)(ma) -> ncma
            C[ik] += np.transpose(tmp_c, (0, 2, 3, 1)) # ncma -> nmac

    @lazy_property.LazyProperty
    def shc_B_H(self):
        SH_H = self._R_to_k_H(self.SH_R, hermitian=False)
        shc_K_H = -1j*self._R_to_k_H(self.SR_R, hermitian=False)
        self._shc_B_H_einsum_opt(shc_K_H, self.S_H, self.D_H)
        shc_L_H = -1j*self._R_to_k_H(self.SHR_R, hermitian=False)
        self._shc_B_H_einsum_opt(shc_L_H, SH_H, self.D_H)
        return (self.delE_K[:,np.newaxis,:,:,np.newaxis]*self.S_H[:,:,:,np.newaxis,:] +
            self.E_K[:,np.newaxis,:,np.newaxis,np.newaxis]*shc_K_H[:,:,:,:,:] - shc_L_H)
#end SHC


