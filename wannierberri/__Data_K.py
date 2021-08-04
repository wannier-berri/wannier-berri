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

   
class Data_K(System):
    def __init__(self,system,dK,grid,Kpoint=None,npar=0,fftlib='fftw',npar_k=1 ):
#        self.spinors=system.spinors
        self.iRvec=system.iRvec
        self.real_lattice=system.real_lattice
        self.recip_lattice=system.recip_lattice
        self.NKFFT=grid.FFT
        self.select_K=np.ones(self.NKFFT_tot,dtype=bool)
        self.findif=grid.findif
        self.Kpoint=Kpoint
        self.num_wann=system.num_wann
        self.frozen_max=system.frozen_max
        self.random_gauge=system.random_gauge
        self.degen_thresh=system.degen_thresh
        self.delta_fz=system.delta_fz
        self.nkptot = self.NKFFT[0]*self.NKFFT[1]*self.NKFFT[2]
        self.ksep = system.ksep
        self.use_wcc_phase=system.use_wcc_phase
        if self.use_wcc_phase:
            self.wannier_centres_reduced=system.wannier_centres_reduced
            self.wannier_centres_cart=system.wannier_centres_cart
        else:
            self.wannier_centres_reduced=np.zeros((self.num_wann,3))
            self.wannier_centres_cart=np.zeros((self.num_wann,3))
        ## TODO : create the plans externally, one per process 
#        print( "iRvec in data_K is :\n",self.iRvec)
        #self.fft_R_to_k=FFT_R_to_k(self.iRvec,self.NKFFT,self.num_wann,self.wannier_centres,numthreads=npar if npar>0 else 1,lib=fftlib,convention=system.convention)
        self.fft_R_to_k=FFT_R_to_k(self.iRvec,self.NKFFT,self.num_wann,self.wannier_centres_reduced,self.real_lattice,
                numthreads=npar if npar>0 else 1,lib=fftlib,use_wcc_phase=self.use_wcc_phase)
        self.Emin=system.Emin
        self.Emax=system.Emax
        self.poolmap=pool(npar_k)[0]

        
        if self.use_wcc_phase:
            w_centres_diff = np.array([[j-i for j in self.wannier_centres_reduced] for i in self.wannier_centres_reduced])
            expdK=np.exp(2j*np.pi*(system.iRvec[None,None,:,:] +w_centres_diff[:,:,None,:]).dot(dK))
        else:
            expdK=np.exp(2j*np.pi*system.iRvec.dot(dK))[None,None,:]
        self.HH_R=system.HH_R*expdK
        self.dK=dK
 
        
        for X in ['AA','BB','CC','SS','SA','SHA','SR','SH','SHR']:
            XR=X+'_R'
            hasXR='has_'+X+'_R'
            vars(self)[XR]=None
            vars(self)[hasXR]=False
            if XR in vars(system):
              if vars(system)[XR] is not  None:
                if X in ['SA','SHA','SR','SHR']:
                    vars(self)[XR]=vars(system)[XR]*expdK[:,:,:,None,None]
                else:
                    vars(self)[XR]=vars(system)[XR]*expdK[:,:,:,None]
                vars(self)[hasXR]=True


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
            XX_R=1j*XX_R.reshape( (XX_R.shape)+(1,) ) * self.cRvec_wcc.reshape((shape_cR[0],shape_cR[1],self.nRvec)+(1,)*len(XX_R.shape[3:])+(3,))
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

#    defining sets of degenerate states.  
    @lazy_property.LazyProperty
    def degen(self):
            A=[np.where(E[1:]-E[:-1]>self.degen_thresh)[0]+1 for E in self.E_K ]
            A=[ [0,]+list(a)+[len(E)] for a,E in zip(A,self.E_K) ]
            return [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:]) ]    for a,e in zip(A,self.E_K)]

    def degen_groups(self,ik,degen_thresh):
            A=[np.where(E[1:]-E[:-1]>degen_thresh)[0]+1 for E in self.E_K ]
            A=[ [0,]+list(a)+[len(E)] for a,E in zip(A,self.E_K) ]
            return [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:]) ]    for a,e in zip(A,self.E_K)]

    @lazy_property.LazyProperty
    def true_degen(self):
            A=[np.where(E[1:]-E[:-1]>self.degen_thresh)[0]+1 for E in self.E_K ]
            A=[ [0,]+list(a)+[len(E)] for a,E in zip(A,self.E_K) ]
            return [[(ib1,ib2) for ib1,ib2 in deg if ib2-ib1>1]  for deg in self.degen]

    @lazy_property.LazyProperty
    def E_K_degen(self):
        return [np.array([np.mean(E[ib1:ib2]) for ib1,ib2 in deg]) for deg,E in zip(self.degen,self.E_K)]

    @lazy_property.LazyProperty
    def degen_dic(self):
        return [np.array([np.mean(E[ib1:ib2]) for ib1,ib2 in deg]) for deg,E in zip(self.degen,self.E_K)]

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

    @lazy_property.LazyProperty
    def iter_op_ed(self):
        it=list(range(0,self.NKFFT_tot,self.ksep))+[self.NKFFT_tot]
        return list(zip(it,it[1:]))
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

    @lazy_property.LazyProperty
    def UU_K(self):
        print_my_name_start()
        self.E_K
        # the following is needed only for testing : 
        if self.random_gauge:
            from scipy.stats import unitary_group
            cnt=0
            s=0
            for ik,deg in enumerate(self.true_degen):
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
        return self._R_to_k_H(self.SA_R.copy(), hermitian=False)
    
    @lazy_property.LazyProperty
    def SHA_H(self):
        return self._R_to_k_H(self.SHA_R.copy(), hermitian=False)
#PRB QZYZ18, Qiao's way to calculate SHC

    def _shc_B_H_einsum_opt(self, C, A, B):
        # Optimized version of C += np.einsum('knlc,klma->knmac', A, B). Used in shc_B_H.
        nw = self.num_wann
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
        SH_H = self._R_to_k_H(self.SH_R.copy(), hermitian=False)
        shc_K_H = -1j*self._R_to_k_H(self.SR_R.copy(), hermitian=False)
        self._shc_B_H_einsum_opt(shc_K_H, self.S_H, self.D_H)
        shc_L_H = -1j*self._R_to_k_H(self.SHR_R.copy(), hermitian=False)
        self._shc_B_H_einsum_opt(shc_L_H, SH_H, self.D_H)
        return (self.delE_K[:,np.newaxis,:,:,np.newaxis]*self.S_H[:,:,:,np.newaxis,:] +
            self.E_K[:,np.newaxis,:,np.newaxis,np.newaxis]*shc_K_H[:,:,:,:,:] - shc_L_H)
#end SHC

    @lazy_property.LazyProperty
    def diag_w_centres(self):
        '''
        After rotate. U^+ \tau U
        Wannier Gauge.
        diagnal matrix of wannier centres delta_ij*tau_i (Cartesian)
        '''
        return np.sum(self.UU_K.conj()[:,:,:,None,None] *self.UU_K[:,:,None,:,None]*self.wannier_centres_cart[None,:,None,None,:] , axis=1)


#############
#  Abelian  #
#############
    @lazy_property.LazyProperty
    def vel_nonabelian(self):
         return [ [0.5*(S[ib1:ib2,ib1:ib2]+S[ib1:ib2,ib1:ib2].transpose((1,0,2)).conj()) for ib1,ib2 in deg] for S,deg in zip(self.V_H,self.degen)]


### TODO : check if it is really gaufge-covariant in case of isolated degeneracies
    @lazy_property.LazyProperty
    def mass_nonabelian(self):
        return [ [S[ib1:ib2,ib1:ib2]
                   +sum(np.einsum("mla,lnb->mnab",X,Y) 
                    for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.nb_selected)]  if ib2<self.nb_selected else []))
                     for X,Y in [
                     (-D[ib1:ib2,ibl1:ibl2,:],V[ibl1:ibl2,ib1:ib2,:]),
                     (+V[ib1:ib2,ibl1:ibl2,:],D[ibl1:ibl2,ib1:ib2,:]),
                              ])       for ib1,ib2 in deg]
                     for S,D,V,deg in zip( self.del2E_H,self.D_H,self.V_H,self.degen) ]


    @lazy_property.LazyProperty
    def mass_nonabelian_(self,ik,ib):
        return [ [S[ib1:ib2,ib1:ib2]
                   +sum(np.einsum("mla,lnb->mnab",X,Y) 
                    for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.nb_selected)]  if ib2<self.nb_selected else []))
                     for X,Y in [
                     (-D[ib1:ib2,ibl1:ibl2,:],V[ibl1:ibl2,ib1:ib2,:]),
                     (+V[ib1:ib2,ibl1:ibl2,:],D[ibl1:ibl2,ib1:ib2,:]),
                              ])       for ib1,ib2 in deg]
                     for S,D,V,deg in zip( self.del2E_H,self.D_H,self.V_H,self.degen) ]



    @lazy_property.LazyProperty
    def spin_nonabelian(self):
        return [ [S[ib1:ib2,ib1:ib2] for ib1,ib2 in deg] for S,deg in zip(self.S_H,self.degen)]


##  TODO: When it works correctly - think how to optimize it
    @lazy_property.LazyProperty
    def Berry_nonabelian(self):
        print_my_name_start()
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]
        res= [ [ O[ib1:ib2,ib1:ib2,:]-1j*sum(s*np.einsum("mla,lna->mna",A[ib1:ib2,ib1:ib2,b],A[ib1:ib2,ib1:ib2,c]) for s,b,c in sbc) 
               +sum(s*np.einsum("mla,lna->mna",X,Y) 
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.nb_selected)]  if ib2<self.nb_selected else []))
                     for s,b,c in sbc
                    for X,Y in [(-D[ib1:ib2,ibl1:ibl2,b],A[ibl1:ibl2,ib1:ib2,c]),(-A[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c]),
                                       (-1j*D[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c])]
                           )
                        for ib1,ib2 in deg]
                     for O,A,D,deg in zip( self.Omega_Hbar,self.A_Hbar,self.D_H,self.degen ) ] 
        print_my_name_end()
        return res

    def Berry_nonabelian_W(self,homega):
        if not hasattr(self,'Berry_nonabelian_W_calculated'):
            self.Berry_nonabelian_W_calculated= {}
        if homega not in self.Berry_nonabelian_W_calculated:
            self.Berry_nonabelian_W_calculated[homega]= self.calculate_Berry_nonabelian_W(homega)
        return self.Berry_nonabelian_W_calculated[homega]

### todo : check what is the correct "nonabelian" formulation
    def calculate_Berry_nonabelian_W(self,homega):
        print_my_name_start()
        wnl2=(self.E_K[:,:,None]-self.E_K[:,None,:])**2
        A_H_W=(wnl2/(wnl2-homega**2))[:,:,:,None]*self.A_H
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]
        res= [ [  0.5j*sum(s*np.einsum("mla,lna->mna",X,Y) 
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
                     for s,b,c in sbc
                    for X,Y in [(Aw[ib1:ib2,ibl1:ibl2,b],A[ibl1:ibl2,ib1:ib2,c]),(A[ib1:ib2,ibl1:ibl2,b],Aw[ibl1:ibl2,ib1:ib2,c])]
                           )
                        for ib1,ib2 in deg]
                     for Aw,A,deg in zip( A_H_W,self.A_H,self.degen ) ] 
        print_my_name_end()
        return res

    @lazy_property.LazyProperty
    def Berry_nonabelian_ext1(self):
        print_my_name_start()
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]
        res= [ [ O[ib1:ib2,ib1:ib2,:]-1j*sum(s*np.einsum("mla,lna->mna",A[ib1:ib2,ib1:ib2,b],A[ib1:ib2,ib1:ib2,c]) for s,b,c in sbc) 
                        for ib1,ib2 in deg]
                     for O,A,D,deg in zip( self.Omega_Hbar,self.A_Hbar,self.D_H,self.degen ) ] 
        print_my_name_end()
        return res

    @lazy_property.LazyProperty
    def Berry_nonabelian_ext2(self):
        print_my_name_start()
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]
        res= [ [ 
               +sum(s*np.einsum("mla,lna->mna",X,Y) 
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.nb_selected)]  if ib2<self.nb_selected else []))
                     for s,b,c in sbc
                    for X,Y in [(-D[ib1:ib2,ibl1:ibl2,b],A[ibl1:ibl2,ib1:ib2,c]),(-A[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c]),
                                       ]
                           )
                        for ib1,ib2 in deg]
                     for O,A,D,deg in zip( self.Omega_Hbar,self.A_Hbar,self.D_H,self.degen ) ] 
        print_my_name_end()
        return res

    @lazy_property.LazyProperty
    def Berry_nonabelian_D(self):
        print_my_name_start()
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]
        res= [ [ -1j*sum(s*np.einsum("mla,lna->mna",A[ib1:ib2,ib1:ib2,b],A[ib1:ib2,ib1:ib2,c]) for s,b,c in sbc) 
               +sum(s*np.einsum("mla,lna->mna",X,Y) 
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.nb_selected)]  if ib2<self.nb_selected else []))
                     for s,b,c in sbc
                    for X,Y in [ (-1j*D[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c]) , ]
                           )
                        for ib1,ib2 in deg]
                     for A,D,deg in zip( self.A_Hbar,self.D_H,self.degen ) ] 
        print_my_name_end()
        return res

##  TODO: When it works correctly - think how to optimize it
    @lazy_property.LazyProperty
    def Morb_nonabelian(self):
        print_my_name_start()
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]
        Morb=[ [ M[ib1:ib2,ib1:ib2,:]-e*O[ib1:ib2,ib1:ib2,:]
               +sum(s*np.einsum("mla,lna->mna",X,Y) 
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.nb_selected)]  if ib2<self.nb_selected else []))
                     for s,b,c in sbc
                    for X,Y in [
                    (-D[ib1:ib2,ibl1:ibl2,b],B[ibl1:ibl2,ib1:ib2,c]),
                    (-B.transpose((1,0,2)).conj()[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c]),
                         (-1j*V[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c]),
                              ]
                           )
                        for (ib1,ib2),e in zip(deg,E)]
                     for M,O,A,B,D,V,deg,E,EK in zip( self.Morb_Hbar,self.Omega_Hbar,self.A_Hbar,self.B_Hbarbar,self.D_H,self.V_H,self.degen,self.E_K_degen,self.E_K) ]
        print_my_name_end()
        return Morb


################################################################
#def merge_dataIO(data_list):
#    return { key:np.stack([data[key] for key in data],axis=0) for key in  
#                  set([key for data in data_list for key in data.keys()])  }

