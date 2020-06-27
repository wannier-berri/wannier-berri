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
#import billiard as multiprocessing 
import  multiprocessing 
from .__system import System
from .__utility import  print_my_name_start,print_my_name_end,einsumk, FFT_R_to_k, alpha_A,beta_A

def _rotate_matrix(X):
    return X[1].T.conj().dot(X[0]).dot(X[1])

   
class Data_K(System):
    def __init__(self,system,dK=None,NKFFT=None,Kpoint=None,npar=0,fftlib='fftw'):
#        self.spinors=system.spinors
        self.iRvec=system.iRvec
        self.real_lattice=system.real_lattice
        self.recip_lattice=system.recip_lattice
        self.NKFFT=system.NKFFT if NKFFT is None else NKFFT
        self.Kpoint=Kpoint
        self.num_wann=system.num_wann
        self.frozen_max=system.frozen_max
        self.random_gauge=system.random_gauge
        self.degen_thresh=system.degen_thresh
        ## TODO : create the plans externally, one per process 
        self.fft_R_to_k=FFT_R_to_k(system.iRvec,NKFFT,self.num_wann,numthreads=npar if npar>0 else 1,lib=fftlib)

        try:
            self.poolmap=multiprocessing.Pool(npar).map
#            print ('created a pool of {} workers'.format(npar))
        except Exception as err:
#            print ('failed to create a pool of {} workers : {}'.format(npar,err))
            self.poolmap=lambda fun,lst : [fun(x) for x in lst]
        if dK is not None:
            expdK=np.exp(2j*np.pi*system.iRvec.dot(dK))
            self.dK=dK
        else:
            expdK=np.ones(self.nRvec)
            self.dK=np.zeros(3)
 
        self.HH_R=system.HH_R[:,:,:]*expdK[None,None,:]
        
        for X in ['AA','BB','CC','SS']:
            XR=X+'_R'
            hasXR='has_'+X+'_R'
            vars(self)[XR]=None
            vars(self)[hasXR]=False
            if XR in vars(system):
              if vars(system)[XR] is not  None:
                vars(self)[XR]=vars(system)[XR]*expdK[None,None,:,None]
                vars(self)[hasXR]=True


    def _rotate(self,mat):
        print_my_name_start()
#        return  np.einsum('kml,kmn,knp->klp',self.UU_K.conj(),mat,self.UU_K)
        assert mat.ndim>2
        if mat.ndim==3:
            return  np.array(self.poolmap( _rotate_matrix , zip(mat,self.UU_K)))
        else:
            for i in range(mat.shape[-1]):
                mat[...,i]=self._rotate(mat[...,i])
            return mat

    def _R_to_k_H(self,XX_R,der=0,hermitian=True,asym_before=False,asym_after=False):
        """ converts from real-space matrix elements in Wannier gauge to 
            k-space quantities in k-space. 
            der [=0] - defines the order of comma-derivative 
            hermitian [=True] - consoder the matrix hermitian
            asym_before = True -  takes the antisymmetrc part over the first two cartesian indices before differentiation
            asym_after = True  - asymmetrize after  differentiation
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
            XX_R=1j*XX_R.reshape( (XX_R.shape)+(1,) )*self.cRvec.reshape((1,1,self.nRvec)+(1,)*len(XX_R.shape[3:])+(3,))
        XX_R=asymmetrize(XX_R, asym_after)
        return self._rotate(self.fft_R_to_k( XX_R,hermitian=hermitian)  )


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


    @lazy_property.LazyProperty
    def true_degen(self):
            A=[np.where(E[1:]-E[:-1]>self.degen_thresh)[0]+1 for E in self.E_K ]
            A=[ [0,]+list(a)+[len(E)] for a,E in zip(A,self.E_K) ]
            return [[(ib1,ib2) for ib1,ib2 in deg if ib2-ib1>1]  for deg in self.degen]


    @lazy_property.LazyProperty
    def E_K_degen(self):
        return [np.array([np.mean(E[ib1:ib2]) for ib1,ib2 in deg]) for deg,E in zip(self.degen,self.E_K)]

    @lazy_property.LazyProperty
    def vel_nonabelian(self):
        return [ [0.5*(S[ib1:ib2,ib1:ib2]+S[ib1:ib2,ib1:ib2].transpose((1,0,2)).conj()) for ib1,ib2 in deg] for S,deg in zip(self.V_H,self.degen)]


### TODO : check if it is really gaufge-covariant in case of isolated degeneracies
    @lazy_property.LazyProperty
    def mass_nonabelian(self):
        return [ [S[ib1:ib2,ib1:ib2]
                   +sum(np.einsum("mla,lnb->mnab",X,Y) 
                    for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
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
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
                     for s,b,c in sbc
                    for X,Y in [(-D[ib1:ib2,ibl1:ibl2,b],A[ibl1:ibl2,ib1:ib2,c]),(-A[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c]),
                                       (-1j*D[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c])]
                           )
                        for ib1,ib2 in deg]
                     for O,A,D,deg in zip( self.Omega_Hbar,self.A_Hbar,self.D_H,self.degen ) ] 
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
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
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
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
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
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
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

        
    @property
    def HH_K(self):
        return self.fft_R_to_k( self.HH_R, hermitian=True) 

    @lazy_property.LazyProperty
    def E_K(self):
        print_my_name_start()
        EUU=self.poolmap(np.linalg.eigh,self.HH_K)
        E_K=np.array([euu[0] for euu in EUU])
        self._UU=np.array([euu[1] for euu in EUU])
        print_my_name_end()
        return E_K

    @lazy_property.LazyProperty
#    @property
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
    def delE_K(self):
        print_my_name_start()
        delE_K = np.einsum("klla->kla",self.V_H)
        check=np.abs(delE_K).imag.max()
        if check>1e-10: raiseruntimeError ( "The band derivatives have considerable imaginary part: {0}".format(check) )
        return delE_K.real


    @lazy_property.LazyProperty
    def del2E_H(self):
        return self._R_to_k_H( self.HH_R, der=2 )

    @property
    def del2E_H_diag(self):
        return np.einsum("knnab->knab",self.del2E_H).real

    @lazy_property.LazyProperty
    def dEig_inv(self):
        dEig_threshold=1e-14
        dEig=self.E_K[:,:,None]-self.E_K[:,None,:]
        select=abs(dEig)<dEig_threshold
        dEig[select]=dEig_threshold
        dEig=1./dEig
        dEig[select]=0.
        return dEig

    @lazy_property.LazyProperty
    def D_H(self):
            return -self.V_H*self.dEig_inv[:, :,:,None]

    @lazy_property.LazyProperty
    def V_H(self):
        self.E_K
        return self._R_to_k_H( self.HH_R, der=1 )

    @lazy_property.LazyProperty
    def Morb_Hbar(self):
        return self._R_to_k_H( self.CC_R.copy() )

    @lazy_property.LazyProperty
    def Morb_Hbar_diag(self):
        return np.einsum("klla->kla",self.Morb_Hbar).real

    @lazy_property.LazyProperty
    def Morb_Hbar_der(self):
        return self._R_to_k_H( self.CC_R, der=1 )

    @lazy_property.LazyProperty
    def Morb_Hbar_der_diag(self):
        return np.einsum("kllad->klad",self.Morb_Hbar_der).real



    
   # @lazy_property.LazyProperty
    @property
    def gdD(self):
         # evaluates tildeD  as three terms : gdD1[k,n,l,a,b] , gdD1[k,n,n',l,a,b] ,  gdD2[k,n,l',l,a,b] 
         # which after summing over l',n' will give the generalized derivative

        dDln=-self.del2E_H*self.dEig_inv[:,:,:,None,None]
        dDlln= self.V_H[:, :,:,None, :,None]*self.D_H[:, None,:,:,None, :]
        dDlnn= self.D_H[:, :,:,None, :,None]*self.V_H[:, None,:,:,None, :]
                                 
        dDlln=-(dDlln+dDlln.transpose(0,1,2,3,5,4))*self.dEig_inv[:,:,None,:  ,None,None]
        dDlnn=(dDlnn+dDlnn.transpose(0,1,2,3,5,4))*self.dEig_inv[:,:,None,:  ,None,None]
                                                                
        return dDln,dDlln,dDlnn

   # @lazy_property.LazyProperty
    @property
    def gdAbar(self):
        dAln= self.A_Hbar_der
        dAlln= self.A_Hbar[:,:,:,None,:,None]*self.D_H[:,None,:,:,None,:]
        dAlnn= -self.D_H[:,:,:,None,None,:]*self.A_Hbar[:,None,:,:,:,None]

        return dAln,dAlln,dAlnn

   # @lazy_property.LazyProperty
    @property
    def gdBbar(self):
        dBln= self.B_Hbar_der
        dBlln= self.B_Hbar[:,:,:,None,:,None]*self.D_H[:,None,:,:,None,:]
        dBlnn= -self.D_H[:,:,:,None,None,:]*self.B_Hbar[:,None,:,:,:,None]

        return dBln,dBlln,dBlnn

  #  @lazy_property.LazyProperty
    @property
    def gdBbarplus(self):
        Aln,Alln,Alnn = self.gdAbar 
        Bln,Blln,Blnn = self.gdBbar
        A = self.A_Hbar
        V = self.V_H
        dBPln=  Bln + Aln*self.E_K[:,None,:,None,None] 
        dBPlln= Blln + Alln*self.E_K[:,None,None,:,None,None] 
        dBPlnn= Blnn + Alnn*self.E_K[:,None,None,:,None,None] + A[:,:,:,None,:,None]*V[:,None,:,:,None,:]

        return dBPln,dBPlln,dBPlnn

    @lazy_property.LazyProperty
    def gdOmegabar(self):
        dOn= self.Omega_bar_der_rediag.real
        dOln= (self.Omega_Hbar[:,:,:,:,None].transpose(0,2,1,3,4)*self.D_H[:,:,:,None,:]-self.D_H[:,:,:,None,:].transpose(0,2,1,3,4)*self.Omega_Hbar[:,:,:,:,None]).real

        return dOn,dOln

    @lazy_property.LazyProperty
    def gdHbar(self):
        Hbar = self.Morb_Hbar
        dHn= self.Morb_Hbar_der_diag.real
        dHln= (Hbar[:,:,:,:,None].transpose(0,2,1,3,4)*self.D_H[:,:,:,None,:]-self.D_H[:,:,:,None,:].transpose(0,2,1,3,4)*Hbar[:,:,:,:,None]).real

        return dHn, dHln

   # @lazy_property.LazyProperty
    @property
    def B_Hbarplus_dagger(self):
        B = self.B_Hbar
        A = self.A_Hbar
        Bplus= (B+A*self.E_K[:,None,:,None]).conj()
        return Bplus

    @lazy_property.LazyProperty
    def derOmegaTr(self):
        b=alpha_A
        c=beta_A
        N=None
        Anl = self.A_Hbar.transpose(0,2,1,3)
        Dnl = self.D_H.transpose(0,2,1,3)
        dDln, dDlln,dDlnn= self.gdD
        dAln, dAlln,dAlnn= self.gdAbar
        dOn,dOln = self.gdOmegabar

        o = dOn
        uo = dOln - 2*((Anl[:,:,:,b,N]*dDln[:,:,:,c,:] + Dnl[:,:,:,b,N]*dAln[:,:,:,c,:]) - (Anl[:,:,:,c,N]*dDln[:,:,:,b,:] + Dnl[:,:,:,c,N]*dAln[:,:,:,b,:]) ).real + 2*( Dnl[:,:,:,b,N]*dDln[:,:,:,c,:]  -  Dnl[:,:,:,c,N]*dDln[:,:,:,b,:]  ).imag

        uuo = -2*((Anl[:,:,N,:,b,N]*dDlln[:,:,:,:,c,:] + Dnl[:,:,N,:,b,N]*dAlln[:,:,:,:,c,:]) - (Anl[:,:,N,:,c,N]*dDlln[:,:,:,:,b,:] + Dnl[:,:,N,:,c,N]*dAlln[:,:,:,:,b,:]) ).real + 2*( Dnl[:,:,N,:,b,N]*dDlln[:,:,:,:,c,:]  -  Dnl[:,:,N,:,c,N]*dDlln[:,:,:,:,b,:]  ).imag
        uoo = -2*((Anl[:,:,N,:,b,N]*dDlnn[:,:,:,:,c,:] + Dnl[:,:,N,:,b,N]*dAlnn[:,:,:,:,c,:]) - (Anl[:,:,N,:,c,N]*dDlnn[:,:,:,:,b,:] + Dnl[:,:,N,:,c,N]*dAlnn[:,:,:,:,b,:]) ).real + 2*( Dnl[:,:,N,:,b,N]*dDlnn[:,:,:,:,c,:]  -  Dnl[:,:,N,:,c,N]*dDlnn[:,:,:,:,b,:]  ).imag

        return {'i':o,'oi':uo,'oii':uoo,'ooi':uuo}

    @lazy_property.LazyProperty
    def derHplusTr(self):
        b=alpha_A
        c=beta_A
        N=None
        E=self.E_K
        dHn, dHln = self.gdHbar
        dOn, dOln = self.gdOmegabar
        Onn = self.Omega_Hbar.transpose(0,2,1,3)
        V = self.V_H
        Bplus = self.B_Hbarplus_dagger
        Dln = self.D_H
        dBPln, dBPlln, dBPlnn = self.gdBbarplus
        Dnl = self.D_H.transpose(0,2,1,3)
        dDln, dDlln,dDlnn= self.gdD
        #term 1
        o =(dHn + dOn*E[:,:,N,N]).real
        oo =(Onn[:,:,:,:,N]*V[:,:,:,N,:]).real
        uo =(dHln + dOln*E[:,N,:,N,N]).real
        #term 2
        uo += -2*((Bplus[:,:,:,b,N]*dDln[:,:,:,c,:] + Dnl[:,:,:,b,N]*dBPln[:,:,:,c,:] ) - (Bplus[:,:,:,c,N]*dDln[:,:,:,b,:] + Dnl[:,:,:,c,N]*dBPln[:,:,:,b,:])).real
        uuo = -2*((Bplus[:,:,N,:,b,N]*dDlln[:,:,:,:,c,:] + Dnl[:,:,N,:,b,N]*dBPlln[:,:,:,:,c,:]) - (Bplus[:,:,N,:,c,N]*dDlln[:,:,:,:,b,:] + Dnl[:,:,N,:,c,N]*dBPlln[:,:,:,:,b,:])).real
        uoo = -2*((Bplus[:,:,N,:,b,N]*dDlnn[:,:,:,:,c,:] + Dnl[:,:,N,:,b,N]*dBPlnn[:,:,:,:,c,:]) - (Bplus[:,:,N,:,c,N]*dDlnn[:,:,:,:,b,:] + Dnl[:,:,N,:,c,N]*dBPlnn[:,:,:,:,b,:])).real
        #term 3
        uo += 2*(E[:,:,N,N,N] + E[:,N,:,N,N])*( Dnl[:,:,:,b,N]*dDln[:,:,:,c,:]  -  Dnl[:,:,:,c,N]*dDln[:,:,:,b,:]  ).imag
        uuo +=2*(E[:,:,N,N,N,N] + E[:,N,N,:,N,N])*( Dnl[:,:,N,:,b,N]*dDlln[:,:,:,:,c,:]  -  Dnl[:,:,N,:,c,N]*dDlln[:,:,:,:,b,:]  ).imag
        uoo +=2*(E[:,:,N,N,N,N] + E[:,N,N,:,N,N])*( Dnl[:,:,N,:,b,N]*dDlnn[:,:,:,:,c,:]  -  Dnl[:,:,N,:,c,N]*dDlnn[:,:,:,:,b,:]  ).imag
        #term 4
        uuo += (Dnl[:,:,N,:,b,N]*V[:,:,:,N,N,:]*Dln[:,N,:,:,c,N] - Dnl[:,:,N,:,c,N]*V[:,:,:,N,N,:]*Dln[:,N,:,:,b,N] ).imag
        uoo += (Dnl[:,:,N,:,b,N]*Dln[:,:,:,N,c,N]*V[:,N,:,:,N,:] - Dnl[:,:,N,:,c,N]*Dln[:,:,:,N,b,N]*V[:,N,:,:,N,:]).imag
        
        return {'i':o,'ii':oo,'oi':uo,'oii':uoo,'ooi':uuo}


    @lazy_property.LazyProperty
    def A_Hbar(self):
        return self._R_to_k_H(self.AA_R.copy())

    @lazy_property.LazyProperty
    def A_H(self):
        '''Generalized Berry connection matrix, A^(H) as defined in eqn. (25) of 10.1103/PhysRevB.74.195118.'''
        return self.A_Hbar + 1j*self.D_H

    @lazy_property.LazyProperty
    def A_Hbar_der(self):
        return  self._R_to_k_H(self.AA_R.copy(), der=1) 

    @lazy_property.LazyProperty
    def S_H(self):
        return  self._R_to_k_H( self.SS_R.copy() )

    @lazy_property.LazyProperty
    def S_H_rediag(self):
        return np.einsum("knna->kna",self.S_H).real

    @lazy_property.LazyProperty
    def delS_H(self):
        """d_b S_a """
        return  self._R_to_k_H( self.SS_R[:,:,:,:,None], der=1 )

    @lazy_property.LazyProperty
    def delS_H_rediag(self):
#  d_b S_a
        return np.einsum("knnab->knab",self.delS_H).real

    @lazy_property.LazyProperty
    def Omega_Hbar(self):
        print_my_name_start()
        return  -self._R_to_k_H( self.AA_R, der=1, asym_after=True) 

    @lazy_property.LazyProperty
    def B_Hbar(self):
        print_my_name_start()
        _BB_K=self._R_to_k_H( self.BB_R.copy(),hermitian=False)
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
    def Omega_Hbar_E(self):
         print_my_name_start()
         return np.einsum("km,kmma->kma",self.E_K,self.Omega_Hbar).real



    @lazy_property.LazyProperty
    def A_E_A(self):
         print_my_name_start()
         return np.einsum("kn,knma,kmna->kmna",self.E_K,self.A_Hbar[:,:,:,alpha_A],self.A_Hbar[:,:,:,beta_A]).imag




#  for effective mass
    @lazy_property.LazyProperty
    def Db_Va_re(self):
         print_my_name_start()
         return (self.D_H[:,:,:,None,:]*self.V_H.transpose(0,2,1,3)[:,:,:,:,None]  - 
                   self.D_H.transpose(0,2,1,3)[:,:,:,None,:]  *self.V_H[:,:,:,:,None]
                   ).real

#  for spin derivative
    @lazy_property.LazyProperty
    def Db_Sa_re(self):
         print_my_name_start()
         return (self.D_H[:,:,:,None,:]*self.S_H.transpose(0,2,1,3)[:,:,:,:,None]  - 
                   self.D_H.transpose(0,2,1,3)[:,:,:,None,:]  *self.S_H[:,:,:,:,None]
                   ).real
               


    @lazy_property.LazyProperty
    def D_B(self):
         print_my_name_start()
         tmp=self.D_H.transpose((0,2,1,3))
         return ( (tmp[:,:,:,alpha_A] * self.B_Hbar[:,:,:,beta_A ]).real-
                  (tmp[:,:,:,beta_A ] * self.B_Hbar[:,:,:,alpha_A]).real  )




    @lazy_property.LazyProperty
    def D_E_A(self):
         print_my_name_start()
         return np.array([
                  np.einsum("n,nma,mna->mna",ee,aa[:,:,alpha_A],dh[:,:,beta_A ]).real+
                  np.einsum("n,mna,nma->mna",ee,aa[:,:,beta_A ],dh[:,:,alpha_A]).real 
                    for ee,aa,dh in zip(self.E_K,self.A_Hbar,self.D_H)])
         
    @lazy_property.LazyProperty
    def D_E_D(self):
         print_my_name_start()
         X=-np.einsum("km,knma,kmna->kmna",self.E_K,self.D_H[:,:,:,alpha_A],self.D_H[:,:,:,beta_A ]).imag
         return (   X,-X.transpose( (0,2,1,3) ) )    #-np.einsum("km,knma,kmna->kmna",self.E_K,self.D_H[:,:,:,alpha_A],self.D_H[:,:,:,beta_A ]).imag ,



    @lazy_property.LazyProperty
    def Omega_bar_der(self):
        print_my_name_start()
        _OOmega_K =  self.FFT_R_to_k( (
                        self.AA_R[:,:,:,alpha_A]*self.cRvec[None,None,:,beta_A ] -     
                        self.AA_R[:,:,:,beta_A ]*self.cRvec[None,None,:,alpha_A])[:,:,:,:,None]*self.cRvec[None,None,:,None,:]   , hermitian=True )
        return self._rotate(_OOmega_K)

    @lazy_property.LazyProperty
    def Omega_bar_der_rediag(self):
        return np.einsum("knnad->knad",self.Omega_bar_der).real

    @lazy_property.LazyProperty
    def Omega_bar_D_re(self):
        return (self.Omega_Hbar.transpose(0,2,1,3)[:,:,:,:,None]*self.D_H[:,:,:,None,:]).real


##  properties directly accessed by fermisea2 
    @property 
    def Omega(self):
        oi=( (self.D_H[:,:,:,alpha_A].transpose((0,2,1,3))*self.A_Hbar[:,:,:,beta_A]).real+
               (self.D_H[:,:,:,beta_A]*self.A_Hbar[:,:,:,alpha_A].transpose((0,2,1,3))).real  ) 
        oi+=(-self.D_H[:,:,:,beta_A]*self.D_H[:,:,:,alpha_A].transpose((0,2,1,3))).imag
        i=np.einsum("kiia->kia",self.Omega_Hbar).real
        return {'i':i,'oi': - 2*oi }

    @property
    def Ohmic(self):
        return {'i':self.del2E_H_diag,'oi':self.Db_Va_re}

    @property
    def gyroKspin(self):
        return {'i':self.delS_H_rediag,'oi':self.Db_Sa_re}

    @property
    def SpinTot(self):
        return {'i':self.S_H_rediag}




    def Hplusminus(self,sign,evalJ0=True,evalJ1=True,evalJ2=True):
        assert sign in (1,-1) , "sign should be +1 or -1"
        from collections import defaultdict
        res = defaultdict( lambda : 0)
        if evalJ0:
            if sign==1:
                res['ii']=-2*data.A_E_A
            res['i']+=self.Morb_Hbar_diag + sign*self.Omega_Hbar_E
        if evalJ1:
            res['oi']+=-2*(self.D_B+sign*self.D_E_A)
        if evalJ2:
            C,D=self.D_E_D
            res['oi']+=-2*(C+sign*D)
        return  res

    def Hplus(self,evalJ0=True,evalJ1=True,evalJ2=True):
        return self.Hplusminus(self,+1,evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)

    def Hminus(self,evalJ0=True,evalJ1=True,evalJ2=True):
        return self.Hplusminus(self,-1,evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)


