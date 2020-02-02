#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                     written by                             #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------

import numpy as np
import lazy_property

from .__system import System
from .__utility import  print_my_name_start,print_my_name_end,einsumk, fourier_R_to_k, alpha_A,beta_A
   
class Data_dk(System):
    def __init__(self,system,dk=None,AA=None,BB=None,CC=None,SS=None,NKFFT=None):
        self.spinors=system.spinors
        self.iRvec=system.iRvec
        self.real_lattice=system.real_lattice
        self.recip_lattice=system.recip_lattice
        self.NKFFT=system.NKFFT if NKFFT is None else NKFFT
        self.num_wann=system.num_wann

        if dk is not None:
            expdk=np.exp(2j*np.pi*self.iRvec.dot(dk))
            self.dk=dk
        else:
            expdk=np.ones(self.nRvec)
            self.dk=np.zeros(3)
 
        self.HH_R=system.HH_R[:,:,:]*expdk[None,None,:]
        
        for X in ['AA','BB','CC','SS']:
            XR=X+'_R'
            XX=vars()[X]
            if XX in (None,True):
                try:
                    vars(self)[XR]=vars(system)[XR]*expdk[None,None,:,None]
                except KeyError:
                    if XX : raise AttributeError(XR+" is not defined")



    def _rotate(self,mat):
        print_my_name_start()
        return  np.array([a.dot(b).dot(c) for a,b,c in zip(self.UUH_K,mat,self.UU_K)])


    def _rotate_vec(self,mat):
        print_my_name_start()
        res=np.array(mat)
        for i in range(res.shape[-1]):
            res[:,:,:,i]=self._rotate(mat[:,:,:,i])
        print_my_name_start()
        return res

    def _rotate_mat(self,mat):
        print_my_name_start()
        res=np.array(mat)
        for j in range(res.shape[-1]):
            res[:,:,:,:,j]=self._rotate_vec(mat[:,:,:,:,j])
        print_my_name_start()
        return res


    @lazy_property.LazyProperty
    def nbands(self):
        return self.HH_R.shape[0]



    @lazy_property.LazyProperty
    def kpoints_all(self):
        dkx,dky,dkz=1./self.NKFFT
        return np.array([self.dk-np.array([ix*dkx,iy*dky,iz*dkz]) 
          for ix in range(self.NKFFT[0])
              for iy in range(self.NKFFT[1])
                  for  iz in range(self.NKFFT[2])])%1


    @lazy_property.LazyProperty
    def NKFFT_tot(self):
        return np.prod(self.NKFFT)


#    defining sets of degenerate states.  
# all states below Emin are considered "degenerate" - needed for fermi-sea integral of Berry curvature only (AHC)
# all states above Emax are neglected 
    def set_degen(self,degen_thresh=1e-10,Emin=-np.Inf,Emax=np.Inf):
        try: 
            self._degen
            self._E_K_degen
            print ("degeneracies were already set")
            return
        except AttributeError:
            A=[np.where(E[1:]-E[:-1]>degen_thresh)[0]+1 for E in self.E_K ]
    #        A=[a[:-1][E[a[1:]-1]>=Emin] for E,a in zip(self.E_K,A)]
            A=[ [0,]+list(a)+[len(E)] for a,E in zip(A,self.E_K) ]
            self._degen= [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:]) if e[ib1]<=Emax]    for a,e in zip(A,self.E_K)]
            self._E_K_degen=[np.array([np.mean(E[ib1:ib2]) for ib1,ib2 in deg]) for deg,E in zip(self._degen,self.E_K)]

    @property 
    def degen(self):
        try:
            return self._degen 
        except AttributeError:
            raise RuntimeError("Degeneracies were not set. use the set_degen method first")

    @property 
    def E_K_degen(self):
        try:
            return self._E_K_degen 
        except AttributeError:
            raise RuntimeError("Degenerate energies were not set. use the set_degen method first")

    @lazy_property.LazyProperty
    def vel_nonabelian(self):
        return [ [S[ib1:ib2,ib1:ib2] for ib1,ib2 in deg] for S,deg in zip(self.V_H,self.degen)]

    @lazy_property.LazyProperty
    def spin_nonabelian(self):
        return [ [S[ib1:ib2,ib1:ib2] for ib1,ib2 in deg] for S,deg in zip(self.S_H,self.degen)]


##  TODO: When it works correctly - think how to optimize it
    @lazy_property.LazyProperty
    def Berry_nonabelian(self):
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]
        return [ [ O[ib1:ib2,ib1:ib2,:]-1j*sum(s*np.einsum("mla,lna->mna",A[ib1:ib2,ib1:ib2,b],A[ib1:ib2,ib1:ib2,c]) for s,b,c in sbc) 
               +sum(s*np.einsum("mla,lna->mna",X,Y) 
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
                     for s,b,c in sbc
                    for X,Y in [(-D[ib1:ib2,ibl1:ibl2,b],A[ibl1:ibl2,ib1:ib2,c]),(A[ib1:ib2,ibl1:ibl2,c],D[ibl1:ibl2,ib1:ib2,b]),(-1j*D[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c])]
                           )
                        for ib1,ib2 in deg]
                     for O,A,D,deg in zip( self.Omega_Hbar,self.A_Hbar,self.D_H,self.degen) ] 


##  TODO: When it works correctly - think how to optimize it
    @lazy_property.LazyProperty
    def Morb_nonabelian(self):
        sbc=[(+1,alpha_A,beta_A),(-1,beta_A,alpha_A)]        
        Morb=[ [ M[ib1:ib2,ib1:ib2,:]-e*O[ib1:ib2,ib1:ib2,:]
    #         -1j*e*sum(s*np.einsum("mla,lna->mna",A[ib1:ib2,ib1:ib2,b],A[ib1:ib2,ib1:ib2,c]) for s,b,c in sbc)
               +sum(s*np.einsum("mla,lna->mna",X,Y) 
                   for ibl1,ibl2 in (([  (0,ib1)]  if ib1>0 else [])+ ([  (ib2,self.num_wann)]  if ib2<self.num_wann else []))
                     for s,b,c in sbc
                    for X,Y in [
                    (-D[ib1:ib2,ibl1:ibl2,b],-B[ibl1:ibl2,ib1:ib2,c]),
                    (B.transpose((1,0,2)).conj()[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c]),
                        (-1j*0*V[ib1:ib2,ibl1:ibl2,b],D[ibl1:ibl2,ib1:ib2,c])]
                           )
                        for (ib1,ib2),e in zip(deg,E)]
                     for M,O,A,B,D,V,deg,E,EK in zip( self.Morb_Hbar,self.Omega_Hbar,self.A_Hbar,self.B_Hbarbar,self.D_H,self.V_H,self.degen,self.E_K_degen,self.E_K) ]
        for ik,M in enumerate(Morb):
            for m,e,d in zip(M,self.E_K_degen[ik],self.degen[ik]):
                if np.linalg.norm( m-m.transpose((1,0,2)).conj() ) > 1e-10: 
                     raise RuntimeError("for ik={} e={} ({}-fold) the Morb is non-Hermitian:{} \n{}\n".format(ik,e,d[1]-d[0],np.linalg.norm( m-m.transpose((1,0,2)).conj() ),m)) 
        return Morb
        
        

    @lazy_property.LazyProperty
    def HH_K(self):
        return fourier_R_to_k(self.HH_R,self.iRvec,self.NKFFT,hermitian=True)


    @lazy_property.LazyProperty
    def E_K(self):
        print_my_name_start()
        EUU=[np.linalg.eigh(Hk) for Hk in self.HH_K]
        E_K=np.array([euu[0] for euu in EUU])
        self._UU_K =np.array([euu[1] for euu in EUU])
        print_my_name_end()
        return E_K


    @lazy_property.LazyProperty
    def UU_K(self):
        self.E_K
        return self._UU_K


    @lazy_property.LazyProperty
    def HHUU_K(self):
        return self._rotate(self.HH_K)


    @lazy_property.LazyProperty
    def delHH_K(self):
        print_my_name_start()
        self.E_K
        delHH_R=1j*self.HH_R[:,:,:,None]*self.cRvec[None,None,:,:]
        return fourier_R_to_k(delHH_R,self.iRvec,self.NKFFT,hermitian=True)


    @lazy_property.LazyProperty
    def del2HHUU_K(self):
        return self._rotate_vec(self.del2HH_K)

    @lazy_property.LazyProperty
    def delE_K(self):
        print_my_name_start()
        delE_K = np.einsum("klla->kla",self.delHHUU_K)
        check=np.abs(delE_K).imag.max()
        if check>1e-10: raiseruntimeError ("The band derivatives have considerable imaginary part: {0}".format(check))
        return delE_K.real


    @lazy_property.LazyProperty
    def del2E_K(self):
        print_my_name_start()
        del2HH=1j*self.HH_R[:,:,:,None,None]*self.cRvec[None,None,:,None,:]*self.cRvec[None,None,:,:,None]
        del2HH = fourier_R_to_k(del2HH,self.iRvec,self.NKFFT,hermitian=True)
        del2HH=self._rotate_mat(del2HH)
        del2E_K = np.array([del2HH[:,i,i,:,:] for i in range(del2HH.shape[1])]).transpose( (1,0,2,3) )
        check=np.abs(del2E_K).imag.max()
        if check>1e-10: raiseruntimeError( "The second band derivatives have considerable imaginary part: {0}".format(check) )
        return delE2_K.real


    @lazy_property.LazyProperty
    def UU_K(self):
        print_my_name_start()
        self.E_K
        return self._UU_K


    @lazy_property.LazyProperty
    def UUH_K(self):
        print_my_name_start()
        return self.UU_K.conj().transpose((0,2,1))


    @lazy_property.LazyProperty
    def D_H(self):
            print_my_name_start()
            _delHH_K_=np.copy(self.delHHUU_K)
            dEig_threshold=1e-14
            dEig=self.E_K[:,:,None]-self.E_K[:,None,:]
            select=abs(dEig)<dEig_threshold
            dEig[select]=dEig_threshold
            _delHH_K_[select]=0
            return -_delHH_K_/dEig[:,:,:,None]

    @lazy_property.LazyProperty
    def V_H(self):
        return self._rotate_vec(self.delHH_K)

    @lazy_property.LazyProperty
    def Morb_Hbar(self):
        print_my_name_start()
        _CC_K=fourier_R_to_k( self.CC_R,self.iRvec,self.NKFFT)
        return self._rotate_vec( _CC_K )


    @lazy_property.LazyProperty
    def delHH_dE_SQ_K(self):
         print_my_name_start()
         return  (self.delHH_dE_K[:,:,:,beta_A]
                         *self.delHH_dE_K[:,:,:,alpha_A].transpose((0,2,1,3))).imag

    
    @lazy_property.LazyProperty
    def A_Hbar(self):
        print_my_name_start()
        _AA_K=fourier_R_to_k( self.AA_R,self.iRvec,self.NKFFT,hermitian=True)
        return self._rotate_vec( _AA_K )



    @lazy_property.LazyProperty
    def S_H(self):
        print_my_name_start()
        _SS_K=fourier_R_to_k( self.SS_R,self.iRvec,self.NKFFT)
        return self._rotate_vec( _SS_K )


    @lazy_property.LazyProperty
    def Omega_Hbar(self):
        print_my_name_start()
        _OOmega_K =  fourier_R_to_k( -1j*(
                        self.AA_R[:,:,:,alpha_A]*self.cRvec[None,None,:,beta_A ] -     
                        self.AA_R[:,:,:,beta_A ]*self.cRvec[None,None,:,alpha_A])   , self.iRvec, self.NKFFT,hermitian=True )
        return self._rotate_vec(_OOmega_K)


    @lazy_property.LazyProperty
    def B_Hbar(self):
        print_my_name_start()
        _BB_K=fourier_R_to_k( self.BB_R,self.iRvec,self.NKFFT)
        return self._rotate_vec( _BB_K )

    @lazy_property.LazyProperty
    def B_Hbarbar(self):
        print_my_name_start()
        return self.B_Hbar-self.A_Hbar[:,:,:,:]*self.E_K[:,None,:,None]


#### These are only needed for the "old" (abelian) routines:
#### TODO : when the non-abelian works - may be removed
    @lazy_property.LazyProperty
    def FF_K_rediag(self):
        print_my_name_start()
        _FF_K=fourier_R_to_k( self.FF_R,self.iRvec,self.NKFFT)
#        return np.einsum("kml,kmna,knl->kla",self.UUC_K,_CC_K,self.UU_K).real
        return np.einsum("kmm->km",_FF_K).imag

    @lazy_property.LazyProperty
    def SSUU_K_rediag(self):
        print_my_name_start()
        _SS_K=fourier_R_to_k( self.SS_R,self.iRvec,self.NKFFT)
        _SS_K=self._rotate_vec( _SS_K )
#        return np.einsum("kml,kmna,knl->kla",self.UUC_K,_CC_K,self.UU_K).real
        return np.einsum("kmma->kma",_SS_K).real


    @lazy_property.LazyProperty
    def HHOOmegaUU_K(self):
         print_my_name_start()
         return np.einsum("kmn,knma->kma",self.HHUU_K,self.OOmegaUU_K).real

    @lazy_property.LazyProperty
    def Omega_Hbar_diag(self):
        print_my_name_start()
        return  np.einsum("kiia->kia",self.OOmegaUU_K).real


    @lazy_property.LazyProperty
    def HHAAAAUU_K(self):
#        print ("shapes:",self.HHUU_K.shape,self.AAUU_K[:,:,:,alpha_A].shape,self.AAUU_K[:,:,:,beta_A].shape)
         print_my_name_start()
         return np.einsum("kmi,kina,knma->knma",self.HHUU_K,self.AAUU_K[:,:,:,alpha_A],self.AAUU_K[:,:,:,beta_A]).imag


    @property
    def BBUU_K(self):
        return self.B_Hbar


#        return np.einsum("kml,kmna,knp->klpa",self.UUC_K,_BB_K,self.UU_K)




    @lazy_property.LazyProperty
    def CCUU_K_rediag(self):
        print_my_name_start()
        _CC_K=fourier_R_to_k( self.CC_R,self.iRvec,self.NKFFT)
        _CC_K=self._rotate_vec( _CC_K )
        return np.einsum("klla->kla",_CC_K).real
#        return np.einsum("kml,kmna,knl->kla",self.UUC_K,_CC_K,self.UU_K).real


#    @lazy_property.LazyProperty
#    def delHH_dE_AA_K(self):
#         return ( (self.delHH_dE_K[:,:,:,alpha_A]*self.AAUU_K.transpose((0,2,1,3))[:,:,:,beta_A]).imag+
#               (self.delHH_dE_K.transpose((0,2,1,3))[:,:,:,beta_A]*self.AAUU_K[:,:,:,alpha_A]).imag  )


    @lazy_property.LazyProperty
    def delHH_dE_AA_K(self):
         print_my_name_start()
         return ( (self.delHH_dE_K[:,:,:,alpha_A].transpose((0,2,1,3))*self.AAUU_K[:,:,:,beta_A]).imag+
               (self.delHH_dE_K[:,:,:,beta_A]*self.AAUU_K[:,:,:,alpha_A].transpose((0,2,1,3))).imag  )


    @lazy_property.LazyProperty
    def delHH_dE_AA_delHH_dE_SQ_K(self):
         print_my_name_start()
         return ( (self.delHH_dE_K[:,:,:,alpha_A].transpose((0,2,1,3))*self.AAUU_K[:,:,:,beta_A]).imag+
               (self.delHH_dE_K[:,:,:,beta_A]*self.AAUU_K[:,:,:,alpha_A].transpose((0,2,1,3))).imag  +
                 (self.delHH_dE_K[:,:,:,beta_A]
                         *self.delHH_dE_K[:,:,:,alpha_A].transpose((0,2,1,3))).imag  )



    @lazy_property.LazyProperty
    def delHH_dE_BB_K(self):
         print_my_name_start()
         tmp=self.delHH_dE_K.transpose((0,2,1,3))
         return ( (tmp[:,:,:,alpha_A] * self.BBUU_K[:,:,:,beta_A ]).imag-
                  (tmp[:,:,:,beta_A ] * self.BBUU_K[:,:,:,alpha_A]).imag  )

#         return ( (self.delHH_dE_K[:,:,:,alpha_A]*self.BBUU_K.transpose((0,2,1,3))[:,:,:,beta_A]).imag-
#               (self.delHH_dE_K.transpose((0,2,1,3))[:,:,:,beta_A]*self.BBUU_K[:,:,:,alpha_A]).imag  )

    @lazy_property.LazyProperty
    def delHH_dE_HH_AA_K(self):
         print_my_name_start()
#         return ( np.einsum(  "knl,klma,kmna->kmna",self.HHUU_K,self.AAUU_K[:,:,:,alpha_A],self.delHH_dE_K[:,:,:,beta_A ]).imag+
#                    np.einsum("kln,kmla,knma->kmna",self.HHUU_K,self.AAUU_K[:,:,:,beta_A ],self.delHH_dE_K[:,:,:,alpha_A]).imag )
         return np.array([
                  np.einsum("nl,lma,mna->mna",hh,aa[:,:,alpha_A],delhh[:,:,beta_A ]).imag+
                  np.einsum("ln,mla,nma->mna",hh,aa[:,:,beta_A ],delhh[:,:,alpha_A]).imag 
                    for hh,aa,delhh in zip(self.HHUU_K,self.AAUU_K,self.delHH_dE_K)])
         
    @lazy_property.LazyProperty
    def delHH_dE_SQ_HH_K(self):
         print_my_name_start()
         return ( np.einsum("kml,knma,klna->klmna",self.HHUU_K,self.delHH_dE_K[:,:,:,alpha_A],self.delHH_dE_K[:,:,:,beta_A ]).imag ,
                  np.einsum("knm,kmla,klna->klmna",self.HHUU_K,self.delHH_dE_K[:,:,:,alpha_A],self.delHH_dE_K[:,:,:,beta_A ]).imag ) 





### TODO: old names - to be wiped out from other routines
    @property
    def OOmegaUU_K(self):
        return self.Omega_Hbar

    @property
    def OOmegaUU_K_rediag(self):
        return self.Omega_Hbar_diag
         
    @property
    def AAUU_K(self):
        return self.A_Hbar

    @lazy_property.LazyProperty
    def delHH_dE_K(self):
            return 1j*self.D_H

    @property
    def delHHUU_K(self):
        return self.V_H

    @property
    def CCUU_K(self):
       return self.Morb_Hbar

    @lazy_property.LazyProperty
    def SSUU_K(self):
        return self.S_H
        print_my_name_start()
        _SS_K=fourier_R_to_k( self.SS_R,self.iRvec,self.NKFFT)
        return self._rotate_vec( _SS_K )



unused="""

#    @lazy_property.LazyProperty
#    def UUU_K(self):
#        return self.UU_K[:,:,None,:]*self.UUC_K[:,None,:,:]


    def get_OOmega_K(self):
        try:
            return self._OOmega_K
        except AttributeError:
            print "running get_OOmega_K.."
            self._OOmega_K=    -1j* fourier_R_to_k( 
                        self.AA_R[:,:,:,alpha_A]*self.cRvec[None,None,:,beta_A ] - 
                        self.AA_R[:,:,:,beta_A ]*self.cRvec[None,None,:,alpha_A]   , self.iRvec, self.NKFFT )
             
            return self._OOmega_K


    def get_AA_K(self):
        try:
            return self._AA_K
        except AttributeError:
            print "running get_AA_K.."
            self._AA_K=fourier_R_to_k( self.AA_R,self.iRvec,self.NKFFT)
            return self._AA_K


"""