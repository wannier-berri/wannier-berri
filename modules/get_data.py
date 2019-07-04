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
from scipy.io import FortranFile as FF
from aux import str2bool
import wan_ham as wham
import copy


class Data():

    def __init__(self,seedname="wannier90",tb_file=None,getAA=False,getBB=False,getCC=False,getSS=False,NKFFT=None):
        if tb_file is not None:
            self.__from_tb_file(tb_file,getAA=getAA,NKFFT=NKFFT)
            return
        f=open(seedname+"_HH_save.info","r")
        l=f.readline().split()[:3]
        self.seedname=seedname
        self.num_wann,nRvec,self.spinors=int(l[0]),int(l[1]),str2bool(l[2])
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        iRvec=np.array([f.readline().split()[:4] for i in range(nRvec)],dtype=int)
        f.close()
        self.cell_volume=np.linalg.det(self.real_lattice)
        self.Ndegen=iRvec[:,3]
        self.iRvec=iRvec[:,:3]
        self.cRvec=self.iRvec.dot(self.real_lattice)
        if NKFFT is None:
            self.NKFFT=np.abs(self.iRvec).max(axis=0)*2+1
        else:
            self.NKFFT=NKFFT

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Number of K points:", self.NKFFT)
        print ("Real-space lattice:\n",self.real_lattice)
        #print ("R - points and dege=neracies:\n",iRvec)
        
        self.HH_R=self.__getMat('HH')
        
        if getAA:
            self.AA_R=self.__getMat('AA')
            
        if getBB:
            self.BB_R=self.__getMat('BB')

        if getCC:
            self.CC_R=self.__getMat('CC')

        if getSS:
            self.SS_R=self.__getMat('SS')


    def __from_tb_file(self,tb_file=None,getAA=False,NKFFT=None):
        self.seedname=tb_file.split("/")[-1].split("_")[0]
        f=open(tb_file,"r")
        l=f.readline()
        print "reading TB file {0} ( {1} )".format(tb_file,l)
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.num_wann=int(f.readline())
        nRvec=int(f.readline())
        self.spinors=None
        self.Ndegen=[]
        while len(self.Ndegen)<nRvec:
            self.Ndegen+=f.readline().split()
        self.Ndegen=np.array(self.Ndegen,dtype=int)
        
        self.iRvec=[]
        
        self.HH_R=np.zeros( (self.num_wann,self.num_wann,nRvec) ,dtype=complex)
        
        for ir in range(nRvec):
            f.readline()
            self.iRvec.append(f.readline().split())
            hh=np.array( [[f.readline().split()[2:4] 
                             for n in range(self.num_wann)] 
                                for m in range(self.num_wann)],dtype=float).transpose( (1,0,2) )
            self.HH_R[:,:,ir]=(hh[:,:,0]+1j*hh[:,:,1])/self.Ndegen[ir]
        
        self.iRvec=np.array(self.iRvec,dtype=int)
#        print "iRvec=\n",self.iRvec
        
        if getAA:
          self.AA_R=np.zeros( (self.num_wann,self.num_wann,nRvec,3) ,dtype=complex)
          for ir in range(nRvec):
            f.readline()
#            irvec=
            assert (np.array(f.readline().split(),dtype=int)==self.iRvec[ir]).all()
            aa=np.array( [[f.readline().split()[2:8] 
                             for n in range(self.num_wann)] 
                                for m in range(self.num_wann)],dtype=float)
#            print (aa.shape)
            self.AA_R[:,:,ir,:]=(aa[:,:,0::2]+1j*aa[:,:,1::2]).transpose( (1,0,2) ) /self.Ndegen[ir]
        
        
        f.close()

        self.cell_volume=np.linalg.det(self.real_lattice)
        self.cRvec=self.iRvec.dot(self.real_lattice)
        if NKFFT is None:
            self.NKFFT=np.abs(self.iRvec).max(axis=0)*2+1
        else:
            self.NKFFT=NKFFT

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Number of K points:", self.NKFFT)
        print ("Real-space lattice:\n",self.real_lattice)
        #print ("R - points and dege=neracies:\n",iRvec)
        


   
    def get_nRvec(self):
        return self.iRvec.shape[0]
    
    nRvec=property(get_nRvec)

    def __getMat(self,suffix):

        f=FF(self.seedname+"_" + suffix+"_R.dat")
        MM_R=np.array([[np.array(f.read_record('2f8'),dtype=float) for m in range(self.num_wann)] for n in range(self.num_wann)])
        MM_R=MM_R[:,:,:,0]+1j*MM_R[:,:,:,1]
        f.close()
        ncomp=MM_R.shape[2]/self.nRvec
        if ncomp==1:
            print "reading 0d for ",suffix
            return MM_R/self.Ndegen[None,None,:]
        elif ncomp==3:
            print "reading 1d for ",suffix
            return MM_R.reshape(self.num_wann, self.num_wann, 3, self.nRvec).transpose(0,1,3,2)/self.Ndegen[None,None,:,None]
        elif ncomp==9:
            print "reading 2d for ",suffix
            return MM_R.reshape(self.num_wann, self.num_wann, 3,3, self.nRvec).transpose(0,1,4,3,2)/self.Ndegen[None,None,:,None,None]



class Data_dk(Data):
    def __init__(self,data,dk=None,AA=None,BB=None,CC=None,SS=None,NKFFT=None):
        self.num_wann=data.num_wann
        self.spinors=data.spinors
        self.iRvec=data.iRvec
        self.cRvec=data.cRvec
        self.cell_volume=data.cell_volume
        self.NKFFT=data.NKFFT if NKFFT is None else NKFFT
        
        if dk is not None:
            expdk=np.exp(2j*np.pi*self.iRvec.dot(dk))
        else:
            expdk=np.ones(self.nRvec)


        self.HH_R=data.HH_R[:,:,:]*expdk[None,None,:]
        
        if AA in (None,True):
            try:
                self.AA_R=data.AA_R[:,:,:,:]*expdk[None,None,:,None]
            except AttributeError:
                if AA : raise AttributeError("AA_R is not defined")


    def get_AA_K(self):
        try:
            return self._AA_K
        except AttributeError:
            print "running get_AA_K.."
            self._AA_K=wham.fourier_R_to_k( self.AA_R,self.iRvec,self.NKFFT)
            return self._AA_K


    def get_AAUU_K(self):
        try:
            return self._AAUU_K
        except AttributeError:
            print "running get_AAUU_K.."
            _AA_K=wham.fourier_R_to_k( self.AA_R,self.iRvec,self.NKFFT)
            self._AAUU_K=np.einsum("kml,kmna,knp->klpa",self.UUC_K,_AA_K,self.UU_K)
            return self._AAUU_K
    
            
    def get_OOmega_K(self):
        try:
            return self._OOmega_K
        except AttributeError:
            print "running get_OOmega_K.."
            self._OOmega_K=    -1j* wham.fourier_R_to_k( 
                        self.AA_R[:,:,:,wham.alpha]*self.cRvec[None,None,:,wham.beta ] - 
                        self.AA_R[:,:,:,wham.beta ]*self.cRvec[None,None,:,wham.alpha]   , self.iRvec, self.NKFFT )
             
            return self._OOmega_K


    def get_OOmegaUU_K(self):
        try:
            return self._OOmegaUU_K
        except AttributeError:
            print "running get_OOmegaUU_K.."
            _OOmega_K=    -1j* wham.fourier_R_to_k( 
                        self.AA_R[:,:,:,wham.alpha]*self.cRvec[None,None,:,wham.beta ] - 
                        self.AA_R[:,:,:,wham.beta ]*self.cRvec[None,None,:,wham.alpha]   , self.iRvec, self.NKFFT )
            self._OOmegaUU_K=np.einsum("knmi,kmna->kia",self.UUU_K,_OOmega_K).real
            return self._OOmegaUU_K


    def _get_eig_deleig(self):
        print "running get_eog_deleig.."
        self._E_K,self._delE_K, self._UU_K, self._HH_K, self._delHH_K =   wham.get_eig_deleig(self.NKFFT,self.HH_R,self.iRvec,self.cRvec)

    
    def get_E_K(self):
        try:
            return self._E_K
        except AttributeError:
            self._get_eig_deleig()
            return self._E_K

    def get_delE_K(self):
        try:
            return self._delE_K
        except AttributeError:
            self._get_eig_deleig()
            return self._delE_K

    def get_UU_K(self):
        try:
            return self._UU_K
        except AttributeError:
            self._get_eig_deleig()
            return self._UU_K


    def get_UUC_K(self):
        try:
            return self._UUC_K
        except AttributeError:
            self._UUC_K=self.UU_K.conj()
            return self._UUC_K


    def get_UUU_K(self):
        try:
            return self._UUU_K
        except AttributeError:
            self._UUU_K=self.UU_K[:,:,None,:]*self.UU_K.conj()[:,None,:,:]
            return self._UUU_K


    def get_HH_K(self):
        try:
            return self._HH_K
        except AttributeError:
            self._get_eig_deleig()
            return self._HH_K

    def get_delHH_K(self):
        try:
            return self._delHH_K
        except AttributeError:
            self._get_eig_deleig()
            return self._delHH_K

    def get_delHH_dE_K(self):
        try:
            return self._delHH_dE_K
        except AttributeError:
            _delHH_K_=np.einsum("kml,kmna,knp->klpa",self.UU_K.conj(),self.delHH_K,self.UU_K)
            dEig_threshold=1e-14
            dEig=self.E_K[:,:,None]-self.E_K[:,None,:]
            select=abs(dEig)<dEig_threshold
            dEig[select]=dEig_threshold
            _delHH_K_[select]=0
            self._delHH_dE_K=-1j*_delHH_K_/dEig[:,:,:,None]
            return self._delHH_dE_K


    def get_delHH_dE_SQ_K(self):
        try:
            return self._delHH_dE_SQ_K
        except AttributeError:
            self._delHH_dE_SQ_K= (self.delHH_dE_K[:,:,:,wham.beta]
                         *self.delHH_dE_K[:,:,:,wham.alpha].transpose((0,2,1,3))).imag
            return self._delHH_dE_SQ_K

#        AHC2=-2*fac*np.array([sum( (delhh[n:,:m,b]*delhh[:m,n:,a].T).imag.sum() 
#           for delhh,n,m in zip(data.delHH_dE_K,seln,selm) ) for a,b in zip(alpha,beta)] )
#        if printJ: print ("J2 term:",AHC2)


            
    def get_delHH_dE_AA_K(self):
        try:
            return self._delHH_dE_AA_K
        except AttributeError:
            self._delHH_dE_AA_K=(  
               (self.delHH_dE_K[:,:,:,wham.beta]*self.AAUU_K.transpose((0,2,1,3))[:,:,:,wham.alpha]).imag+
               (self.delHH_dE_K.transpose((0,2,1,3))[:,:,:,wham.alpha]*self.AAUU_K[:,:,:,wham.beta]).imag  )
            return self._delHH_dE_AA_K
            

    AA_K=property(get_AA_K)
    AAUU_K=property(get_AAUU_K)
    OOmega_K=property(get_OOmega_K)
    OOmegaUU_K=property(get_OOmegaUU_K)
    UU_K=property(get_UU_K)
    UUC_K=property(get_UUC_K)
    HH_K=property(get_HH_K)
    delHH_K=property(get_delHH_K)
    delHH_dE_K=property(get_delHH_dE_K)
    delHH_dE_SQ_K=property(get_delHH_dE_SQ_K)
    delHH_dE_AA_K=property(get_delHH_dE_AA_K)
    E_K=property(get_E_K)
    delE_K=property(get_delE_K)
    UUU_K=property(get_UUU_K)


