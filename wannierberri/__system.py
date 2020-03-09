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

import numpy as np
from scipy.io import FortranFile as FF
import copy
import lazy_property

from .__utility import str2bool, alpha_A, beta_A
from colorama import init
from termcolor import cprint 



class System():

    def __init__(self,seedname="wannier90",tb_file=None,
                    getAA=False,
                    getBB=False,getCC=False,
                    getSS=False,
                    getFF=False,
                    use_ws=True,
                    frozen_max=-np.Inf,
                    random_gauge=False,
                    degen_thresh=-1
                                ):


        self.frozen_max=frozen_max
        self.random_gauge=random_gauge
        self.degen_thresh=degen_thresh

        if tb_file is not None:
            self.__from_tb_file(tb_file,getAA=getAA)
            return
        cprint ("Reading from {}".format(seedname+"_HH_save.info"),'green', attrs=['bold'])

        f=open(seedname+"_HH_save.info","r")
        l=f.readline().split()[:3]
        self.seedname=seedname
        self.num_wann,nRvec,self.spinors=int(l[0]),int(l[1]),str2bool(l[2])
        self.nRvec0=nRvec
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice).T
        iRvec=np.array([f.readline().split()[:4] for i in range(nRvec)],dtype=int)
        
        self.Ndegen=iRvec[:,3]
        self.iRvec=iRvec[:,:3]
        self.NKFFTmin=np.abs(self.iRvec).max(axis=0)*2

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        #print ("R - points and dege=neracies:\n",iRvec)
        has_ws=str2bool(f.readline().split("=")[1].strip())
        
        if has_ws and use_ws:
            print ("using ws_dist")
            self.ws_map=ws_dist_map(self.iRvec,self.num_wann,f.readlines())
            self.iRvec=np.array(self.ws_map._iRvec_ordered,dtype=int)
        else:
            self.ws_map=None
        
        f.close()
        if getCC:
           getBB=True

        self.HH_R=self.__getMat('HH')
        
        if getAA:
            self.AA_R=self.__getMat('AA')

        if getBB:
            self.BB_R=self.__getMat('BB')
        
        if getCC:
            try:
                self.CC_R=1j*self.__getMat('CCab')
            except:
                _CC_R=self.__getMat('CC')
                self.CC_R=1j*(_CC_R[:,:,:,alpha_A,beta_A]-_CC_R[:,:,:,beta_A,alpha_A])

        if getFF:
            try:
                self.FF_R=1j*self.__getMat('FFab')
            except:
                _FF_R=self.__getMat('FF')
                self.FF_R=1j*(_FF_R[:,:,:,alpha_A,beta_A]-_FF_R[:,:,:,beta_A,alpha_A])

        if getSS:
            self.SS_R=self.__getMat('SS')
        cprint ("Reading the system finished successfully",'green', attrs=['bold'])
        

    def __from_tb_file(self,tb_file=None,getAA=False):
        self.seedname=tb_file.split("/")[-1].split("_")[0]
        f=open(tb_file,"r")
        l=f.readline()
        cprint ("reading TB file {0} ( {1} )".format(tb_file,l.strip()),'green', attrs=['bold'])
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice).T
        self.num_wann=int(f.readline())
        nRvec=int(f.readline())
        self.nRvec0=nRvec
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
        
        if getAA:
          self.AA_R=np.zeros( (self.num_wann,self.num_wann,nRvec,3) ,dtype=complex)
          for ir in range(nRvec):
            f.readline()
            assert (np.array(f.readline().split(),dtype=int)==self.iRvec[ir]).all()
            aa=np.array( [[f.readline().split()[2:8] 
                             for n in range(self.num_wann)] 
                                for m in range(self.num_wann)],dtype=float)
            self.AA_R[:,:,ir,:]=(aa[:,:,0::2]+1j*aa[:,:,1::2]).transpose( (1,0,2) ) /self.Ndegen[ir]
        
        
        f.close()

        self.NKFFTmin=np.abs(self.iRvec).max(axis=0)*2

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        cprint ("Reading the system finished successfully",'green', attrs=['bold'])

        #print ("R - points and dege=neracies:\n",iRvec)
        
    @property
    def cRvec(self):
        return self.iRvec.dot(self.real_lattice)


    @property 
    def nRvec(self):
        return self.iRvec.shape[0]


    @lazy_property.LazyProperty
    def cell_volume(self):
        return abs(np.linalg.det(self.real_lattice))


    def __getMat(self,suffix):

        f=FF(self.seedname+"_" + suffix+"_R.dat")
        MM_R=np.array([[np.array(f.read_record('2f8'),dtype=float) for m in range(self.num_wann)] for n in range(self.num_wann)])
        MM_R=MM_R[:,:,:,0]+1j*MM_R[:,:,:,1]
        f.close()
        ncomp=MM_R.shape[2]/self.nRvec0
        if ncomp==1:
            result=MM_R/self.Ndegen[None,None,:]
        elif ncomp==3:
            result= MM_R.reshape(self.num_wann, self.num_wann, 3, self.nRvec0).transpose(0,1,3,2)/self.Ndegen[None,None,:,None]
        elif ncomp==9:
            result= MM_R.reshape(self.num_wann, self.num_wann, 3,3, self.nRvec0).transpose(0,1,4,3,2)/self.Ndegen[None,None,:,None,None]
        else:
            raise RuntimeError("in __getMat: invalid ncomp : {0}".format(ncomp))
        if self.ws_map is None:
            return result
        else:
            return self.ws_map(result)
        

#
# the following  implements the use_ws_distance = True  (see Wannier90 documentation for details)
#



class map_1R():
   def __init__(self,lines,irvec):
       lines_split=[np.array(l.split(),dtype=int) for l in lines]
       self.dict={(l[0]-1,l[1]-1):l[2:].reshape(-1,3) for l in lines_split}
       self.irvec=np.array([irvec])
       
   def __call__(self,i,j):
       try :
           return self.dict[(i,j)]
       except KeyError:
           return self.irvec
          

class ws_dist_map():
    def __init__(self,iRvec,num_wann,lines):
        nRvec=iRvec.shape[0]
        self.num_wann=num_wann
        self._iRvec_new=dict()
        n_nonzero=np.array([l.split()[-1] for l in lines[:nRvec]],dtype=int)
        lines=lines[nRvec:]
        nonzero=[]
        for ir,nnz in enumerate(n_nonzero):
            map1r=map_1R(lines[:nnz],iRvec[ir])
            for iw in range(num_wann):
                for jw in range(num_wann):
                    self._add_star(ir,map1r(iw,jw),iw,jw)
            lines=lines[nnz:]
        self._iRvec_ordered=sorted(self._iRvec_new)
        for ir  in range(nRvec):
            chsum=0
            for irnew in self._iRvec_new:
                if ir in self._iRvec_new[irnew]:
                    chsum+=self._iRvec_new[irnew][ir]
            chsum=np.abs(chsum-np.ones( (num_wann,num_wann) )).sum() 
            if chsum>1e-12: print ("WARNING: Check sum for {0} : {1}".format(ir,chsum))


    def _add_star(self,ir,irvec_new,iw,jw):
        weight=1./irvec_new.shape[0]
        for irv in irvec_new:
            self._add(ir,irv,iw,jw,weight)


    def _add(self,ir,irvec_new,iw,jw,weight):
        irvec_new=tuple(irvec_new)
        if not (irvec_new in self._iRvec_new):
             self._iRvec_new[irvec_new]=dict()
        if not ir in self._iRvec_new[irvec_new]:
             self._iRvec_new[irvec_new][ir]=np.zeros((self.num_wann,self.num_wann),dtype=float)
        self._iRvec_new[irvec_new][ir][iw,jw]+=weight
        
    def __call__(self,matrix):
        ndim=len(matrix.shape)-3
        num_wann=matrix.shape[0]
        reshaper=(num_wann,num_wann)+(1,)*ndim
        print ("check:",matrix.shape,reshaper,ndim)
        matrix_new=np.array([ sum(matrix[:,:,ir]*self._iRvec_new[irvecnew][ir].reshape(reshaper)
                                  for ir in self._iRvec_new[irvecnew] ) 
                                       for irvecnew in self._iRvec_ordered]).transpose( (1,2,0)+tuple(range(3,3+ndim)) )

        assert ( np.abs(matrix_new.sum(axis=2)-matrix.sum(axis=2)).max()<1e-12)
        return matrix_new
             


