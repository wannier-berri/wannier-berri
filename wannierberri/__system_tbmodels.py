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

from .__utility import str2bool, alpha_A, beta_A , fourier_q_to_R
from colorama import init
from termcolor import cprint 
from .__system import System


class System_TBmodels(System):

    def __init__(self,tbmodel=None,getAA=False,
                          frozen_max=-np.Inf,
                          random_gauge=False,
                          degen_thresh=-1 ,
                    ):
        self.seedname=[]
        self.frozen_max=frozen_max
        self.random_gauge=random_gauge
        self.degen_thresh=degen_thresh 
        real=tbmodel.uc
        self.dimr=real.shape[1]
        zeros_real=np.eye((3),dtype=float)
        zeros_real[:self.dimr,:self.dimr]=np.array(real)
        self.real_lattice=zeros_real
        
        recip=2*np.pi*np.linalg.inv(zeros_real).T
        self.recip_lattice=recip
        
        self.num_wann=tbmodel.size
        self.spinors=False
        
        
        Rvec=np.array([R[0] for R in tbmodel.hop.items()],dtype=int)
        uniques=np.unique(Rvec,axis=0).astype('int32')# extract the unique directions
        uniquesneg=np.array([-u for u in uniques])
        R_all=np.concatenate((uniques,uniquesneg),axis=0)
        R_all=np.unique(R_all,axis=0)
        self.iRvec = R_all
        nRvec=self.iRvec.shape[0]
        self.nRvec0=nRvec
        

        if self.dimr==2:
            column=np.zeros((nRvec),dtype='int32')
            array_R=np.column_stack((R_all,column))
            self.iRvec=array_R
        index0=np.argwhere(np.all(([0,0,0]-self.iRvec)==0, axis=1))
        self.HH_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0),dtype=complex)
        for hop in tbmodel.hop.items():
            R=np.array(hop[0],dtype=int)
            hops=np.array(hop[1]).reshape((self.num_wann,self.num_wann))
            iR=int(np.argwhere(np.all((R-R_all)==0, axis=1)))
            inR=int(np.argwhere(np.all((-R-R_all)==0, axis=1)))
            self.HH_R[:,:,iR]+=hops
            self.HH_R[:,:,inR]+=np.conjugate(hops)
        if getAA:
            
            self.AA_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0,3),dtype=complex)
            
            for i in range(self.num_wann):
                self.AA_R[i,i,index0,:self.dimr]=tbmodel.pos[i,:]
                
        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        cprint ("Reading the system from TBmodels model finished successfully",'green', attrs=['bold'])
        
    @lazy_property.LazyProperty
    def NKFFTmin(self):
        NKFFTmin=np.ones(3,dtype=int)
        for i in range(3):
            R=self.iRvec[:,i]
            if len(R[R>0])>0: 
                NKFFTmin[i]+=R.max()
            if len(R[R<0])>0: 
                NKFFTmin[i]-=R.min()
        return NKFFTmin
