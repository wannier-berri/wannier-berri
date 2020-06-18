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


class System_PythTB(System):

    def __init__(self,ptb_model=None,getAA=False,
                          frozen_max=-np.Inf,
                          random_gauge=False,
                          degen_thresh=-1 ,
                    ):
        self.seedname=[]
        self.frozen_max=frozen_max
        self.random_gauge=random_gauge
        self.degen_thresh=degen_thresh 
        real=ptb_model._lat
        self.dimr=real.shape[1]
        zeros_real=np.eye((3),dtype=float)
        zeros_real[:self.dimr,:self.dimr]=np.array(real)
        self.real_lattice=zeros_real
        
        recip=2*np.pi*np.linalg.inv(zeros_real).T
        self.recip_lattice=recip
        
        self.num_wann=ptb_model._norb
        if ptb_model._nspin==1:
            self.spinors=False
        elif ptb_model._nspin==2:
            self.spinors=True
        else:
            raise Exception("\n\nWrong value of nspin!")
        
        Rvec=np.array([R[-1] for R in ptb_model._hoppings],dtype=int)
        Rvec = [tuple(row) for row in Rvec] 
        uniques=np.unique(Rvec,axis=0).astype('int32')# extract the unique directions
        uniquesneg=np.array([-u for u in uniques])
        R_all=np.concatenate((uniques,uniquesneg),axis=0)
        R_all=np.unique(R_all,axis=0)
        self.iRvec = R_all
        nRvec=self.iRvec.shape[0]
        self.nRvec0=nRvec
        
#        self.Ndegen=[]
#        while len(self.Ndegen)<nRvec:
#            self.Ndegen+=f.readline().split()
        if self.dimr==2:
            column=np.zeros((nRvec),dtype='int32')
            array_R=np.column_stack((R_all,column))
            self.iRvec=array_R
        index0=np.argwhere(np.all(([0,0,0]-self.iRvec)==0, axis=1))
        self.HH_R=np.zeros((ptb_model._norb,ptb_model._norb,self.nRvec0),dtype=complex)
        for nhop in ptb_model._hoppings:
            i=nhop[1]
            j=nhop[2]
            index=np.argwhere(np.all((nhop[-1]-self.iRvec[:,:self.dimr])==0, axis=1))

            self.HH_R[i,j,index]+=nhop[0]
            # Check if the negative vector is already included. if not, add the conjugate
            indexrep=np.argwhere(np.all((-nhop[-1]-uniques[:,:self.dimr])==0, axis=1))
            if indexrep==[]:
                indexneg=np.argwhere(np.all((-nhop[-1]-self.iRvec[:,:self.dimr])==0, axis=1))
                self.HH_R[j,i,indexneg]+=np.conjugate(nhop[0])
        # set onsite energies from the diagonal of HH_R at R=[000]
        for i in range(ptb_model._norb):
            self.HH_R[i,i,index0]=ptb_model._site_energies[i]
        
        
        if getAA:
            
            self.AA_R=np.zeros((ptb_model._norb,ptb_model._norb,self.nRvec0,3),dtype=complex)
            
            for i in range(ptb_model._norb):
                self.AA_R[i,i,index0,:self.dimr]=ptb_model._orb[i,:]
                
        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        cprint ("Reading the system from {} finished successfully",'green', attrs=['bold'])