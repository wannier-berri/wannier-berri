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


class System_tb(System):

    def __init__(self,tb_file="wannier90_tb.dat",getAA=False,
                          frozen_max=-np.Inf,
                          random_gauge=False,
                          degen_thresh=-1 ,
                    ):
        self.seedname=tb_file.split("/")[-1].split("_")[0]
        f=open(tb_file,"r")
        l=f.readline()
        self.frozen_max=frozen_max
        self.random_gauge=random_gauge
        self.degen_thresh=degen_thresh 
        cprint ("reading TB file {0} ( {1} )".format(tb_file,l.strip()),'green', attrs=['bold'])
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice).T
        self.num_wann=int(f.readline())
        nRvec=int(f.readline())
        self.nRvec0=nRvec
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
        else: 
            self.AA_R = None
        
        f.close()

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        cprint ("Reading the system from {} finished successfully".format(tb_file),'green', attrs=['bold'])

        
