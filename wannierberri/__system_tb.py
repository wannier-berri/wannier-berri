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

from .__utility import str2bool, alpha_A, beta_A ,real_recip_lattice
from colorama import init
from termcolor import cprint 
from .__system import System


class System_tb(System):
    """
    System initialized from the `*_tb.dat` file, which can be written either by  `Wannier90 <http://wannier.org>`_ code, 
    or composed by the user based on some tight-binding model. 
    See Wannier90 `code <https://github.com/wannier-developers/wannier90/blob/2f4aed6a35ab7e8b38dbe196aa4925ab3e9deb1b/src/hamiltonian.F90#L698-L799>`_
    for details of the format. 
    
    Parameters
    ----------
    tb_file : str
        name (and path) of file to be read
    getAA : bool
        if ``True`` the position matrix elements are read. Needed for quantities derived from Berry connection or Berry curvature. 
    frozen_max : float
        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary.
    degen_thresh : float
        threshold to consider bands as degenerate. Used in calculation of Fermi-surface integrals
    random_gauge : bool
        applies random unitary rotations to degenerate states. Needed only for testing, to make sure that gauge covariance is preserved
    ksep: int
        separate k-point into blocks with size ksep to save memory when summing internal bands matrix. Working on gyotropic_Korb and berry_dipole. 
    delta_fz:float
        size of smearing for B matrix with frozen window, from frozen_max-delta_fz to frozen_max. 
    """

    def __init__(self,tb_file="wannier90_tb.dat",getAA=False,
                          frozen_max=-np.Inf,
                          degen_thresh=-1 ,
                          random_gauge=False,
                          ksep=50,
                          delta_fz=0.1
                    ):
        self.seedname=tb_file.split("/")[-1].split("_")[0]
        f=open(tb_file,"r")
        l=f.readline()
        self.frozen_max=frozen_max
        self.random_gauge=random_gauge
        self.degen_thresh=degen_thresh 
        self.ksep=ksep
        self.delta_fz=delta_fz
        cprint ("reading TB file {0} ( {1} )".format(tb_file,l.strip()),'green', attrs=['bold'])
        real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.real_lattice,self.recip_lattice= real_recip_lattice(real_lattice=real_lattice)
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
        self.set_symmetry()

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        cprint ("Reading the system from {} finished successfully".format(tb_file),'green', attrs=['bold'])

        
