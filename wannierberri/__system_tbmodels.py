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
from termcolor import cprint 
from .__system import System


class System_TBmodels(System):
    """This interface initializes the System class from a tight-binding 
    model created with `TBmodels. <http://z2pack.ethz.ch/tbmodels/doc/1.3/index.html>`_
    It defines the Hamiltonian matrix HH_R (from hoppings matrix elements)
    and the AA_R  matrix (from orbital coordinates) used to calculate Berry
    related quantities.
    
    
    Parameters
    ----------
    tbmodel : class
        name of the TBmodels tight-binding model class.

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System` 
    """
    
    def __init__(self,tbmodel,**parameters ):
        self.set_parameters(**parameters)
        self.seedname='model_TBmodels'
        if self.spin : raise ValueError("System_TBmodels class cannot be used for evaluation of spin properties")

        # Extract the parameters from the model
        real=tbmodel.uc
        self.dimr=real.shape[1]
        zeros_real=np.eye((3),dtype=float)
        self.periodic[:self.dimr]=True
        self.periodic[self.dimr:]=False
        zeros_real[:self.dimr,:self.dimr]=np.array(real)
        self.real_lattice=zeros_real
        recip=2*np.pi*np.linalg.inv(zeros_real).T
        self.recip_lattice=recip
        
        self.num_wann=tbmodel.size
        self.spinors=False
        
        Rvec=np.array([R[0] for R in tbmodel.hop.items()],dtype=int)
        Rvec = [tuple(row) for row in Rvec] 
        Rvecs=np.unique(Rvec,axis=0).astype('int32')   
        
        nR=Rvecs.shape[0]
        if self.dimr==2:
            column=np.zeros((nR),dtype='int32')
            Rvecs=np.column_stack((Rvecs,column))
            
        Rvecsneg=np.array([-r for r in Rvecs])
        R_all=np.concatenate((Rvecs,Rvecsneg),axis=0)
        R_all=np.unique(R_all,axis=0)
        
        # Find the R=[000] index (used later)
        index0=np.argwhere(np.all(([0,0,0]-R_all)==0, axis=1))
        # make sure it exists; otherwise, add it manually
        # add it manually
        if index0.size==0:
            R_all=np.column_stack((np.array([0,0,0]),R_all.T)).T
            index0=0
        
        self.iRvec = R_all
        nRvec=self.iRvec.shape[0]
        self.nRvec0=nRvec
        # Define HH_R matrix from hoppings
        self.HH_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0),dtype=complex)
        for hop in tbmodel.hop.items():
            R=np.array(hop[0],dtype=int)
            hops=np.array(hop[1]).reshape((self.num_wann,self.num_wann))
            iR=int(np.argwhere(np.all((R-R_all[:,:self.dimr])==0, axis=1)))
            inR=int(np.argwhere(np.all((-R-R_all[:,:self.dimr])==0, axis=1)))
            self.HH_R[:,:,iR]+=hops
            self.HH_R[:,:,inR]+=np.conjugate(hops.T)
        
        if self.getAA:
            self.AA_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0,3),dtype=complex)
            for i in range(self.num_wann):
                self.AA_R[i,i,index0,:]=tbmodel.pos[i,:].dot(self.real_lattice[:tbmodel.dim])

        if self.getBB:
            self.BB_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0,3),dtype=complex)
            for i in range(self.num_wann):
                self.BB_R[i,i,index0,:]=self.AA_R[i,i,index0,:]*self.HH_R[i,i,index0]

        if self.getCC:
            self.CC_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0,3),dtype=complex)

        self.set_symmetry()
        self.check_periodic()
                
        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Reommended size of FFT grid", self.NKFFT_recommended)
        print ("Real-space lattice:\n",self.real_lattice)
        cprint ("Reading the system from TBmodels finished successfully",'green', attrs=['bold'])
        
