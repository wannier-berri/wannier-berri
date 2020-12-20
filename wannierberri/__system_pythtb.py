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


class System_PythTB(System):
    """This interface is an way to initialize the System class from a tight-binding 
    model created with  `PythTB. <http://www.physics.rutgers.edu/pythtb/>`_ 
    It defines the Hamiltonian matrix HH_R (from hoppings matrix elements)
    and the AA_R  matrix (from orbital coordinates) used to calculate 
    Berry related quantities.

    Parameters
    ----------
    ptb_model : class
        name of the PythTB tight-binding model class.


    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System` 
    """
    def __init__(self,ptb_model , **parameters):
        
        self.set_parameters(**parameters)
        self.seedname='model_PythTB'

        # Extract the parameters from the model
        real=ptb_model._lat
        self.dimr=real.shape[1]
        self.periodic[:self.dimr]=True
        self.periodic[self.dimr:]=False
        zeros_real=np.eye((3),dtype=float)
        zeros_real[:self.dimr,:self.dimr]=np.array(real)
        self.real_lattice=zeros_real
        recip=2*np.pi*np.linalg.inv(zeros_real).T
        self.recip_lattice=recip
        
        self.num_wann=ptb_model._norb
# TODO: adapt the interface to spinful models
        if ptb_model._nspin==1:
            self.spinors=False
        elif ptb_model._nspin==2:
            self.spinors=True
        else:
            raise Exception("\n\nWrong value of nspin!")
        
        Rvec=np.array([R[-1] for R in ptb_model._hoppings],dtype=int)
        Rvec = [tuple(row) for row in Rvec] 
        Rvecs=np.unique(Rvec,axis=0).astype('int32')
        
        nR=Rvecs.shape[0]
        # If the model is two-dimensional, add a column of zeros to iRvec
        if self.dimr==2:
            column=np.zeros((nR),dtype='int32')
            Rvecs=np.column_stack((Rvecs,column))
        
        Rvecsneg=np.array([-r for r in Rvecs])
        R_all=np.concatenate((Rvecs,Rvecsneg),axis=0)
        R_all=np.unique(R_all,axis=0)
        
        # Find the R=[000] index (used later)
        index0=np.argwhere(np.all(([0,0,0]-R_all)==0, axis=1))
        # if there are no intracell hoppings, R=[000] is not present in R_all
        # add it manually
        if index0.size==0:
            R_all=np.column_stack((np.array([0,0,0]),R_all.T)).T
            index0=0
        
        self.iRvec = R_all
        nRvec=self.iRvec.shape[0]
        self.nRvec0=nRvec
        # Define HH_R matrix from hoppings
        self.HH_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0),dtype=complex)
        for nhop in ptb_model._hoppings:
            i=nhop[1]
            j=nhop[2]
            iR=np.argwhere(np.all((nhop[-1]-self.iRvec[:,:self.dimr])==0, axis=1))
            inR=np.argwhere(np.all((-nhop[-1]-self.iRvec[:,:self.dimr])==0, axis=1))

            self.HH_R[i,j,iR]+=nhop[0]
            self.HH_R[j,i,inR]+=np.conjugate(nhop[0])
        # Set the onsite energies at H(R=[000])
        for i in range(ptb_model._norb):
            self.HH_R[i,i,index0]=ptb_model._site_energies[i]
        
        if self.getAA:
            self.AA_R=np.zeros((self.num_wann,self.num_wann,self.nRvec0,3),dtype=complex)
            for i in range(self.num_wann):
                self.AA_R[i,i,index0,:]=ptb_model._orb[i,:].dot(self.real_lattice[:self.dimr])
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
        cprint ("Reading the system from PythTB finished successfully",'green', attrs=['bold'])
