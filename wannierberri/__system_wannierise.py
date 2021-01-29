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
#   some parts of this file are originate                    #
# from the translation of Wannier90 code                     #
#------------------------------------------------------------#

import numpy as np
from scipy.io import FortranFile 
import copy
import lazy_property
import functools
import multiprocessing 
from .__utility import str2bool, alpha_A, beta_A, iterate3dpm, real_recip_lattice,fourier_q_to_R
from termcolor import cprint 
from .__system_w90 import System_w90, ws_dist_map_gen
from .__w90_files import EIG,MMN,CheckPoint,SPN,UHU,SIU,SHU
from time import time
import pickle
from itertools import repeat

class System_Wannierise(System_w90):
    """
    System initialized from the Wannier functions consructed internally by WannierBerri 
    
    Parameters
    ----------
    aidata : :class:`~wannierberri.AbInitioData` 
        the data from AbInitio code, should be disentangled already.
    transl_inv : bool
        Use Eq.(31) of `Marzari&Vanderbilt PRB 56, 12847 (1997) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.56.12847>`_ for band-diagonal position matrix elements
    npar : int
        number of processes used in the constructor
    fft : str
        library used to perform the fast Fourier transform from **q** to **R**. ``fftw`` or ``numpy``. (practically does not affect performance, 
        anyway mostly time of the constructor is consumed by reading the input files)

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System` 
    """

    def __init__(self,aidata,
                    transl_inv=True,
                    fft='fftw',
                    npar=multiprocessing.cpu_count()  , 
                    **parameters
                    ):

        self.set_parameters(**parameters)
        self.seedname=aidata.seedname
        if not aidata.disentangled: 
            warning ("no disentanglement was performed on the abinitio data. Are you sure you know what you are doing???")

        self.real_lattice,self.recip_lattice=real_recip_lattice(aidata.real_lattice,aidata.recip_lattice)
        self.mp_grid=aidata.mp_grid
        self.iRvec,self.Ndegen=self.wigner_seitz()
        self.nRvec0=len(self.iRvec)
        self.num_wann=aidata.NW

        if  self.use_ws:
            print ("using ws_distance")
            #ws_map=ws_dist_map_gen(np.copy(self.iRvec),np.copy(chk.wannier_centres), np.copy(chk.mp_grid),np.copy(self.real_lattice),npar=npar)
            ws_map=ws_dist_map_gen(self.iRvec,aidata.wannier_centres, self.mp_grid,self.real_lattice, npar=npar)
        
        eig=EIG(seedname)
        if self.getAA or self.getBB:
            mmn=MMN(seedname,npar=npar)

        kpt_mp_grid=[tuple(k) for k in np.array( np.round(chk.kpt_latt*np.array(chk.mp_grid)[None,:]),dtype=int)%chk.mp_grid]
#        print ("kpoints:",kpt_mp_grid)
        
        fourier_q_to_R_loc=functools.partial(fourier_q_to_R, mp_grid=chk.mp_grid,kpt_mp_grid=kpt_mp_grid,iRvec=self.iRvec,ndegen=self.Ndegen,numthreads=npar,fft=fft)

        timeFFT=0
        HHq=chk.get_HH_q(eig)
        t0=time()
        self.HH_R=fourier_q_to_R_loc( HHq )
        timeFFT+=time()-t0

        if self.getAA:
            AAq=chk.get_AA_q(mmn,transl_inv=transl_inv)
            t0=time()
            self.AA_R=fourier_q_to_R_loc(AAq)
            timeFFT+=time()-t0

        if self.getBB:
            t0=time()
            self.BB_R=fourier_q_to_R_loc(chk.get_AA_q(mmn,eig))
            timeFFT+=time()-t0

        if self.getCC:
            uhu=UHU(seedname)
            t0=time()
            self.CC_R=fourier_q_to_R_loc(chk.get_CC_q(uhu,mmn))
            timeFFT+=time()-t0
            del uhu

        if self.getSS:
            spn=SPN(seedname)
            t0=time()
            self.SS_R=fourier_q_to_R_loc(chk.get_SS_q(spn))
            if self.getSHC:
                self.SR_R=fourier_q_to_R_loc(chk.get_SR_q(spn,mmn))
                self.SH_R=fourier_q_to_R_loc(chk.get_SH_q(spn,eig))
                self.SHR_R=fourier_q_to_R_loc(chk.get_SHR_q(spn,mmn,eig))
            timeFFT+=time()-t0
            del spn

        if self.getSA:
            siu=SIU(seedname)
            t0=time()
            self.SA_R=fourier_q_to_R_loc(chk.get_SA_q(siu,mmn))
            timeFFT+=time()-t0
            del siu

        if self.getSHA:
            shu=SHU(seedname)
            t0=time()
            self.SHA_R=fourier_q_to_R_loc(chk.get_SHA_q(shu,mmn))
            timeFFT+=time()-t0
            del shu

        print ("time for FFT_q_to_R : {} s".format(timeFFT))
        if  self.use_ws:
            for X in ['HH','AA','BB','CC','SS','FF','SA','SHA','SR','SH','SHR']:
                XR=X+'_R'
                if hasattr(self,XR) :
                    print ("using ws_dist for {}".format(XR))
                    vars(self)[XR]=ws_map(vars(self)[XR])
            self.iRvec=np.array(ws_map._iRvec_ordered,dtype=int)

        self.finalise_init()

    @property
    def NKFFT_recommended(self):
        return self.mp_grid

    def wigner_seitz(self,mp_grid):
        ws_search_size=np.array([1]*3)
        dist_dim=np.prod((ws_search_size+1)*2+1)
        origin=divmod((dist_dim+1),2)[0]-1
        real_metric=self.real_lattice.dot(self.real_lattice.T)
        mp_grid=np.array(mp_grid)
        irvec=[]
        ndegen=[]
        for n in iterate3dpm(mp_grid*ws_search_size):
            # Loop over the 125 points R. R=0 corresponds to i1=i2=i3=0,
            # or icnt=63  (62 starting from zero)
            dist=[]
            for i in iterate3dpm((1,1,1)+ws_search_size):
                ndiff=n-i*mp_grid
                dist.append(ndiff.dot(real_metric.dot(ndiff)))
            dist_min = np.min(dist)
            if  abs(dist[origin] - dist_min) < 1.e-7 :
                irvec.append(n)
                ndegen.append(np.sum( abs(dist - dist_min) < 1.e-7 ))
    
        return np.array(irvec),np.array(ndegen)


