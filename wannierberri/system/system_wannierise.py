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

import functools
import multiprocessing
from ..__utility import real_recip_lattice, fourier_q_to_R
from .system_w90 import System_w90
from time import time

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
                    npar=multiprocessing.cpu_count(),
                    **parameters
                    ):

        self.set_parameters(**parameters)
        self.npar = npar
        self.seedname=aidata.seedname
        aidata.check_disentangled()
        self.real_lattice,self.recip_lattice=real_recip_lattice(aidata.chk.real_lattice,aidata.chk.recip_lattice)
        self.mp_grid=aidata.chk.mp_grid
        self.wannier_centers_cart_auto = aidata.wannier_centres
        self.num_wann=aidata.chk.num_wann
        self.iRvec,self.Ndegen=self.wigner_seitz(aidata.chk.mp_grid)
        self.nRvec0=len(self.iRvec)

        eig=aidata.eig
        if self.need_R_any(['AA','BB']):
            mmn=aidata.mmn

        print(f"Need A:{self.need_R_any('AA')}, berry:{self.berry}")

        fourier_q_to_R_loc=functools.partial(fourier_q_to_R,
                                             mp_grid=aidata.chk.mp_grid,
                                             kpt_mp_grid=aidata.kpt_mp_grid,
                                             iRvec=self.iRvec,
                                             ndegen=self.Ndegen,
                                             numthreads=npar,
                                             fft=fft)

        timeFFT=0
        HHq=aidata.chk.get_HH_q(eig)
        t0=time()
        self.set_R_mat('Ham', fourier_q_to_R_loc(HHq))
        timeFFT+=time()-t0

        if self.need_R_any('AA'):
            AAq=aidata.chk.get_AA_q(mmn,transl_inv=transl_inv)
            t0=time()
            self.set_R_mat('AA',fourier_q_to_R_loc(AAq))
            timeFFT+=time()-t0

        if self.need_R_any('BB'):
            t0=time()
            self.BB_R=fourier_q_to_R_loc(aidata.chk.get_AA_q(mmn,eig))
            timeFFT+=time()-t0

        if self.need_R_any('CC'):
            uhu=aidata.uhu
            t0=time()
            self.CC_R=fourier_q_to_R_loc(aidata.chk.get_CC_q(uhu,mmn))
            timeFFT+=time()-t0
            del uhu

        if self.need_R_any(['SS', 'SR', 'SH', 'SHR']):
            spn=aidata.spn # TODO : access through aidata, not directly
            t0=time()
            if self.need_R_any('SS'):
                self.SS_R=fourier_q_to_R_loc(aidata.get_SS_q(spn))
            if self.need_R_any('SR'):
                self.SR_R=fourier_q_to_R_loc(aidata.get_SR_q(spn,mmn))
            if self.need_R_any('SH'):
                self.SH_R=fourier_q_to_R_loc(aidata.get_SH_q(spn,eig))
            if self.need_R_any('SHR'):
                self.SHR_R=fourier_q_to_R_loc(aidata.get_SHR_q(spn,mmn,eig))
            timeFFT+=time()-t0
            del spn

        if self.need_R_any('SA'):
            siu=aidata.siu # TODO : access through aidata, not directly
            t0=time()
            self.SA_R=fourier_q_to_R_loc(aidata.get_SA_q(siu,mmn))
            timeFFT+=time()-t0
            del siu

        if self.need_R_any('SHA'):
            shu=aidata.shu # TODO : access through aidata, not directly
            t0=time()
            self.SHA_R=fourier_q_to_R_loc(aidata.get_SHA_q(shu,mmn))
            timeFFT+=time()-t0
            del shu

        print ("time for FFT_q_to_R : {} s".format(timeFFT))
        self.do_at_end_of_init()
