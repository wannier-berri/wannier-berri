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

    def __init__(self,
                 aidata,
                 transl_inv=True,
                 guiding_centers=False,
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



        print(f"Need A:{self.need_R_any('AA')}, berry:{self.berry}")

        fourier_q_to_R_loc=functools.partial(fourier_q_to_R,
                                             mp_grid=aidata.chk.mp_grid,
                                             kpt_mp_grid=aidata.kpt_mp_grid,
                                             iRvec=self.iRvec,
                                             ndegen=self.Ndegen,
                                             numthreads=npar,
                                             fft=fft)

        self.set_R_matrices_from_chk(chk=aidata.chk, mmn=aidata.mmn, eig=aidata.eig, seedname=aidata.seedname,
                                     fourier_q_to_R_loc=fourier_q_to_R_loc,
                                    transl_inv=transl_inv, guiding_centers=guiding_centers)
        self.do_at_end_of_init()
