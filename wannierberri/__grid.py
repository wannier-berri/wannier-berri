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

from collections import Iterable
import numpy as np
from  .__Kpoint import KpointBZ




class Grid():
    """ A class containing information about the k-grid. 

    Parameters
    -----------
    system : :class:`~wannierberri.__system.System` 
        which the calculations will be made
    length :  float
        (angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    NK : int  or list or numpy.array(3) 
        number of k-points along each directions 
    NKFFT : int 
        number of k-points in the FFT grid along each directions 
    NKdiv : int 
        number of k-points in the division (K-) grid along each directions 
    GammaCentered : bool 
        wether grid is Gamma-Centered or shifted
    minimalFFT : bool
        force the FFT grid to be equal to the minimal allowed for the system 

    Notes
    -----
    `NK`, `NKdiv`, `NKFFT`  may be given as size-3 integer arrays or lists. Also may be just numbers -- in that case the number of kppoints is the same in all directions

    the following conbinations of (NK,NKFFT,NKdiv,length) parameters may be used:

    - `length` (preferred)
    - `NK`
    - `NK`, `NKFFT`
    - `length`, `NKFFT`
    - `NKdiv`, `NKFFT`

    The others will be evaluated automatically. 

    """

    def __init__(self,system,length=None,NKdiv=None,NKFFT=None,NK=None,minimalFFT=True,GammaCentered=True):

        NKFFTmin=system.NKFFTmin 
        self.symgroup=system.symgroup
        self.GammaCentered=GammaCentered
        self.div,self.FFT=determineNK(NKdiv,NKFFT,NK,NKFFTmin,self.symgroup,minimalFFT=minimalFFT,length=length)

    @property
    def dense(self):
        return self.div*self.FFT

    def get_K_list(self):
        if self.GammaCentered :
            shift=(self.div%2-1)/(2*self.div)
        else :
            shift=np.zeros(3)
#        print ("shift={}".format(shift))
        print ("generating K_list")
        K_list=[KpointBZ(K=shift, NKFFT=self.FFT,symgroup=self.symgroup )]
        K_list+=K_list[0].divide(self.div)
        if not np.all( self.div%2==1):
            del K_list[0]
        return K_list



def one2three(nk):
    if nk is None:
        return None
    if isinstance(nk, Iterable):
        if len(nk)!=3 :
            raise RuntimeError("nk should be specified either as one  number or 3 numbers. found {}".format(nk))
        return np.array(nk)
    return np.array((nk,)*3)


def iterate_vector(v1,v2):
    return ((x,y,z) for x in range(v1[0],v2[0]) for y in range(v1[1],v2[1]) for z in range(v1[2],v2[2]) )


def autoNK(NK,NKFFTmin,symgroup,minimalFFT):
    # frist determine all symmetric sets between NKFFTmin and 2*NKFFTmin
    FFT_symmetric=np.array([fft for fft in iterate_vector(NKFFTmin,NKFFTmin*3) if symgroup.symmetric_grid(fft) ])
    NKFFTmin=FFT_symmetric[np.argmin(FFT_symmetric.prod(axis=1))]
    print ("Minimal symmetric FFT grid : ",NKFFTmin)
    if minimalFFT:
        FFT=NKFFTmin
    else:
        FFT_symmetric=np.array([fft for fft in iterate_vector(NKFFTmin,NKFFTmin*2) if symgroup.symmetric_grid(fft) ])
        NKdiv_tmp=np.array(np.round(NK[None,:]/FFT_symmetric),dtype=int)
        NKdiv_tmp[NKdiv_tmp<=0]=1
        NKchange=NKdiv_tmp*FFT_symmetric/NK[None,:]
        sel=(NKchange>1)
        NKchange[sel]=1./NKchange[sel]
        NKchange=NKchange.min(axis=1)
        FFT=FFT_symmetric[np.argmax(NKchange)]
    NKdiv=np.array(np.round(NK/FFT),dtype=int)
    NKdiv[NKdiv<=0]=1
    return NKdiv,FFT


def determineNK(NKdiv,NKFFT,NK,NKFFTmin,symgroup,minimalFFT=False,length=None):
    print ("determining grids from NK={} ({}), NKdiv={} ({}), NKFFT={} ({})".format(NK,type(NK),NKdiv,type(NKdiv),NKFFT,type(NKdiv)))
    NKdiv=one2three(NKdiv)
    NKFFT=one2three(NKFFT)
    NK=one2three(NK)

    if length is not None:
        if NK is None: 
            NK=np.array(np.round(length/(2*np.pi)*np.linalg.norm(symgroup.recip_lattice,axis=1)),dtype=int)
            print ("length={} was converted into NK={}".format(length,NK))
        else: 
            print ("WARNING : length is disregarded in presence of NK")

    for nkname in 'NKdiv','NK','NKFFT':
        nk=locals()[nkname]
        if nk is not None:
            assert symgroup.symmetric_grid(nk) , " {}={} is not consistent with the given symmetry ".format(nkname,nk)



    if (NKdiv is not None) and (NKFFT is not None):
        assert np.all(NKFFT>=NKFFTmin) , "the given FFT grid {} is smaller then minimal allowed {} for this system. Increase the FFT grid".format(NKFFT,NKFFTmin)
        if length is not None:
            print ("WARNING : length is disregarded in presence of NKdiv,NKFFT")
        elif NK is not None:
            print ("WARNING : NK is disregarded in presence of NKdiv,NKFFT")
    elif NK is not None:
        if NKdiv is not None:
            print ("WARNING : NKdiv is disregarded in presence of NK or length")
        if NKFFT is not None: 
            assert np.all(NKFFT>=NKFFTmin) , "the given FFT grid {} is smaller then minimal allowed {} for this system. Increase the FFT grid".format(NKFFT,NKFFTmin)
            NKdiv=np.array(np.round(NK/NKFFT),dtype=int)
            NKdiv[NKdiv<=0]=1
        else: 
            NKdiv,NKFFT=autoNK(NK,NKFFTmin,symgroup,minimalFFT)
    else : 
        raise ValueError("you need to specify either NK or a pair (NKdiv,NKFFT) or (NK,NKFFT) . found NK={}, NKdiv={}, NKFFT={} ".format(NK,NKdiv,NKFFT))
    if NK is not None:
        if not np.all(NK==NKFFT*NKdiv) :
            print ( "WARNING : the requested k-grid {} was adjusted to {}. Hope that it is fine".format(NK,NKFFT*NKdiv))
    return NKdiv,NKFFT
