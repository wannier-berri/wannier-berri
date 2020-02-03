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
from collections import Iterable
from . import __result as result
from .__berry import fac_ahc, fac_morb, calcImf_K, calcImfgh_K

def calcAHC(data,Efermi,degen_thresh=None):
    E_K=data.E_K
    degen_bands,E_K_av=__get_degen_bands(E_K,degen_thresh)
    AHC=np.zeros( (len(Efermi),3) )
    for ik in range(data.NKFFT_tot):
        imf=calcImf_K(data,degen_bands[ik],ik )
        for e,f in zip(E_K_av[ik],imf):
            sel= Efermi>e
            AHC[Efermi>e]+=f
    AHC*=fac_ahc/(data.NKFFT_tot*data.cell_volume)
    return result.EnergyResultAxialV(Efermi,AHC)


def calcMorb(data,Efermi,degen_thresh=None,mf=-2,mg=1,mh=1,J=3):
    E_K=data.E_K
    degen_bands,E_K_av=__get_degen_bands(E_K,degen_thresh)
    Morb=np.zeros( (len(Efermi),3) )
    for ik in range(data.NKFFT_tot):
        imf,img,imh=calcImfgh_K(data,degen,ik )
        for e,f,g,h in zip(E_K_av,imf,img,imh):
            sel= Efermi>e
            Morb[sel]+=(g*mg+h*mh)
            Morb[sel]+=mf*f[None,:]*Efermi[sel,None]
    Morb*=fac_morb/(data.NKFFT_tot)
    return result.EnergyResultAxialV(Efermi,Morb)
#    return __calcSmth_band(data,Efermi,(3,) ,function,   degen_thresh=degen_thresh)*fac_morb/(data.NKFFT_tot)

#def calcMorb_LC(data,Efermi,degen_thresh=None):
   
#def calcMorb_IC(data,Efermi,degen_thresh=None):


##  a general procedure to evaluate smth band-by-band in the Fermi sea
def __calcSmth_band(data,Efermi,shape,function, degen_thresh=None):
    if degen_thresh is None:
        degen_thresh=-1
    E_K=data.E_K
    degen_bands,E_K_av=__get_degen_bands(E_K,degen_thresh)

    if not(isinstance(Efermi, Iterable)): 
        Efermi=np.array([Efermi])

    RES=np.zeros( (len(Efermi),)+tuple(shape) )
    for ik in range(data.NKFFT_tot) :
        function(RES,Efermi,data,degen_bands[ik],E_K_av[ik],ik)
    return RES 


def __get_degen_bands(E_K,degen_thresh):
    A=[np.hstack( ([0],np.where(E[1:]-E[:1]>degen_thresh)[0]+1, [E.shape[0]]) ) for E in E_K ]
    deg= [[(ib1,ib2) for ib1,ib2 in zip(a,a[1:])] for a in A]
    Eav= [ np.array( [E[b1:b2].mean() for b1,b2 in deg  ]) for E,deg in zip(E_K,deg)]
    return deg,Eav


def __average_degen(X,degen):
    np.array( [X[b1:b2].mean(axis=0) for b1,b2 in degen  ]) 