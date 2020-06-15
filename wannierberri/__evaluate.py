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

import multiprocessing 
import functools
import numpy as np
from collections import Iterable
import lazy_property
from copy import copy
from time import time
import pickle
import glob

from .__Data_K import Data_K
from . import __symmetry as SYM
from  .__Kpoint import KpointBZ,exclude_equiv_points
from . import __utility as utility
from .__utility import MSG_not_symmetric
   

def process(paralfunc,K_list,nproc,symgroup=None):
    t0=time()
    selK=[ik for ik,k in enumerate(K_list) if k.res is None]
    dK_list=[K_list[ik].Kp_fullBZ for ik in selK]
    if len(dK_list)==0:
        print ("nothing to process now")
        return 0
    print ("processing {0}  points :".format(len(dK_list)) )
    if nproc<=0:
        res = [paralfunc(k) for k in dK_list]
        nproc_=1
    else:
        p=multiprocessing.Pool(nproc)
        res= p.map(paralfunc,dK_list)
        p.close()
        nproc_=nproc
    if not (symgroup is None):
        res=[symgroup.symmetrize(r) for r in res]
    for i,ik in enumerate(selK):
        K_list[ik].set_res(res[i])
    t=time()-t0
    print ("time for processing {0:6d} K-points : {1:10.4f} ; per K-point {2:15.4f} ; proc-sec per K-point : {3:15.4f}".format(len(selK),t,t/len(selK),t*nproc_/len(selK)) )
    return len(dK_list)
        



def one2three(nk):
    if nk is None:
        return None
    if isinstance(nk, Iterable):
        if len(nk)!=3 :
            raise RuntimeError("nk should be specified either as one  number or 3 numbers. found {}".format(nk))
        return np.array(nk)
    return np.array((nk,)*3)


def iterate_vector(v1,v2):
    return ((x,y,z) for x in range(v1[0],v2[0]) for y in range(v1[1],v2[2]) for z in range(v1[2],v2[2]) )


def autoNK(NK,NKFFTmin,symgroup,minimalFFT):
    # frist determine all symmetric sets between NKFFTmin and 2*NKFFTmin
    FFT_symmetric=np.array([fft for fft in iterate_vector(NKFFTmin,NKFFTmin*3) if symgroup.symmetric_grid(fft) ])
    NKFFTmin=FFT_symmetric[np.argmin(FFT_symmetric.prod(axis=1))]
    print ("Minimal symmetric FFT grid : ",NKFFTmin)
    if minimalFFT:
        return NKFFTmin
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


def determineNK(NKdiv,NKFFT,NK,NKFFTmin,symgroup,minimalFFT=False):
    print ("determining grids from NK={} ({}), NKdiv={} ({}), NKFFT={} ({})".format(NK,type(NK),NKdiv,type(NKdiv),NKFFT,type(NKdiv)))
    NKdiv=one2three(NKdiv)
    NKFFT=one2three(NKFFT)
    NK=one2three(NK)
    for nkname in 'NKdiv','NK','NKFFT':
        nk=locals()[nkname]
        if nk is not None:
            assert symgroup.symmetric_grid(nk) , " {}={} is not consistent with the given symmetry ".format(nkname,nk)

    if (NKdiv is not None) and (NKFFT is not None):
        assert np.all(NKFFT>=NKFFTmin) , "the given FFT grid {} is smaller then minimal allowed {} for this system. Increase the FFT grid".format(NKFFT,NKFFTmin)
        if NK is not None:
            print ("WARNING : NK is disregarded in presence of NKdiv,NKFFT")
        pass
    elif NK is not None:
        if NKdiv is not None:
            print ("WARNING : NKdiv is disregarded in presence of NK")
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



def evaluate_K(func,system,NK=None,NKdiv=None,nproc=0,NKFFT=None,minimalFFT=False,
            adpt_mesh=2,adpt_num_iter=0,adpt_nk=1,fout_name="result",
             symmetry_gen=[SYM.Identity],suffix="",
             GammaCentered=True,file_Klist="K_list.pickle",restart=False,start_iter=0):
    """This function evaluates in parallel or serial an integral over the Brillouin zone 
of a function func, which whould receive only one argument of type Data_K, and return 
a numpy.array of whatever dimensions

the user has to provide 2 grids:  of K-points - NKdiv and FFT grid (k-points) NKFFT

The parallelisation is done by K-points

As a result, the integration will be performed over NKFFT x NKdiv
"""
    
    if file_Klist is not None:
        if not file_Klist.endswith(".pickle"):
            file_Klist+=".pickle"
    cnt_exclude=0
    
    symgroup=SYM.Group(symmetry_gen,recip_lattice=system.recip_lattice)
    assert symgroup.check_basis_symmetry(system.real_lattice)  , "the real basis is not symmetric"+MSG_not_symmetric
    assert symgroup.check_basis_symmetry(system.recip_lattice) , "the reciprocal basis is not symmetric"+MSG_not_symmetric

    NKdiv,NKFFT=determineNK(NKdiv,NKFFT,NK,system.NKFFTmin,symgroup,minimalFFT=minimalFFT)

    print ("using NKdiv={}, NKFFT={}".format( NKdiv,NKFFT))
    print ("using NKdiv={}, NKFFT={}, NKtot={}".format( NKdiv,NKFFT,NKdiv*NKFFT))
    
    

    paralfunc=functools.partial(
        _eval_func_k, func=func,system=system,NKFFT=NKFFT )

    if GammaCentered :
        shift=(NKdiv%2-1)/(2*NKdiv)
    else :
        shift=np.zeros(3)
    print ("shift={}".format(shift))

    if restart:
        try:
            K_list=pickle.load(open(file_Klist,"rb"))
            print ("{0} K-points were read from {1}".format(len(K_list),file_Klist))
            if len(K_list)==0:
                print ("WARNING : {0} contains zero points starting from scrath".format(file_Klist))
                restart=False
        except Exception as err:
            restart=False
            print ("WARNING: {}".format( err) )
            print ("WARNING : reading from {0} failed, starting from scrath".format(file_Klist))
            
    if not restart:
        print ("generating K_list")
        K_list=[KpointBZ(K=shift, NKFFT=NKFFT,symgroup=symgroup )]
        K_list+=K_list[0].divide(NKdiv)
        if not np.all( NKdiv%2==1):
            del K_list[0]
        print ("Done, sum of weights:{}".format(sum(Kp.factor for Kp in K_list)))
        start_iter=0

    suffix="-"+suffix if len(suffix)>0 else ""

    if restart:
        print ("searching for start_iter")
        try:
            start_iter=int(sorted(glob.glob(fout_name+"*"+suffix+"_iter-*.dat"))[-1].split("-")[-1].split(".")[0])
        except Exception as err:
            print ("WARNING : {0} : failed to read start_iter. Setting to zero".format(err))
            start_iter=0

    if adpt_num_iter<0:
        adpt_num_iter=-adpt_num_iter*np.prod(NKdiv)/np.prod(adpt_mesh)/adpt_nk/3
    adpt_num_iter=int(round(adpt_num_iter))


    if (adpt_mesh is None) or np.max(adpt_mesh)<=1:
        adpt_num_iter=0
    else:
        if not isinstance(adpt_mesh, Iterable):
            adpt_mesh=[adpt_mesh]*3
        adpt_mesh=np.array(adpt_mesh)
    
    counter=0


    for i_iter in range(adpt_num_iter+1):
        print ("iteration {0} - {1} points. New points are:".format(i_iter,len([K for K in  K_list if K.res is None])) ) 
        for i,K in enumerate(K_list):
          if not K.evaluated:
            print (" K-point {0} : {1} ".format(i,K))
        counter+=process(paralfunc,K_list,nproc,symgroup=symgroup)
        
        try:
            if file_Klist is not None:
                pickle.dump(K_list,open(file_Klist,"wb"))
        except Exception as err:
            print ("Warning: {0} \n the K_list was not pickled".format(err))
            
        result_all=sum(kp.get_res for kp in K_list)
        
        if not (restart and i_iter==0):
            result_all.write(fout_name+"-{}"+suffix+"_iter-{0:04d}.dat".format(i_iter+start_iter))
        
        if i_iter >= adpt_num_iter:
            break
             
        # Now add some more points
        Kmax=np.array([K.max for K in K_list]).T
        select_points=set().union( *( np.argsort( Km )[-adpt_nk:] for Km in Kmax )  )
        
        l1=len(K_list)
        for iK in select_points:
            K_list+=K_list[iK].divide(adpt_mesh)
        print ("checking for equivalent points in all points (of new  {} points)".format(len(K_list)-l1))
        nexcl=exclude_equiv_points(K_list,new_points=len(K_list)-l1)
        print (" excluded {0} points".format(nexcl))
        print ("sum of weights now :{}".format(sum(Kp.factor for Kp in K_list)))
        
    
    print ("Totally processed {0} K-points ".format(counter))
    return result_all
       


def _eval_func_k(K,func,system,NKFFT):
    data=Data_K(system,K,NKFFT=NKFFT)
    return func(data)

