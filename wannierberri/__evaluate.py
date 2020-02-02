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

from .__data_dk import Data_dk
from . import __symmetry as SYM
from  .__kpoint import KpointBZ,exclude_equiv_points
from . import __utility as utility
   

def process(paralfunc,k_list,nproc,symgroup=None):
    t0=time()
    selK=[ik for ik,k in enumerate(k_list) if k.res is None]
    dk_list=[k_list[ik].kp_fullBZ for ik in selK]
    if len(dk_list)==0:
        print ("nothing to process now")
        return 0
    print ("processing {0}  points :".format(len(dk_list)) )
    if nproc<=0:
        res = [paralfunc(k) for k in dk_list]
        nproc_=1
    else:
        p=multiprocessing.Pool(nproc)
        res= p.map(paralfunc,dk_list)
        p.close()
        nproc_=nproc
    if not (symgroup is None):
        res=[symgroup.symmetrize(r) for r in res]
    for i,ik in enumerate(selK):
        k_list[ik].set_res(res[i])
    t=time()-t0
    print ("time for processing {0} k-points : {1} ; per point {2} ; proc-sec per point : {3}".format(len(selK),t,t/len(selK),t*nproc_/len(selK)) )
    return len(dk_list)
        



def one2three(nk):
    if isinstance(nk, Iterable):
        if len(nk)!=3 :
            raise RuntimeError("nk should be specified either a on number or 3numbers. found {}".format(nk))
        return nk
    return (nk,)*3



def autonk(nk,nkfftmin):
    if nk<nkfftmin:
        return 1,nkfftmin
    else:
        lst=[]
        for i in range(nkfftmin,nkfftmin*2):
            if nk%i==0:
                return nk//i,i
            j=nk//i
            lst.append( min( abs( j*i-nk),abs(j*i+i-nk)))
    i=nkfftmin+np.argmin(lst)
    j=nk//i
    j=[j,j+1][np.argmin([abs( j*i-nk),abs(j*i+i-nk)])]
    return j,i
#    return int(round(nk/nkfftmin)),nkfftmin)


def determineNK(NKdiv,NKFFT,NK,NKFFTmin):
    if ((NKdiv is None) or (NKFFT is None)) and ((NK is None) or (NKFFTmin is None)  ):
        raise ValueError("you need to specify either  (NK,NKFFTmin) or a pair (NKdiv,NKFFT). found ({},{}) and ({},{}) ".format(NK,NKFFTmin,NKdiv,NKFFT))
    if not ((NKdiv is None) or (NKFFT is None)):
        return np.array(one2three(NKdiv)),np.array(one2three(NKFFT))
    lst=[autonk(nk,nkfftmin) for nk,nkfftmin in zip(one2three(NK),one2three(NKFFTmin))]
    return np.array([l[0] for l in lst]),np.array([l[1] for l in lst])



def evaluate_K(func,system,NK=None,NKdiv=None,nproc=0,NKFFT=None,
            adpt_mesh=2,adpt_num_iter=0,adpt_nk=1,fout_name="result",
             symmetry_gen=[SYM.Identity],suffix="",
             GammaCentered=True,file_klist="k_list.pickle",restart=False,start_iter=0):
    """This function evaluates in parallel or serial an integral over the Brillouin zone 
of a function func, which whould receive only one argument of type Data_dk, and return 
a numpy.array of whatever dimensions

the user has to provide 2 grids of K-points - FFT grid anf NKdiv

The parallelisation is done by NKdiv

As a result, the integration will be performed ove NKFFT x NKdiv
"""
    
    
    if not file_klist.endswith(".pickle"):
        file_klist+=".pickle"
    cnt_exclude=0
    
    NKdiv,NKFFT=determineNK(NKdiv,NKFFT,NK,system.NKFFTmin)

    print ("using NKdiv={}, NKFFT={}, NKtot={}".format( NKdiv,NKFFT,NKdiv*NKFFT))
    
    symgroup=SYM.Group(symmetry_gen,basis=system.recip_lattice)

    paralfunc=functools.partial(
        _eval_func_k, func=func,system=system,NKFFT=NKFFT )

    if GammaCentered :
        shift=(NKdiv%2-1)/(2*NKdiv)
    else :
        shift=np.zeros(3)
    print ("shift={}".format(shift))

    if restart:
        try:
            k_list=pickle.load(open(file_klist,"rb"))
            print ("{0} k-points were read from {1}".format(len(k_list),file_klist))
            if len(k_list)==0:
                print ("WARNING : {0} contains zero points starting from scrath".format(file_klist))
                restart=False
        except Exception as err:
            restart=False
            print ("WARNING: {}".format( err) )
            print ("WARNING : reading from {0} failed, starting from scrath".format(file_klist))
            
    if not restart:
        print ("generating k_list")
        k_list=KpointBZ(k=shift, NKFFT=NKFFT,symgroup=symgroup ).divide(NKdiv)
        print ("Done, sum of eights:{}".format(sum(kp.factor for kp in k_list)))
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
        print ("iteration {0} - {1} points. New points are:".format(i_iter,len([k for k in  k_list if k.res is None])) ) #,np.prod(NKFFT)*sum(dk.prod() for dk in dk_list))) 
        for i,k in enumerate(k_list):
          if not k.evaluated:
            print (" k-point {0} : {1} ".format(i,k))
        counter+=process(paralfunc,k_list,nproc,symgroup=symgroup)
        
        try:
            pickle.dump(k_list,open(file_klist,"wb"))
        except Exception as err:
            print ("Warning: {0} \n the k_list was not pickled".format(err))
            
        result_all=sum(kp.get_res for kp in k_list)
        
        if not (restart and i_iter==0):
            result_all.write(fout_name+"-{}"+suffix+"_iter-{0:04d}.dat".format(i_iter+start_iter))
        
        if i_iter >= adpt_num_iter:
            break
             
        # Now add some more points
        kmax=np.array([k.max for k in k_list]).T
        select_points=set().union( *( np.argsort( km )[-adpt_nk:] for km in kmax )  )
        
        l1=len(k_list)
        for ik in select_points:
            k_list+=k_list[ik].divide(adpt_mesh)
#            del k_list[ik]
        
#        print ("sum of weights:{}".format(sum(kp.factor for kp in k_list)))
        print ("checking for equivalent points in all points")
        nexcl=exclude_equiv_points(k_list,new_points=len(k_list)-l1)
        print (" excluded {0} points".format(nexcl))
        print ("sum of eights now :{}".format(sum(kp.factor for kp in k_list)))
        
    
    print ("Totally processed {0} k-points ".format(counter))
    return result_all
       


def _eval_func_k(k,func,system,NKFFT):
    data_dk=Data_dk(system,k,NKFFT=NKFFT)
    return func(data_dk)

