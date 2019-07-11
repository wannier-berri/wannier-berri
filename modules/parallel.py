#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                     written by                             #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#



import  multiprocessing 
import functools
import numpy as np
from data_dk import Data_dk
from collections import Iterable




def process(paralfunc,dk_list,nproc):
    if nproc<=0:
        return [paralfunc(dk) for dk in dk_list]
    else:
        p=multiprocessing.Pool(nproc)
        return p.map(paralfunc,dk_list)


def eval_integral_BZ(func,Data,NKdiv=np.ones(3,dtype=int),nproc=0,NKFFT=None,
            adpt_mesh=None,adpt_percent=None):
    """This function evaluates in parallel or serial an integral over the Brillouin zone 
of a function func, which whould receive only one argument of type Data_dk, and return whatever object,
for which the '+' operation is defined.

the user has to provide 2 grids of K-points - FFT grid anf NKdiv

The parallelisation is done by NKdiv

As a result, the integration will be performed ove NKFFT x NKdiv
"""
    NKFFT=Data.NKFFT if NKFFT is None else NKFFT
    dk1=1./(NKFFT*NKdiv)
    dk_list=[dk1*np.array([x,y,z]) for x in range(NKdiv[0]) 
        for y in range(NKdiv[1]) for z in range(NKdiv[2]) ]
    paralfunc=functools.partial(
        _eval_func_dk, func=func,Data=Data,NKFFT=NKFFT )

    result_nonrefined=process(paralfunc,dk_list,nproc)

    return_before_refinement=sum(result_nonrefined)/np.prod(NKdiv)
    if (adpt_mesh is None) or adpt_mesh<=1:
        return return_before_refinement,return_before_refinement
    
##    now refining high-contribution points
    if not isinstance(adpt_mesh, Iterable):
        adpt_mesh=[adpt_mesh]*3
    adpt_mesh=np.array(adpt_mesh)
    include_original= np.all( adpt_mesh%2==1)
    weight=1./np.prod(adpt_mesh)
## If percent of refined, choose it such, as to double the calculation time by refining
    if adpt_percent is None:
        adpt_percent=100./np.prod(adpt_mesh)
    result_nonrefined=[np.array(a) for a in result_nonrefined]
    NK_sel=int(round( len(result_nonrefined)*adpt_percent/100. ))
    select_points=np.argsort([ np.abs(a).max() for a in result_nonrefined])[:NK_sel]
    dk_list_refined=[]
    print ("dividing {0} points into {1} ".format(NK_sel,adpt_mesh))

#now form the list of additional k-points:
    for ipoint in select_points:
        if include_original:
            result_nonrefined[ipoint]*=weight
        else:
            result_nonrefined[ipoint]*=0.
        k0=dk_list[ipoint]
        dk_adpt=dk1/adpt_mesh
        adpt_shift=(-dk1+dk_adpt)/2.
        dk_list_add=[k0+adpt_shift+dk_adpt*np.array([x,y,z])
                                 for x in range(adpt_mesh[0]) 
                                  for y in range(adpt_mesh[1]) 
                                   for z in range(adpt_mesh[2])
                            if not (include_original and np.all(np.array([x,y,z])*2+1==adpt_mesh)) ]
        dk_list_refined+=dk_list_add
    result_refined=process(paralfunc,dk_list_refined,nproc)
    return_after_refinement=(sum(result_nonrefined)+weight*sum(result_refined) )/np.prod(NKdiv)
    return return_after_refinement,return_before_refinement



def _eval_func_dk(dk,func,Data,NKFFT):
    data_dk=Data_dk(Data,dk,NKFFT=NKFFT)
    return func(data_dk)

