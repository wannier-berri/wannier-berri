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




def process(paralfunc,k_list,nproc):
    print ("processing {0}  points".format(len(k_list)))
    if nproc<=0:
        return [paralfunc(k) for k in k_list]
    else:
        p=multiprocessing.Pool(nproc)
        res= p.map(paralfunc,k_list)
        p.close()
        return res

def eval_integral_BZ(func,Data,NKdiv=np.ones(3,dtype=int),nproc=0,NKFFT=None,
            adpt_mesh=2,adpt_num_iter=0,adpt_thresh=None,adpt_nk=1,fout_name="result",fun_write=None):
    """This function evaluates in parallel or serial an integral over the Brillouin zone 
of a function func, which whould receive only one argument of type Data_dk, and return 
a numpy.array of whatever dimensions
(TODO: in future it might return whatever object, for which the '+','-',abs and max  operation are defined)

the user has to provide 2 grids of K-points - FFT grid anf NKdiv

The parallelisation is done by NKdiv

As a result, the integration will be performed ove NKFFT x NKdiv
"""
        
        
    NKFFT=Data.NKFFT if NKFFT is None else NKFFT
    dk1=1./(NKFFT*NKdiv)
    k_list=[dk1*np.array([x,y,z]) for x in range(NKdiv[0]) 
        for y in range(NKdiv[1]) for z in range(NKdiv[2]) ]
    dk_list=[np.copy(dk1)  for i in range(len(k_list))]


    print ("iteration {0} - {1} points, sum of weights = {2}".format(0,len(k_list),np.prod(NKFFT)*sum(dk.prod() for dk in dk_list))) 
#    print (" k_list : ", k_list)
#    print ("dk_list : ",dk_list)
    

    paralfunc=functools.partial(
        _eval_func_k, func=func,Data=Data,NKFFT=NKFFT )

    result_K=process(paralfunc,k_list,nproc)

    result_all=[sum(res*np.prod(dk) for res,dk in zip(result_K,dk_list))*np.prod(NKFFT)]
    if not (fun_write is None):
        fun_write(result_all[-1],fout_name+".dat")
    
    
    if (adpt_mesh is None) or np.max(adpt_mesh)<=1:
        return result_all[-1]
    
##    now refining high-contribution points
    if not isinstance(adpt_mesh, Iterable):
        adpt_mesh=[adpt_mesh]*3
    adpt_mesh=np.array(adpt_mesh)
    include_original= np.all( adpt_mesh%2==1)
    weight=1./np.prod(adpt_mesh)
    #keep the number of refined points constant
#    NK_sel=int(round( len(result_K)*adpt_percent/np.prod(adpt_mesh)/100 ))
    NK_sel=adpt_nk
    
    if adpt_num_iter<0:
        adpt_num_iter=-adpt_num_iter*np.prod(NKdiv)/np.prod(adpt_mesh)/NK_sel/2

#Noe start refinement iterations:
    for i_iter in range(int(round(adpt_num_iter))):

        select_points=np.array( list(
                    set(np.argsort([ np.abs(a).max()*dk.prod() for a,dk in zip(result_K,dk_list)])[-NK_sel:]).union(
                    set(np.argsort([ np.abs(a).sum()*dk.prod() for a,dk in zip(result_K,dk_list)])[-NK_sel:]) )
                                 )  )
        print ("dividing {0} points into {1} : {2}".format(NK_sel,adpt_mesh,select_points))
        k_list_refined=[]
        dk_list_refined=[]
        #now form the list of additional k-points:
        for ipoint in select_points:
                
            k0=k_list[ipoint]
            dk_adpt=dk_list[ipoint]/adpt_mesh
            adpt_shift=(-dk_list[ipoint]+dk_adpt)/2.
            k_list_add=[k0+adpt_shift+dk_adpt*np.array([x,y,z])
                                 for x in range(adpt_mesh[0]) 
                                  for y in range(adpt_mesh[1]) 
                                   for z in range(adpt_mesh[2])
                            if not (include_original and np.all(np.array([x,y,z])*2+1==adpt_mesh)) ]

            if include_original:
                result_K[ipoint]*=weight
                dk_list[ipoint][:]=dk_list[ipoint]/adpt_mesh
            else:
                result_K[ipoint]*=0.
                dk_list[ipoint]*=0.

            k_list_refined+=k_list_add
            dk_list_refined+=[np.copy(dk_adpt) for i in range(len(k_list_add))]

        result_K+=process(paralfunc,k_list_refined,nproc)
        dk_list+=dk_list_refined
        k_list+=k_list_refined
        result_all.append(sum(res*np.prod(dk) for res,dk in zip(result_K,dk_list) )*np.prod(NKFFT))
        print ("iteration {0} - {1} points, sum of weights = {2}".format(i_iter+1,len(k_list),np.prod(NKFFT)*sum(dk.prod() for dk in dk_list))) 
        print (" k_list : \n {0}\n".format("\n".join(
              "{0:3d} :  {1:8.5f} {2:8.5f} {3:8.5f} :    {4:8.5f} {5:8.5f} {6:8.5f} : {7:15.8f}".format(
                     i,k[0][0],k[0][1],k[0][2],k[1][0],k[1][1],k[1][2] ,np.linalg.norm(k[2]))
                   for i,k in enumerate( zip(k_list,dk_list,result_K) ) )  ) )
        if not (fun_write is None):
            fun_write(result_all[-1],fout_name+"_iter-{0:04d}.dat".format(i_iter+1))
        if not (adpt_thresh is None):
            if adpt_thresh>0:
                if adpt_thresh > np.max(np.abs(result_all[-1]-result_all[-2]))/(np.max(np.abs(result_all[-1]))+np.max(np.abs(result_all[-2])))*2:
                   break
    print ("Totally processed {0} k-points ".format(len(result_K)))
    return result_all



def _eval_func_k(k,func,Data,NKFFT):
    data_dk=Data_dk(Data,k,NKFFT=NKFFT)
    return func(data_dk)

