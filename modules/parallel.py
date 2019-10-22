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
import lazy_property




def process(paralfunc,k_list,nproc):
    selK=[ik for ik,k in enumerate(k_list) if k.res is None]
    dk_list=[k_list[ik].kp_fullBZ for ik in selK]
    print ("processing {0}  points".format(len(dk_list)))
    if nproc<=0:
        res = [paralfunc(k) for k in dk_list]
    else:
        p=multiprocessing.Pool(nproc)
        res= p.map(paralfunc,dk_list)
        p.close()
    for i,ik in enumerate(selK):
        k_list[ik].set_res(res[i])




class Symmetry():
    def __init__(self):
       pass
       
    def transform_data(self,res):
       return res


class  KpointBZ():

    def __init__(self,k=np.zeros(3),dk=np.ones(3),NKFFT=np.ones(3),symmetries=[]):
        self.k=np.copy(k)
        self.dk=np.copy(dk)    
        self.symmetries=symmetries
        self.res=None
        self.NKFFT=NKFFT
       
    def set_res(self,res):
        self.res=sum(sym.transform_data(res) for sym in self.symmetries)*np.prod(self.dk)


    @lazy_property.LazyProperty
    def kp_fullBZ(self):
        print (self.k,self.NKFFT)
        return self.k/self.NKFFT

    @lazy_property.LazyProperty
    def max(self):
        return np.max(self.res)

    @lazy_property.LazyProperty
    def norm(self):
        return np.linalg.norm(self.res)

    def fraction(self,ndiv):
        assert (ndiv.shape==(3,))
        kp=KpointBZ(self.k,self.dk/ndiv,self.symmetries)
        if self.res is not None:
            kp.res=self.res/np.prod(ndiv)
        return kp
        
    def divide(self,ndiv):
        assert (ndiv.shape==(3,))
        assert (np.all(ndiv>0))
        include_original= np.all( ndiv%2==1)
        
        k0=self.k
        dk_adpt=self.dk/ndiv
        adpt_shift=(-k0+dk_adpt)/2.
        k_list_add=[KpointBZ(k=k0+adpt_shift+dk_adpt*np.array([x,y,z]),dk=dk_adpt,NKFFT=self.NKFFT,symmetries=self.symmetries)
                                 for x in range(ndiv[0]) 
                                  for y in range(ndiv[1]) 
                                   for z in range(ndiv[2])
                            if not (include_original and np.all(np.array([x,y,z])*2+1==adpt_mesh)) ]
        if include_original:
            k_list_add.append(self.fraction(ndiv))
        return k_list_add



def eval_integral_BZ(func,Data,NKdiv=np.ones(3,dtype=int),nproc=0,NKFFT=None,
            adpt_mesh=2,adpt_num_iter=0,adpt_thresh=None,adpt_nk=1,fout_name="result",fun_write=None,symmetry_gen=[]):
    """This function evaluates in parallel or serial an integral over the Brillouin zone 
of a function func, which whould receive only one argument of type Data_dk, and return 
a numpy.array of whatever dimensions
(TODO: in future it might return whatever object, for which the '+','-',abs and max  operation are defined)

the user has to provide 2 grids of K-points - FFT grid anf NKdiv

The parallelisation is done by NKdiv

As a result, the integration will be performed ove NKFFT x NKdiv
"""
        
        
    NKFFT=Data.NKFFT if NKFFT is None else NKFFT

    paralfunc=functools.partial(
        _eval_func_k, func=func,Data=Data,NKFFT=NKFFT )

    all_symmetries=[Symmetry()]
    k_list=KpointBZ(symmetries=all_symmetries,NKFFT=NKFFT ).divide(NKdiv)

        
    result_all=[]
    if adpt_num_iter<0:
        adpt_num_iter=-adpt_num_iter*np.prod(NKdiv)/np.prod(adpt_mesh)/adpt_nk/2
    adpt_num_iter=int(round(adpt_num_iter))


    if (adpt_mesh is None) or np.max(adpt_mesh)<=1:
        adpt_num_iter=0
    else:
        if not isinstance(adpt_mesh, Iterable):
            adpt_mesh=[adpt_mesh]*3
        adpt_mesh=np.array(adpt_mesh)


    
    counter=len(k_list)

    for i_iter in range(adpt_num_iter+1):
        print ("iteration {0} - {1} points".format(i_iter,len([k for k in  k_list if k.res is None])) ) #,np.prod(NKFFT)*sum(dk.prod() for dk in dk_list))) 
        process(paralfunc,k_list,nproc)
        result_all.append(sum(kp.res for kp in k_list))

        if not (fun_write is None):
            fun_write(result_all[-1],fout_name+"_iter-{0:04d}.dat".format(i_iter))
        
        if i_iter == adpt_num_iter:
            break
             
        # Now add some more points
        select_points=np.sort( list(
                    set(np.argsort([ k.max  for k in k_list])[-adpt_nk:]).union(
                    set(np.argsort([ k.norm for k in k_list])[-adpt_nk:])      )
                                 )  )[-1::-1]
        
        cnt1=len(k_list)
        for ik in select_points:
            k_list+=k_list[ik].divide(adpt_mesh)
            del k_list[ik]
        cnt2=len(k_list)
        counter+=cnt2-cnt1
    
    print ("Totally processed {0} k-points ".format(counter))
    return result_all[-1]
       

#        print ("iteration {0} - {1} points, sum of weights = {2}".format(i_iter+1,len(k_list),np.prod(NKFFT)*sum(dk.prod() for dk in dk_list))) 
#        print (" k_list : \n {0}\n".format("\n".join(
#              "{0:3d} :  {1:8.5f} {2:8.5f} {3:8.5f} :    {4:8.5f} {5:8.5f} {6:8.5f} : {7:15.8f}".format(
#                     i,k[0][0],k[0][1],k[0][2],k[1][0],k[1][1],k[1][2] ,np.linalg.norm(k[2]))
#                   for i,k in enumerate( zip(k_list,dk_list,result_K) ) )  ) )
#        if not (fun_write is None):
#            fun_write(result_all[-1],fout_name+"_iter-{0:04d}.dat".format(i_iter+1))
#        if not (adpt_thresh is None):
#            if adpt_thresh>0:
#                if adpt_thresh > np.max(np.abs(result_all[-1]-result_all[-2]))/(np.max(np.abs(result_all[-1]))+np.max(np.abs(result_all[-2])))*2:
#                   break
#    print ("Totally processed {0} k-points ".format(len(result_K)))



def _eval_func_k(k,func,Data,NKFFT):
    data_dk=Data_dk(Data,k,NKFFT=NKFFT)
    return func(data_dk)

