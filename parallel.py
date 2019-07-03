import  multiprocessing 
import functools
import numpy as np
import get_data


def eval_func_dk(dk,func,Data,NK):
    data_dk=get_data.Data_dk(Data,dk,NK=NK)
    return func(data_dk)


def eval_integral_BZ(func,Data,NKdiv=np.ones(3,dtype=int),parallel=False,nproc=1,NK=None):
    NK=Data.NK if NK is None else NK
    dk1=1./(NK*NKdiv)
    dk_list=[dk1*np.array([x,y,z]) for x in range(NKdiv[0]) for y in range(NKdiv[1]) for z in range(NKdiv[2]) ]
    paralfunc=functools.partial(
        eval_func_dk, func=func,Data=Data,NK=NK )
    if parallel:
        p=multiprocessing.Pool(nproc)
        return sum(p.map(paralfunc,dk_list))/len(dk_list)
    else:
        return sum(paralfunc(dk) for dk in dk_list)/len(dk_list)
