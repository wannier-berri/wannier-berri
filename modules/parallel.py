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
import get_data





def eval_integral_BZ(func,Data,NKdiv=np.ones(3,dtype=int),parallel=False,nproc=1,NKFFT=None):
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
    if parallel:
        p=multiprocessing.Pool(nproc)
        return sum(p.map(paralfunc,dk_list))/len(dk_list)
    else:
        return sum(paralfunc(dk) for dk in dk_list)/len(dk_list)


def _eval_func_dk(dk,func,Data,NKFFT):
    data_dk=get_data.Data_dk(Data,dk,NKFFT=NKFFT)
    return func(data_dk)

