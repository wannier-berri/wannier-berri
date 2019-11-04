import parent
from utility import  print_my_name_start,print_my_name_end,voidsmoother
import numpy as np
from scipy import constants as constants
from collections import Iterable


from berry import eval_J0,get_occ

class SPNresult(parent.AxialVectorResult):
   pass

def calcSpinTot(data,Efermi=None,occ_old=None,smoother=voidsmoother):

    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        SPN=np.zeros( ( nFermi,3) ,dtype=float )
        for iFermi in range(nFermi):
            SPN[iFermi]=calcSpinTot(data,Efermi=Efermi[iFermi],occ_old=occ_old)
        return SPNresult(Efermi,np.cumsum(SPN,axis=0),smoother=smoother)

    occ_new=get_occ(data.E_K,Efermi)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    delocc=occ_new_selk!=occ_old_selk
    occ_old[:,:]=occ_new[:,:]
    return eval_J0(data.SSUU_K_rediag[selectK], delocc)/data.NKFFT_tot
