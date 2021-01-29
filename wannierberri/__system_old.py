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
from scipy.io import FortranFile as FF
import copy
import lazy_property
from collections import Iterable
from .__utility import str2bool, alpha_A, beta_A , real_recip_lattice
from  .symmetry import Group
from .__system import System, ws_dist_map



class System_old():
    """
    An obsolete class for describing a system. Its constructor requires input binary files prepared 
     by a special `branch <https://github.com/stepan-tsirkin/wannier90/tree/save4wberri>`_ of ``postw90.x`` .
    Therefore not recommended for a feneral user. 



    """

    def __init__(self,  **parameters ):

        self.set_parameters(**parameters)
        self.old_format=True

        cprint ("Reading from {}".format(seedname+"_HH_save.info"),'green', attrs=['bold'])

        f=open(seedname+"_HH_save.info" if self.old_format else seedname+"_R.info","r")
        l=f.readline().split()[:3]
        self.seedname=seedname
        self.num_wann,nRvec=int(l[0]),int(l[1])
        self.nRvec0=nRvec
        real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.real_lattice,self.recip_lattice=real_recip_lattice(real_lattice=real_lattice)
        iRvec=np.array([f.readline().split()[:4] for i in range(nRvec)],dtype=int)
        
        self.Ndegen=iRvec[:,3]
        self.iRvec=iRvec[:,:3]

        self.cRvec=self.iRvec.dot(self.real_lattice)

        has_ws=str2bool(f.readline().split("=")[1].strip())

        
        if has_ws and self.use_ws:
            print ("using ws_dist")
            self.ws_map=ws_dist_map_read(self.iRvec,self.num_wann,f.readlines())
            self.iRvec=np.array(self.ws_map._iRvec_ordered,dtype=int)
        else:
            self.ws_map=None
        
        f.close()

        self.HH_R=self.__getMat('HH')

        
        if self.getAA:
            self.AA_R=self.__getMat('AA')

        if self.getBB:
            self.BB_R=self.__getMat('BB')
        
        if self.getCC:
            try:
                self.CC_R=1j*self.__getMat('CCab')
            except:
                _CC_R=self.__getMat('CC')
                self.CC_R=1j*(_CC_R[:,:,:,alpha_A,beta_A]-_CC_R[:,:,:,beta_A,alpha_A])

        if self.getFF:
            try:
                self.FF_R=1j*self.__getMat('FFab')
            except:
                _FF_R=self.__getMat('FF')
                self.FF_R=1j*(_FF_R[:,:,:,alpha_A,beta_A]-_FF_R[:,:,:,beta_A,alpha_A])

        if self.getSS:
            self.SS_R=self.__getMat('SS')

        self.finalise_init()        


    def __getMat(self,suffix):

        f=FF(self.seedname+"_" + suffix+"_R"+(".dat" if self.old_format else ""))
        MM_R=np.array([[np.array(f.read_record('2f8'),dtype=float) for m in range(self.num_wann)] for n in range(self.num_wann)])
        MM_R=MM_R[:,:,:,0]+1j*MM_R[:,:,:,1]
        f.close()
        ncomp=MM_R.shape[2]/self.nRvec0
        if ncomp==1:
            result=MM_R/self.Ndegen[None,None,:]
        elif ncomp==3:
            result= MM_R.reshape(self.num_wann, self.num_wann, 3, self.nRvec0).transpose(0,1,3,2)/self.Ndegen[None,None,:,None]
        elif ncomp==9:
            result= MM_R.reshape(self.num_wann, self.num_wann, 3,3, self.nRvec0).transpose(0,1,4,3,2)/self.Ndegen[None,None,:,None,None]
        else:
            raise RuntimeError("in __getMat: invalid ncomp : {0}".format(ncomp))
        if self.ws_map is None:
            return result
        else:
            return self.ws_map(result)


class ws_dist_map_read(ws_dist_map):
    def __init__(self,iRvec,num_wann,lines):
        nRvec=iRvec.shape[0]
        self.num_wann=num_wann
        self._iRvec_new=dict()
        n_nonzero=np.array([l.split()[-1] for l in lines[:nRvec]],dtype=int)
        lines=lines[nRvec:]
        for ir,nnz in enumerate(n_nonzero):
            map1r=map_1R(lines[:nnz],iRvec[ir])
            for iw in range(num_wann):
                for jw in range(num_wann):
                    self._add_star(ir,map1r(iw,jw),iw,jw)
            lines=lines[nnz:]
        self._init_end(nRvec)

        


