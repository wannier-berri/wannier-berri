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
#   some parts of this file are originate                    #
# from the translation of Wannier90 code                     #
#------------------------------------------------------------#

import numpy as np
from scipy.io import FortranFile 
import copy
import lazy_property
import functools
import multiprocessing 
from .__utility import str2bool, alpha_A, beta_A, iterate3dpm, fourier_q_to_R
from colorama import init
from termcolor import cprint 
from .__system import System, ws_dist_map
from .__w90_files import EIG,MMN,CheckPoint,SPN,UHU

class System_w90(System):

    def __init__(self,seedname="wannier90",
                    berry=False,spin=False,morb=False,
                    use_ws=True,
                    frozen_max=-np.Inf,
                    random_gauge=False,
                    degen_thresh=-1 ,
                    num_proc=2  ):

        self.seedname=seedname

        self.morb  = morb
        self.berry = berry
        self.spin  = spin

        self.AA_R=None
        self.BB_R=None
        self.CC_R=None
        self.FF_R=None
        self.SS_R=None


        getAA = False
        getBB = False
        getCC = False
        getSS = False
        getFF = False
        
        if self.morb: 
            getAA=getBB=getCC=True
        if self.berry: 
            getAA=True
        if self.spin: 
            getSS=True

        self.frozen_max=frozen_max
        self.random_gauge=random_gauge
        self.degen_thresh=degen_thresh

        chk=CheckPoint(self.seedname)
        self.real_lattice=chk.real_lattice
        self.recip_lattice=chk.recip_lattice
        self.iRvec,self.Ndegen=self.wigner_seitz(chk.mp_grid)
#        print ("number of R-vectors: {} ; vectrors:\n {}".format(self.iRvec.shape[0], self.iRvec,self.Ndegen))
        self.nRvec0=len(self.iRvec)
        self.num_wann=chk.num_wann

        eig=EIG(seedname)
        if getAA or getBB:
            mmn=MMN(seedname)

        kpt_mp_grid=[tuple(k) for k in np.array( np.round(chk.kpt_latt*np.array(chk.mp_grid)[None,:]),dtype=int)%chk.mp_grid]
#        print ("kpoints:",kpt_mp_grid)
        fourier_q_to_R_loc=functools.partial(fourier_q_to_R, mp_grid=chk.mp_grid,kpt_mp_grid=kpt_mp_grid,iRvec=self.iRvec,ndegen=self.Ndegen,num_proc=num_proc)

        self.HH_R=fourier_q_to_R_loc( chk.get_HH_q(eig) )
#        for i in range(self.nRvec):
#            print (i,self.iRvec[i],"H(R)=",self.HH_R[0,0,i])

        if getAA:
            self.AA_R=fourier_q_to_R_loc(chk.get_AA_q(mmn))

        if getBB:
            self.BB_R=fourier_q_to_R_loc(chk.get_AA_q(mmn,eig))

        if getCC:
            uhu=UHU(seedname)
            self.CC_R=fourier_q_to_R_loc(chk.get_CC_q(uhu,mmn))

        if getSS:
            spn=SPN(seedname)
            self.SS_R=fourier_q_to_R_loc(chk.get_SS_q(spn))

        if  use_ws:
            print ("using ws_distance")
            ws_map=ws_dist_map_gen(self.iRvec,chk.wannier_centres, chk.mp_grid,self.real_lattice)
            for X in ['HH','AA','BB','CC','SS','FF']:
                XR=X+'_R'
                if vars(self)[XR] is not None:
                    print ("using ws_dist for {}".format(XR))
                    vars(self)[XR]=ws_map(vars(self)[XR])
            self.iRvec=np.array(ws_map._iRvec_ordered,dtype=int)

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)

    def wigner_seitz(self,mp_grid):
        real_metric=self.real_lattice.T.dot(self.real_lattice)
        mp_grid=np.array(mp_grid)
        irvec=[]
        ndegen=[]
        for n in iterate3dpm(mp_grid):
            # Loop over the 125 points R. R=0 corresponds to i1=i2=i3=0,
            # or icnt=63  (62 starting from zero)
            dist=[]
            for i in iterate3dpm((2,2,2)):
                ndiff=n-i*mp_grid
                dist.append(ndiff.dot(real_metric.dot(ndiff)))
            dist_min = np.min(dist)
            if  abs(dist[62] - dist_min) < 1.e-7 :
                irvec.append(n)
                ndegen.append(np.sum( abs(dist - dist_min) < 1.e-7 ))
    
        return np.array(irvec),np.array(ndegen)


    def to_tb_file(self,tb_file=None):
        if tb_file is None: 
            tb_file=self.seedname+"_fromchk_tb.dat"
        f=open(tb_file,"w")
        f.write("written by wannier-berri form the chk file\n")
#        cprint ("reading TB file {0} ( {1} )".format(tb_file,l.strip()),'green', attrs=['bold'])
        np.savetxt(f,self.real_lattice)
        f.write("{}\n".format(self.num_wann))
        f.write("{}\n".format(self.nRvec))
        for i in range(0,self.nRvec,15):
            a=self.Ndegen[i:min(i+15,self.nRvec)]
            f.write("  ".join("{:2d}".format(x) for x in a)+"\n")
        for iR in range(self.nRvec):
            f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
            f.write("".join("{0:3d} {1:3d} {2:15.8e} {3:15.8e}\n".format(
                         m+1,n+1,self.HH_R[m,n,iR].real*self.Ndegen[iR],self.HH_R[m,n,iR].imag*self.Ndegen[iR]) 
                             for n in range(self.num_wann) for m in range(self.num_wann)) )
        if hasattr(self,'AA_R'):
          for iR in range(self.nRvec):
            f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
            f.write("".join("{0:3d} {1:3d} ".format(
                         m+1,n+1) + " ".join("{:15.8e} {:15.8e}".format(a.real,a.imag) for a in self.AA_R[m,n,iR]*self.Ndegen[iR] )+"\n"
                             for n in range(self.num_wann) for m in range(self.num_wann)) )
        f.close()



class ws_dist_map_gen(ws_dist_map):

    def __init__(self,iRvec,wannier_centres, mp_grid,real_lattice):
    ## Find the supercell translation (i.e. the translation by a integer number of
    ## supercell vectors, the supercell being defined by the mp_grid) that
    ## minimizes the distance between two given Wannier functions, i and j,
    ## the first in unit cell 0, the other in unit cell R.
    ## I.e., we find the translation to put WF j in the Wigner-Seitz of WF i.
    ## We also look for the number of equivalent translation, that happen when w_j,R
    ## is on the edge of the WS of w_i,0. The results are stored 
    ## a dictionary shifts_iR[(iR,i,j)]
        ws_search_size=np.array([2]*3)
        ws_distance_tol=1e-5
        cRvec=iRvec.dot(real_lattice)
        mp_grid=np.array(mp_grid)
        shifts_int_all= np.array([ijk  for ijk in iterate3dpm(ws_search_size+1)])*np.array(mp_grid[None,:])
        self.num_wann=wannier_centres.shape[0]
        self._iRvec_new=dict()

        for ir,iR in enumerate(iRvec):
          for jw in range(self.num_wann):
            for iw in range(self.num_wann):
              # function JW translated in the Wigner-Seitz around function IW
              # and also find its degeneracy, and the integer shifts needed
              # to identify it
              R_in=-wannier_centres[iw] +cRvec[ir] + wannier_centres[ jw]
              dist=np.linalg.norm( R_in[None,:]+shifts_int_all.dot(real_lattice),axis=1)
              irvec_new=iR+shifts_int_all[ dist-dist.min() < ws_distance_tol ].copy()
              self._add_star(ir,irvec_new,iw,jw)
        self._init_end(iRvec.shape[0])



