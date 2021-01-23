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
from colorama import init
from termcolor import cprint 



class System():
    """
    The base class for describing a system. Although it has its own constructor, it requires input binary files prepared by a special 
    `branch <https://github.com/stepan-tsirkin/wannier90/tree/save4wberri>`_ of ``postw90.x`` .
    Therefore this class by itself it is not recommended for a feneral user. Instead, 
    please use the child classes, e.g  :class:`~wannierberri.System_w90` or :class:`~wannierberri.System_tb`


    Parameters
    -----------
    seedname : str
        the seedname used in Wannier90
    berry : bool 
        set ``True`` if quantities derived from Berry connection or Berry curvature will be used. Default: {}
    spin : bool
        set ``True`` if quantities derived from spin  will be used.
    morb : bool
        set ``True`` if quantities derived from orbital moment  will be used. Requires the ``.uHu`` file.
    periodic : [bool,bool,bool]
        set ''True'' for periodic directions and ''False''for confined (e.g. slab direction for 2D systems). Not relevant for :class:`~wannierberri.System_TBmodels` and  :class:`~wannierberri.System_PythTB`
    SHCryoo : bool 
        set ``True`` if quantities derived from Ryoo's spin-current elements will be used. (RPS 2019)
    SHCqiao : bool
        set ``True`` if quantities derived from Qiao's approximated spin-current elements will be used. (QZYZ 2018).
    use_ws : bool
        minimal distance replica selection method :ref:`sec-replica`.  equivalent of ``use_ws_distance`` in Wannier90.
    frozen_max : float
        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary.
    degen_thresh : float
        threshold to consider bands as degenerate. Used in calculation of Fermi-surface integrals
    random_gauge : bool
        applies random unitary rotations to degenerate states. Needed only for testing, to make sure that gauge covariance is preserved
    ksep: int
        separate k-point into blocks with size ksep to save memory when summing internal bands matrix. Working on gyotropic_Korb and berry_dipole. 
    delta_fz:float
        size of smearing for B matrix with frozen window, from frozen_max-delta_fz to frozen_max. 

    """

    def __init__(self, old_format=False,    **parameters ):


        self.set_parameters(**parameters)
        self.old_format=old_format

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

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Recommended size of FFT grid", self.NKFFT_recommended)
        print ("Real-space lattice:\n",self.real_lattice)
        #print ("R - points and dege=neracies:\n",iRvec)
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

        self.set_symmetry()

        cprint ("Reading the system finished successfully",'green', attrs=['bold'])


    def set_parameters(self,**parameters):
        self.default_parameters={
                    'seedname':'wannier',
                    'frozen_max':np.Inf,
                    'berry':False,
                    'morb':False,
                    'spin':False,
                    'SHCryoo':False,
                    'SHCqiao':False,
                    'random_gauge':False,
                    'degen_thresh':-1 ,
                    'delta_fz':0.1,
                    'ksep': 1 ,
                    'Emin': -np.Inf ,
                    'Emax': np.Inf ,
                    'use_ws':True,
                    'periodic':(True,True,True)
                       }

        for param in self.default_parameters:
            if param in parameters:
                vars(self)[param]=parameters[param]
            else: 
                vars(self)[param]=self.default_parameters[param]
        self.periodic=np.array(self.periodic)


    def check_periodic(self):
        exclude=np.zeros(self.nRvec,dtype=bool)
        for i,per in enumerate(self.periodic):
            if not per:
                sel=(self.iRvec[:,i]!=0)
                if np.any(sel) :
                    print ("""WARNING : you declared your ystemas non-periodic along direction {i}, but there are {nrexcl} of total {nr} R-vectors with R[{i}]!=0. 
        They will be excluded, please make sure you know what you are doing """.format(i=i,nrexcl=sum(sel),nr=self.nRvec ) )
                    exclude[sel]=True
        if np.any(exclude):
            notexclude=np.logical_not(exclude)
            self.iRvec=self.iRvec[notexclude]
            for X in ['HH','AA','BB','CC','SS','FF']:
                XR=X+'_R'
                if hasattr(self,XR) :
                    vars(self)[XR]=vars(self)[XR][:,:,notexclude]

    @property
    def getAA(self):
        return self.morb or self.berry or self.SHCryoo or self.SHCqiao

    @property
    def getBB(self):
        return self.morb

    @property
    def getCC(self):
        return self.morb


    @property
    def getSS(self):
        return self.spin or self.SHCryoo or self.SHCqiao

    @property
    def getFF(self):
        return False

    @property
    def getSA(self):
        return self.SHCryoo

    @property
    def getSHA(self):
        return self.SHCryoo

    @property
    def getSHC(self):
        return self.SHCqiao


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
        

    def _FFT_compatible(self,FFT,iRvec):
        "check if FFT is enough to fit all R-vectors"
        return np.unique(iRvec%FFT,axis=0).shape[0]==iRvec.shape[0]


#    @lazy_property.LazyProperty
    @property
    def NKFFT_recommended(self):
        "finds a minimal FFT grid on which different R-vectors do not overlap"
        NKFFTrec=np.ones(3,dtype=int)
        for i in range(3):
            R=self.iRvec[:,i]
            if len(R[R>0])>0: 
                NKFFTrec[i]+=R.max()
            if len(R[R<0])>0: 
                NKFFTrec[i]-=R.min()
        assert self._FFT_compatible(NKFFTrec,self.iRvec)
        return NKFFTrec

    def set_symmetry(self,symmetry_gen=[]):
        """ 
        Set the symmetry group of the :class:`~wannierberri.__system.System` 

        Parameters
        ----------
        symmetry_gen : list of :class:`~wannierberri.symmetry.Symmetry` or str
            The generators of the symmetry group. 

        Notes
        -----
        + Only the generators of the symmetry group are essential. However, no problem if more symmetries are provided. 
          The code further evaluates all possible products of symmetry operations, until the full group is restored.
        + Providing `Identity` is not needed. It is included by default
        + Operations are given as objects of class:`~wannierberri.Symmetry.symmetry` or by name as `str`, e.g. ``'Inversion'`` , ``'C6z'``, or products like ``'TimeReversal*C2x'``.
        + ``symetyry_gen=[]`` is equivalent to not calling this function at all
        + Only the **point group** operations are important. Hence, for non-symmorphic operations, only the rotational part should be given, neglecting the translation.

        """
        self.symgroup=Group(symmetry_gen,recip_lattice=self.recip_lattice,real_lattice=self.real_lattice)


    @lazy_property.LazyProperty
    def cRvec(self):
        return self.iRvec.dot(self.real_lattice)

    @property 
    def nRvec(self):
        return self.iRvec.shape[0]


    @lazy_property.LazyProperty
    def cell_volume(self):
        return abs(np.linalg.det(self.real_lattice))



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
        

#
# the following  implements the use_ws_distance = True  (see Wannier90 documentation for details)
#



class map_1R():
   def __init__(self,lines,irvec):
       lines_split=[np.array(l.split(),dtype=int) for l in lines]
       self.dict={(l[0]-1,l[1]-1):l[2:].reshape(-1,3) for l in lines_split}
       self.irvec=np.array([irvec])
       
   def __call__(self,i,j):
       try :
           return self.dict[(i,j)]
       except KeyError:
           return self.irvec
          

class ws_dist_map():
        
    def __call__(self,matrix):
        ndim=len(matrix.shape)-3
        num_wann=matrix.shape[0]
        reshaper=(num_wann,num_wann)+(1,)*ndim
#        print ("check:",matrix.shape,reshaper,ndim)
        matrix_new=np.array([ sum(matrix[:,:,ir]*self._iRvec_new[irvecnew][ir].reshape(reshaper)
                                  for ir in self._iRvec_new[irvecnew] ) 
                                       for irvecnew in self._iRvec_ordered]).transpose( (1,2,0)+tuple(range(3,3+ndim)) )
        assert ( np.abs(matrix_new.sum(axis=2)-matrix.sum(axis=2)).max()<1e-12)
        return matrix_new

    def _add_star(self,ir,irvec_new,iw,jw):
        weight=1./irvec_new.shape[0]
        for irv in irvec_new:
            self._add(ir,irv,iw,jw,weight)


    def _add(self,ir,irvec_new,iw,jw,weight):
        irvec_new=tuple(irvec_new)
        if not (irvec_new in self._iRvec_new):
             self._iRvec_new[irvec_new]=dict()
        if not ir in self._iRvec_new[irvec_new]:
             self._iRvec_new[irvec_new][ir]=np.zeros((self.num_wann,self.num_wann),dtype=float)
        self._iRvec_new[irvec_new][ir][iw,jw]+=weight

    def _init_end(self,nRvec):
        self._iRvec_ordered=sorted(self._iRvec_new)
        for ir  in range(nRvec):
            chsum=0
            for irnew in self._iRvec_new:
                if ir in self._iRvec_new[irnew]:
                    chsum+=self._iRvec_new[irnew][ir]
            chsum=np.abs(chsum-np.ones( (self.num_wann,self.num_wann) )).sum() 
            if chsum>1e-12: print ("WARNING: Check sum for {0} : {1}".format(ir,chsum))



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

        


