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
import copy
import lazy_property
from .__utility import str2bool, alpha_A, beta_A , real_recip_lattice
from  .symmetry import Group
from colorama import init
from termcolor import cprint 





class System():

    default_parameters =  {    'seedname':'wannier90',
                    'frozen_max': -np.Inf,
                    'berry':False,
                    'morb':False,
                    'spin':False,
                    'SHCryoo':False,
                    'SHCqiao':False,
                    'use_ws':True,
                    'periodic':(True,True,True),
                    'use_wcc_phase':False,
                    'wannier_centers_cart':None,
                    'wannier_centers_reduced' : None,
                    '_getFF' : False,
                       }


    __doc__ = """
    The base class for describing a system. Although it has its own constructor, it requires input binary files prepared by a special 
    `branch <https://github.com/stepan-tsirkin/wannier90/tree/save4wberri>`_ of ``postw90.x`` .
    Therefore this class by itself it is not recommended for a feneral user. Instead, 
    please use the child classes, e.g  :class:`~wannierberri.System_w90` or :class:`~wannierberri.System_tb`


    Parameters
    -----------
    seedname : str
        the seedname used in Wannier90. Default: ``{seedname}``
    berry : bool 
        set ``True`` if quantities derived from Berry connection or Berry curvature will be used. Default: ``{berry}``
    spin : bool
        set ``True`` if quantities derived from spin  will be used. Default:``{spin}``
    morb : bool
        set ``True`` if quantities derived from orbital moment  will be used. Requires the ``.uHu`` file. Default: ``{morb}``
    periodic : [bool,bool,bool]
        set ``True`` for periodic directions and ``False`` for confined (e.g. slab direction for 2D systems). If less then 3 values provided, the rest are treated as ``False`` . Default : ``{periodic}``
    SHCryoo : bool 
        set ``True`` if quantities derived from Ryoo's spin-current elements will be used. (RPS 2019) Default: ``{SHCryoo}``
    SHCqiao : bool
        set ``True`` if quantities derived from Qiao's approximated spin-current elements will be used. (QZYZ 2018). Default: ``{SHCqiao}``
    use_ws : bool
        minimal distance replica selection method :ref:`sec-replica`.  equivalent of ``use_ws_distance`` in Wannier90. Default: ``{use_ws}``
    frozen_max : float
        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary. Default: ``{frozen_max}``
    _getFF : bool
        generate the FF_R matrix based on the uIu file. May be used for only testing so far. Default : ``{_getFF}``
    use_wcc_phase: bool
        using wannier centers in Fourier transform. Correspoinding to Convention I (True), II (False) in Ref."Tight-binding formalism in the context of the PythTB package". Default: ``{use_wcc_phase}``
    wannier_centers_cart :  array-like(num_wann,3)
        use the given wannier_centers (cartesian) instead of those determined automatically. Incompatible with `wannier_centers_reduced`
    wannier_centers_reduced :  array-like(num_wann,3)
        use the given wannier_centers (reduced) instead of those determined automatically. Incompatible with `wannier_centers_cart`

    """ .format(**default_parameters)



    def set_parameters(self,**parameters):

        for param in self.default_parameters:
            if param in parameters:
                vars(self)[param]=parameters[param]
            else: 
                vars(self)[param]=self.default_parameters[param]
        periodic=np.zeros(3,dtype=bool)
        periodic[:len(self.periodic)]=self.periodic
        self.periodic=periodic


    def check_periodic(self):
        exclude=np.zeros(self.nRvec,dtype=bool)
        for i,per in enumerate(self.periodic):
            if not per:
                sel=(self.iRvec[:,i]!=0)
                if np.any(sel) :
                    print ("""WARNING : you declared your system as non-periodic along direction {i}, but there are {nrexcl} of total {nr} R-vectors with R[{i}]!=0. 
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
        return self._getFF

    @property
    def getSA(self):
        return self.SHCryoo

    @property
    def getSHA(self):
        return self.SHCryoo

    @property
    def getSHC(self):
        return self.SHCqiao


    def do_at_end_of_init(self):
        self.set_symmetry  ()
        self.check_periodic()
        self.set_wannier_centers  ()
        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Recommended size of FFT grid", self.NKFFT_recommended)



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
                         m+1,n+1,self.Ham_R[m,n,iR].real*self.Ndegen[iR],self.Ham_R[m,n,iR].imag*self.Ndegen[iR]) 
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

    @lazy_property.LazyProperty
    def cRvec_p_wcc(self):
        """ 
        With self.use_wcc_phase=True it is R+tj-ti. With self.use_wcc_phase=False it is R. [i,j,iRvec,a] (Cartesian)
        """
        if self.use_wcc_phase:
            return self.cRvec[None,None,:,:]+ self.diff_wcc_cart[:,:,None,:]
        else:
            return self.cRvec[None,None,:,:]

    @lazy_property.LazyProperty
    def diff_wcc_cart(self):
        """ 
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [i,j,a] (Cartesian)
        """
        wannier_centers = self.wannier_centers_cart
        return np.array([[j-i for j in wannier_centers] for i in wannier_centers])

    @lazy_property.LazyProperty
    def diff_wcc_red(self):
        """ 
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [m,n,a] (Reduced)
        """
        wannier_centers = self.wannier_centers_reduced
        return np.array([[j-i for j in wannier_centers] for i in wannier_centers])

    def set_wannier_centers(self):
        if self.wannier_centers_cart is not None:
            if self.wannier_centers_reduced is not None:
                raise ValueError("one should not specify both wannier_centers_cart and wannier_centers_reduced")
            else:
                self.wannier_centers_reduced = self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))
        elif self.wannier_centers_reduced is not None:
                self.wannier_centers_cart = self.wannier_centers_reduced.dot(self.real_lattice)
        elif hasattr(self,"wannier_centers_cart_auto"):
                self.wannier_centers_cart = self.wannier_centers_cart_auto
                self.wannier_centers_reduced = self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))
#        self.wannier_centers_cart*=00.01
#        self.wannier_centers_reduced*=00.01
#        print ("Wannier_centers\n",self.wannier_centers_cart,self.wannier_centers_reduced)
        if self.use_wcc_phase: 
            if self.wannier_centers_cart is None:
                raise ValueError("use_wcc_phase = True, but the wannier centers could not be detyermined")
            if hasattr(self,'AA_R'):
                AA_R_new = np.copy(self.AA_R)
                AA_R_new[np.arange(self.num_wann),np.arange(self.num_wann),self.iR0,:] -= self.wannier_centers_cart
            if hasattr(self,'BB_R'):
                print ("WARNING: orbital moment does not work with wcc_phase so far")
                BB_R_new = self.BB_R.copy() - self.Ham_R[:,:,:,None]*self.wannier_centers_cart[None,:,None,:]
            if hasattr(self,'CC_R'):
                print ("WARNING: orbital moment does not work with wcc_phase so far")
                norm = np.linalg.norm(self.CC_R - self.conj_XX_R(self.CC_R))
                assert norm<1e-10 , f"norm={norm}"
                assert hasattr(self,'BB_R') , "if you use CC_R, you need also BB_R"
                T  =  self.wannier_centers_cart[:,None,None,:,None]*self.BB_R[:,:,:,None,:]
                CC_R_new  =  self.CC_R.copy() + 1.j*sum(   
                            s*( -T[:,:,:,a,b]   # -t_i^a * B_{ij}^b(R)
                                -self.conj_XX_R(T[:,:,:,b,a])    # - B_{ji}^a(-R)^*  * t_j^b 
                                +self.wannier_centers_cart[:,None,None,a]*self.Ham_R[:,:,:,None] * self.wannier_centers_cart[None,:,None,b]  # + t_i^a*H_ij(R)t_j^b
                            )
                        for (s,a,b) in [(+1,alpha_A,beta_A) , (-1,beta_A,alpha_A)] )
                norm = np.linalg.norm(CC_R_new - self.conj_XX_R(CC_R_new))
                assert norm<1e-10 , f"norm={norm}"


            # not sure if the following is correct (Stepan)
            if hasattr(self,'SA_R'):
                assert hasattr(self,'SS_R') , "if you use SA_R, you need also SS_R"
                SA_R_new  =  self.SA_R.copy() - self.SS_R[:,:,:,:,None]*self.wannier_centers_cart[None,:,None,None,:]
            if hasattr(self,'SHA_R'):
                assert hasattr(self,'SH_R') , "if you use SA_R, you need also SH_R"
                SHA_R_new  =  self.SHA_R.copy() - self.SS_R[:,:,:,:,None]*self.wannier_centers_cart[None,:,None,None,:]

            for X in ['AA','BB','CC','SA','SHA']:
                if hasattr(self,X+'_R'):
                    vars(self)[X+'_R'] = locals()[X+'_R_new']

            for X in ['SA','SHA','SR','SH','SHR']:
                if hasattr(self,X+'_R'):
                    pass
#                    raise NotImplementedError(f"use_wcc_phases=True is not implemented for {X}_R")


    @property
    def iR0(self):
        return self.iRvec.tolist().index([0,0,0])

    @lazy_property.LazyProperty
    def reverseR(self):
        """maps the R vector -R"""
        iRveclst= self.iRvec.tolist()
        mapping = np.all( self.iRvec[:,None,:]+self.iRvec[None,:,:] == 0 , axis = 2 )
        # check if some R-vectors do not have partners
        notfound = np.where(np.logical_not(mapping.any(axis=1)))[0]
        for ir in notfound:
            print ("WARNING : R[{}] = {} does not have a -R partner".format(ir,self.iRvec[ir]) )
        # check if some R-vectors have more then 1 partner 
        morefound = np.where(np.sum(mapping,axis=1)>1)[0]
        if len(morefound>0):
            raise RuntimeError( "R vectors number {} have more then one negative partner : \n{} \n{}".format(
                            morefound,self.iRvec[morefound],np.sum(mapping,axis=1) ) )
        lst1,lst2=[],[]
        for ir1 in range(self.nRvec):
            ir2 = np.where(mapping[ir1])[0]
            if len(ir2)==1:
                lst1.append(ir1)
                lst2.append(ir2[0])
        return np.array(lst1),np.array(lst2)

    def conj_XX_R(self,XX_R):
        """ reverses the R-vector and takes the hermitian conjugate """
        XX_R_new = np.zeros_like(XX_R)
        lst1,lst2 = self.reverseR
        assert np.all(self.iRvec[lst1] + self.iRvec[lst2] ==0 )
        XX_R_new [:,:,lst1] = XX_R[:,:,lst2]
        return XX_R_new.swapaxes(0,1).conj()

    @property 
    def nRvec(self):
        return self.iRvec.shape[0]


    @lazy_property.LazyProperty
    def cell_volume(self):
        return abs(np.linalg.det(self.real_lattice))


    @property
    def iR0(self):
        return self.iRvec.tolist().index([0,0,0])

    @lazy_property.LazyProperty
    def reverseR(self):
        """maps the R vector -R"""
        iRveclst= self.iRvec.tolist()
        mapping = np.all( self.iRvec[:,None,:]+self.iRvec[None,:,:] == 0 , axis = 2 )
        # check if some R-vectors do not have partners
        notfound = np.where(np.logical_not(mapping.any(axis=1)))[0]
        for ir in notfound:
            print ("WARNING : R[{}] = {} does not have a -R partner".format(ir,self.iRvec[ir]) )
        # check if some R-vectors have more then 1 partner 
        morefound = np.where(np.sum(mapping,axis=1)>1)[0]
        if len(morefound>0):
            raise RuntimeError( "R vectors number {} have more then one negative partner : \n{} \n{}".format(
                            morefound,self.iRvec[morefound],np.sum(mapping,axis=1) ) )
        lst1,lst2=[],[]
        for ir1 in range(self.nRvec):
            ir2 = np.where(mapping[ir1])[0]
            if len(ir2)==1:
                lst1.append(ir1)
                lst2.append(ir2[0])
#                print (ir1,self.iRvec[ir1] , ir2,self.iRvec[ir2[0]])
        return np.array(lst1),np.array(lst2)

    def conj_XX_R(self,XX_R):
        """ reverses the R-vector and takes the hermitian conjugate """
        XX_R_new = np.zeros_like(XX_R)
        lst1,lst2 = self.reverseR
        assert np.all(self.iRvec[lst1] + self.iRvec[lst2] ==0 )
#        print (XX_R.shape,XX_R_new.shape,lst1,lst2)
        XX_R_new [:,:,lst1] = np.copy(XX_R)[:,:,lst2]
        XX_R_new[:] = XX_R_new.swapaxes(0,1).conj()
        return np.copy(XX_R_new)

    def check_hermitian(self,XX):
        if hasattr(self,XX):
            XX_R = np.copy(vars(self)[XR])
            assert (np.max(abs(XX_R-self.conh_XX_R(XX_R)))<1e-8) , f"{XX} should obey X(-R) = X(R)^\dagger"
        else:
            print (f"{XX} is missing,nothing to check")


