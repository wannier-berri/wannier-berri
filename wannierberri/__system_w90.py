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

from .__utility import str2bool, alpha_A, beta_A
from colorama import init
from termcolor import cprint 



class System():

    def __init__(self,seedname="wannier90",tb_file=None,
                    getAA=False,
                    getBB=False,getCC=False,
                    getSS=False,
                    getFF=False,
                    use_ws=True,
                    frozen_max=-np.Inf,
                    random_gauge=False,
                    degen_thresh=-1 ,
                    old_format=False
                                ):


        self.frozen_max=frozen_max
        self.random_gauge=random_gauge
        self.degen_thresh=degen_thresh
        self.old_format=old_format
        self.AA_R=None
        self.BB_R=None
        self.CC_R=None
        self.FF_R=None
        self.SS_R=None

        if tb_file is not None:
            self.__from_tb_file(tb_file,getAA=getAA)
            return
        cprint ("Reading from {}".format(seedname+"_HH_save.info"),'green', attrs=['bold'])

        f=open(seedname+"_HH_save.info" if self.old_format else seedname+"_R.info","r")
        l=f.readline().split()[:3]
        self.seedname=seedname
        self.num_wann,nRvec,self.spinors=int(l[0]),int(l[1]),str2bool(l[2])
        self.nRvec0=nRvec
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice).T
        iRvec=np.array([f.readline().split()[:4] for i in range(nRvec)],dtype=int)
        
        self.Ndegen=iRvec[:,3]
        self.iRvec=iRvec[:,:3]

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        #print ("R - points and dege=neracies:\n",iRvec)
        has_ws=str2bool(f.readline().split("=")[1].strip())
        
        if has_ws and use_ws:
            print ("using ws_dist")
            self.ws_map=ws_dist_map(self.iRvec,self.num_wann,f.readlines())
            self.iRvec=np.array(self.ws_map._iRvec_ordered,dtype=int)
        else:
            self.ws_map=None
        
        f.close()
        if getCC:
           getBB=True

        self.HH_R=self.__getMat('HH')


        
        if getAA:
            self.AA_R=self.__getMat('AA')

        if getBB:
            self.BB_R=self.__getMat('BB')
        
        if getCC:
            try:
                self.CC_R=1j*self.__getMat('CCab')
            except:
                _CC_R=self.__getMat('CC')
                self.CC_R=1j*(_CC_R[:,:,:,alpha_A,beta_A]-_CC_R[:,:,:,beta_A,alpha_A])

        if getFF:
            try:
                self.FF_R=1j*self.__getMat('FFab')
            except:
                _FF_R=self.__getMat('FF')
                self.FF_R=1j*(_FF_R[:,:,:,alpha_A,beta_A]-_FF_R[:,:,:,beta_A,alpha_A])

        if getSS:
            self.SS_R=self.__getMat('SS')
        cprint ("Reading the system finished successfully",'green', attrs=['bold'])
        
    
    @lazy_property.LazyProperty
    def NKFFTmin(self):
        NKFFTmin=np.ones(3,dtype=int)
        for i in range(3):
            R=self.iRvec[:,i]
            if len(R[R>0])>0: 
                NKFFTmin[i]+=R.max()
            if len(R[R<0])>0: 
                NKFFTmin[i]-=R.min()
        return NKFFTmin

    def __from_tb_file(self,tb_file=None,getAA=False):
        self.seedname=tb_file.split("/")[-1].split("_")[0]
        f=open(tb_file,"r")
        l=f.readline()
        cprint ("reading TB file {0} ( {1} )".format(tb_file,l.strip()),'green', attrs=['bold'])
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice).T
        self.num_wann=int(f.readline())
        nRvec=int(f.readline())
        self.nRvec0=nRvec
        self.spinors=None
        self.Ndegen=[]
        while len(self.Ndegen)<nRvec:
            self.Ndegen+=f.readline().split()
        self.Ndegen=np.array(self.Ndegen,dtype=int)
        
        self.iRvec=[]
        
        self.HH_R=np.zeros( (self.num_wann,self.num_wann,nRvec) ,dtype=complex)
        
        for ir in range(nRvec):
            f.readline()
            self.iRvec.append(f.readline().split())
            hh=np.array( [[f.readline().split()[2:4] 
                             for n in range(self.num_wann)] 
                                for m in range(self.num_wann)],dtype=float).transpose( (1,0,2) )
            self.HH_R[:,:,ir]=(hh[:,:,0]+1j*hh[:,:,1])/self.Ndegen[ir]
        
        self.iRvec=np.array(self.iRvec,dtype=int)
        
        if getAA:
          self.AA_R=np.zeros( (self.num_wann,self.num_wann,nRvec,3) ,dtype=complex)
          for ir in range(nRvec):
            f.readline()
            assert (np.array(f.readline().split(),dtype=int)==self.iRvec[ir]).all()
            aa=np.array( [[f.readline().split()[2:8] 
                             for n in range(self.num_wann)] 
                                for m in range(self.num_wann)],dtype=float)
            self.AA_R[:,:,ir,:]=(aa[:,:,0::2]+1j*aa[:,:,1::2]).transpose( (1,0,2) ) /self.Ndegen[ir]
        else: 
            self.AA_R = None
        
        f.close()

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)
        cprint ("Reading the system finished successfully",'green', attrs=['bold'])

        
    @property
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
    def __init__(self,iRvec,num_wann,lines):
        nRvec=iRvec.shape[0]
        self.num_wann=num_wann
        self._iRvec_new=dict()
        n_nonzero=np.array([l.split()[-1] for l in lines[:nRvec]],dtype=int)
        lines=lines[nRvec:]
        nonzero=[]
        for ir,nnz in enumerate(n_nonzero):
            map1r=map_1R(lines[:nnz],iRvec[ir])
            for iw in range(num_wann):
                for jw in range(num_wann):
                    self._add_star(ir,map1r(iw,jw),iw,jw)
            lines=lines[nnz:]
        self._iRvec_ordered=sorted(self._iRvec_new)
        for ir  in range(nRvec):
            chsum=0
            for irnew in self._iRvec_new:
                if ir in self._iRvec_new[irnew]:
                    chsum+=self._iRvec_new[irnew][ir]
            chsum=np.abs(chsum-np.ones( (num_wann,num_wann) )).sum() 
            if chsum>1e-12: print ("WARNING: Check sum for {0} : {1}".format(ir,chsum))


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
        
    def __call__(self,matrix):
        ndim=len(matrix.shape)-3
        num_wann=matrix.shape[0]
        reshaper=(num_wann,num_wann)+(1,)*ndim
        print ("check:",matrix.shape,reshaper,ndim)
        matrix_new=np.array([ sum(matrix[:,:,ir]*self._iRvec_new[irvecnew][ir].reshape(reshaper)
                                  for ir in self._iRvec_new[irvecnew] ) 
                                       for irvecnew in self._iRvec_ordered]).transpose( (1,2,0)+tuple(range(3,3+ndim)) )

        assert ( np.abs(matrix_new.sum(axis=2)-matrix.sum(axis=2)).max()<1e-12)
        return matrix_new
             


import numpy as np
from scipy.io import FortranFile
import multiprocessing 



class CheckPoint():

    def __init__(self,seedname):
        seedname=seedname.strip()
        FIN=FortranFile(seedname+'.chk','r')
        readint   = lambda : FIN.read_record('i4')
        readfloat = lambda : FIN.read_record('f8')
        def readcomplex():
            a=readfloat()
            return a[::2]+1j*a[1::2]
        readstr   = lambda : "".join(c.decode('ascii')  for c in FIN.read_record('c') ) 

#        print ( 'Reading restart information from file '+seedname+'.chk :')
        self.comment=readstr() 
#        print (self.comment)
        self.num_bands          = readint()[0]
        num_exclude_bands       = readint()[0]
        self.exclude_bands      = readint()
#        print(self.exclude_bands,num_exclude_bands)
        assert  len(self.exclude_bands)==num_exclude_bands
        self.real_lattice=readfloat().reshape( (3 ,3),order='F')
#        print ("real lattice:\n",self.real_lattice)
        self.recip_lattice=readfloat().reshape( (3 ,3),order='F')
#        print ("reciprocal lattice:\n",self.recip_lattice)
        assert np.linalg.norm(self.real_lattice.dot(self.recip_lattice.T)/(2*np.pi)-np.eye(3)) < 1e-14
        self.num_kpts = readint()[0]
        self.mp_grid  = readint()
        assert len(self.mp_grid)==3
        assert self.num_kpts==np.prod(self.mp_grid)
        self.kpt_latt=readfloat().reshape( (self.num_kpts,3))
        self.nntot    = readint()[0]
        self.num_wann = readint()[0]
        self.checkpoint=readstr().strip()
#        print ("checkpoint:"+self.checkpoint)
        self.have_disentangled=bool(readint()[0])
        if self.have_disentangled:
            self.omega_invariant=readfloat()[0]
            lwindow=np.array( readint().reshape( (self.num_kpts,self.num_bands)),dtype=bool )
            ndimwin=readint()
            u_matrix_opt=readcomplex().reshape( (self.num_kpts,self.num_wann,self.num_bands) )
        u_matrix=readcomplex().reshape( (self.num_kpts,self.num_wann,self.num_wann) )
        m_matrix=readcomplex().reshape( (self.num_kpts,self.nntot,self.num_wann,self.num_wann) )
        if self.have_disentangled:
            self.v_matrix=[u.dot(u_opt[:,:nd]) for u,u_opt,nd in 
                                    zip(u_matrix,u_matrix_opt,ndimwin)]
        else:
            self.v_matrix=[u  for u in u_matrix ] 
        self.wannier_centers=readfloat().reshape((self.num_wann,3))
        self.wannier_spreads=readfloat().reshape((self.num_wann))
        self.win_min = np.array( [np.where(lwin)[0].min() for lwin in lwindow] )
        self.win_max = np.array( [wm+nd for wm,nd in zip(self.win_min,ndimwin)]) 
#        print ("win_min : ",self.win_min)

#        for i in range(10):
#            print (FIN.read_record('i4')



    def get_HH_q(self,eig):
        assert (eig.NK,eig.NB)==(self.num_kpts,self.num_bands)
        HH_q=np.array([ (V.conj()*E[None,wmin:wmax]).dot(V.T) 
                        for V,E,wmin,wmax in zip(self.v_matrix,eig.data,self.win_min,self.win_max) ])
        print ('HH_q:',HH_q.shape)
        return 0.5*(HH_q+HH_q.transpose(0,2,1).conj())


    def get_AA_q(self,mmn,eig=None):  # if eig is present - it is BB_q 
        mmn.set_bk(self)
        AA_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3) ,dtype=complex)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb=mmn.neighbours[ik,ib]
                data=mmn.data[ik,ib,self.win_min[ik]:self.win_max[ik],self.win_min[iknb]:self.win_max[iknb]]
                if eig is not None:
                    data*=eig.data[ik,:,None]
                v1=self.v_matrix[ik]
                v2=self.v_matrix[iknb]
                AA_q[ik]+=1.j*v1.dot(data).dot(v2.T)[:,:,None]*mmn.wk[ik,ib]*mmn.bk_cart[ik,ib,None,None,:]
        AA_q=0.5*(AA_q+AA_q.transpose( (0,2,1,3) ).conj())
        print ('AA_q:',AA_q.shape)
        return AA_q



class MMN():

    def __init__(self,seedname,num_proc=4):
        f_mmn_in=open(seedname+".mmn","r").readlines()
        print ("reading {}.mmn: ".format(seedname)+f_mmn_in[0])
        s=f_mmn_in[1]
        self.NB,self.NK,self.NNB=np.array(s.split(),dtype=int)
        self.data=np.zeros( (self.NK,self.NNB,self.NB,self.NB), dtype=complex )
        headstring=np.array([s.split() for s in f_mmn_in[2::1+self.NB**2] ]
                    ,dtype=int).reshape(self.NK,self.NNB,5)
        self.G=headstring[:,:,2:]
        self.neighbours=headstring[:,:,1]-1
        assert np.all( headstring[:,:,0]-1==np.arange(self.NK)[:,None])
        block=1+self.NB*self.NB
        allmmn=( f_mmn_in[3+j*block:2+(j+1)*block]  for j in range(self.NNB*self.NK) )
        p=multiprocessing.Pool(num_proc)
        self.data= np.array(p.map(str2arraymmn,allmmn)).reshape(self.NK,self.NNB,self.NB,self.NB)

    def set_bk(self,chk):
      try :
        self.bk
        self.wk
        return
      except:
        bk_latt=np.array(np.round( [(chk.kpt_latt[nbrs]-chk.kpt_latt+G)*chk.mp_grid[None,:] for nbrs,G in zip(self.neighbours.T,self.G.transpose(1,0,2))] ).transpose(1,0,2),dtype=int)
#        print (self.bk_latt)
        bk_latt_unique=np.array([b for b in set(tuple(bk) for bk in bk_latt.reshape(-1,3))],dtype=int)
        assert len(bk_latt_unique)==self.NNB
        bk_cart_unique=bk_latt_unique.dot(chk.recip_lattice/chk.mp_grid[:,None])
        bk_cart_unique_length=np.linalg.norm(bk_cart_unique,axis=1)
        srt=np.argsort(bk_cart_unique_length)
        bk_latt_unique=bk_latt_unique[srt]
        bk_cart_unique=bk_cart_unique[srt]
        bk_cart_unique_length=bk_cart_unique_length[srt]
        brd=[0,]+list(np.where(bk_cart_unique_length[1:]-bk_cart_unique_length[:-1]>1e-7)[0]+1)+[self.NNB,]
        shell_mat=np.array([ bk_cart_unique[b1:b2].T.dot(bk_cart_unique[b1:b2])  for b1,b2 in zip (brd,brd[1:])])
        shell_mat_line=shell_mat.reshape(-1,9)
        u,s,v=np.linalg.svd(shell_mat_line,full_matrices=False)
        s=1./s
        weight_shell=np.eye(3).reshape(1,-1).dot(v.T.dot(np.diag(s)).dot(u)).reshape(-1)
        assert np.linalg.norm(sum(w*m for w,m in zip(weight_shell,shell_mat))-np.eye(3))<1e-7
        weight=np.array([w/(b2-b1) for w,b1,b2 in zip(weight_shell,brd,brd[1:]) for i in range(b1,b2)])
        print ("weight=",weight)
        weight_dict  = {tuple(bk):w for bk,w in zip(bk_latt_unique,weight) }
        bk_cart_dict = {tuple(bk):bkcart for bk,bkcart in zip(bk_latt_unique,bk_cart_unique) }
        self.bk_cart=np.array([[bk_cart_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
        self.wk     =np.array([[ weight_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
        


def str2arraymmn(A):
    a=np.array([l.split() for l in A],dtype=float)
    n=int(round(np.sqrt(a.shape[0])))
    return (a[:,0]+1j*a[:,1]).reshape((n,n))


class EIG():
    def __init__(self,seedname):
        data=np.loadtxt(seedname+".eig")
        NB=int(round(data[:,0].max()))
        NK=int(round(data[:,1].max()))
        data=data.reshape(NK,NB,3)
        assert np.linalg.norm(data[:,:,0]-1-np.arange(NB)[None,:])<1e-15
        assert np.linalg.norm(data[:,:,1]-1-np.arange(NK)[:,None])<1e-15
        self.data=data[:,:,2]

    @property 
    def  NK(self):
        return self.data.shape[0]

    @property 
    def  NB(self):
        return self.data.shape[1]
            


seedname='wannier90'
chk=CheckPoint(seedname)
mmn=MMN(seedname)
eig=EIG(seedname)
chk.get_HH_q(eig)
chk.get_AA_q(mmn)
chk.get_AA_q(mmn,eig)

#mmn.set_bk(chk.mp_grid,chk.kpt_latt,chk.recip_lattice)
#chk.get_HH_q(eig)

#mmn2=MMN('Fe')

#eig=np.random.random( (chk.num_kpts,chk.num_bands) )
#HH_q=chk.get_HH_q(eig)
#print (HH_q.shape)



#        ik2mp = lambda ik : ( 
#                ik%(mp_grid[1]*mp_grid[2]) , (ik//mp_grid[0])%mp_grid[1] , 
#                         ik//(mp_grid[0]*mp_grid[1]) )
#        mp2ik = lambda mp: ik*

#    def get_gauge_overlap_matrix( XX , k1,hermitian=False):
#        k1=np.arange(self.num_kpts)
#        if kk2 is None: k2=k1
#        XXq=np.array([ V1.conj().dot(X[wmin1:wmax1,wmin2:wmax2]).dot(V2.T) 
#                        for V1,wmin1,wmax1,X,V2,wmin2,wmax2 in zip(
#                              self.v_matrix[k1],self.wmin[k1],self.wmax[k1],
#                                  XX,
#                              self.v_matrix[k2],self.wmin[k2],self.wmax[k2]) ])
#        if hermitian:
#            XXq=0.5*(XXq+XXq.transpose( (0,2,1) ).conj())
#        return XXq
