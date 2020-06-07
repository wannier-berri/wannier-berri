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
import functools

from .__utility import str2bool, alpha_A, beta_A, iterate3dpm, fourier_q_to_R
from colorama import init
from termcolor import cprint 
from .__system import System

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

        self.AA_R=None
        self.BB_R=None
        self.CC_R=None
        self.FF_R=None
        self.SS_R=None

        chk=CheckPoint(self.seedname)
        self.real_lattice=chk.real_lattice
        self.recip_lattice=chk.recip_lattice
        self.iRvec,self.Ndegen=wigner_seitz(chk.mp_grid,self.real_lattice)
        print ("number of R-vectors: {} ; vectrors:\n {}".format(self.iRvec.shape[0], self.iRvec,self.Ndegen))
        self.nRvec0=len(self.iRvec)
        self.num_wann=chk.num_wann

        eig=EIG(seedname)
        if getAA or getBB:
            mmn=MMN(seedname)

        kpt_mp_grid=np.array( np.round(chk.kpt_latt*np.array(chk.mp_grid)[None,:]),dtype=int)%chk.mp_grid
        fourier_q_to_R_loc=functools.partial(fourier_q_to_R, mp_grid=chk.mp_grid,kpt_mp_grid=kpt_mp_grid,iRvec=self.iRvec,ndegen=self.Ndegen,num_proc=num_proc)

        self.HH_R=fourier_q_to_R_loc( chk.get_HH_q(eig) )

        if getAA:
            self.AA_R=fourier_q_to_R_loc(chk.get_AA_q(mmn))

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Minimal Number of K points:", self.NKFFTmin)
        print ("Real-space lattice:\n",self.real_lattice)

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


    @property
    def cRvec(self):
        return self.iRvec.dot(self.real_lattice)


    @property 
    def nRvec(self):
        return self.iRvec.shape[0]


    @lazy_property.LazyProperty
    def cell_volume(self):
        return abs(np.linalg.det(self.real_lattice))





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
        for V,E,wmin,wmax in zip(self.v_matrix,eig.data,self.win_min,self.win_max):
           print("VEww",V.shape,E.shape,wmin,wmax,E)
        HH_q=np.array([ (V*E[None,wmin:wmax]).dot(V.T.conj()) 
                        for V,E,wmin,wmax in zip(self.v_matrix,eig.data,self.win_min,self.win_max) ])
        print ('HH_q:',HH_q.shape)
        print(" check Haermicity of H: ",np.linalg.norm(HH_q-HH_q.transpose(0,2,1).conj()))
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
        print ("EIGENVALUES:\n",self.data)

    @property 
    def  NK(self):
        return self.data.shape[0]

    @property 
    def  NB(self):
        return self.data.shape[1]
            

def wigner_seitz(mp_grid,real_lattice):
    real_metric=real_lattice.T.dot(real_lattice)
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
