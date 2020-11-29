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
#import billiard as multiprocessing 
import multiprocessing 
from .__utility import str2bool, alpha_A, beta_A, iterate3dpm
from colorama import init
from termcolor import cprint 
from time import time
from itertools import islice
import gc

readstr  = lambda F : "".join(c.decode('ascii')  for c in F.read_record('c') ).strip() 

class CheckPoint():

    def __init__(self,seedname):
        t0=time()
        seedname=seedname.strip()
        FIN=FortranFile(seedname+'.chk','r')
        readint   = lambda : FIN.read_record('i4')
        readfloat = lambda : FIN.read_record('f8')
        def readcomplex():
            a=readfloat()
            return a[::2]+1j*a[1::2]

        print ( 'Reading restart information from file '+seedname+'.chk :')
        self.comment=readstr(FIN) 
        self.num_bands          = readint()[0]
        num_exclude_bands       = readint()[0]
        self.exclude_bands      = readint()
        assert  len(self.exclude_bands)==num_exclude_bands
        self.real_lattice=readfloat().reshape( (3 ,3),order='F')
        self.recip_lattice=readfloat().reshape( (3 ,3),order='F')
        assert np.linalg.norm(self.real_lattice.dot(self.recip_lattice.T)/(2*np.pi)-np.eye(3)) < 1e-14
        self.num_kpts = readint()[0]
        self.mp_grid  = readint()
        assert len(self.mp_grid)==3
        assert self.num_kpts==np.prod(self.mp_grid)
        self.kpt_latt=readfloat().reshape( (self.num_kpts,3))
        self.nntot    = readint()[0]
        self.num_wann = readint()[0]
        self.checkpoint=readstr(FIN)
        self.have_disentangled=bool(readint()[0])
        if self.have_disentangled:
            self.omega_invariant=readfloat()[0]
            lwindow=np.array( readint().reshape( (self.num_kpts,self.num_bands)),dtype=bool )
            ndimwin=readint()
            u_matrix_opt=readcomplex().reshape( (self.num_kpts,self.num_wann,self.num_bands) )
            self.win_min = np.array( [np.where(lwin)[0].min() for lwin in lwindow] )
            self.win_max = np.array( [wm+nd for wm,nd in zip(self.win_min,ndimwin)])
        else:
            self.win_min = np.array( [0]*self.num_kpts )
            self.win_max = np.array( [self.num_wann]*self.num_kpts) 
            
        u_matrix=readcomplex().reshape( (self.num_kpts,self.num_wann,self.num_wann) )
        m_matrix=readcomplex().reshape( (self.num_kpts,self.nntot,self.num_wann,self.num_wann) )
        if self.have_disentangled:
            self.v_matrix=[u.dot(u_opt[:,:nd]) for u,u_opt,nd in 
                                    zip(u_matrix,u_matrix_opt,ndimwin)]
        else:
            self.v_matrix=[u  for u in u_matrix ] 
        self.wannier_centres=readfloat().reshape((self.num_wann,3))
        self.wannier_spreads=readfloat().reshape((self.num_wann))
        del u_matrix,m_matrix
        gc.collect()
        print ("Time to read .chk : {}".format(time()-t0))

    def wannier_gauge(self,mat,ik1,ik2):
        # data should be of form NBxNBx ...   - any form later
        if len(mat.shape)==1:
            mat=np.diag(mat)
        assert mat.shape[:2]==(self.num_bands,)*2
        mat=mat[self.win_min[ik1]:self.win_max[ik1],self.win_min[ik2]:self.win_max[ik2]]
        v1=self.v_matrix[ik1].conj()
        v2=self.v_matrix[ik2].T
        return  np.tensordot( np.tensordot(v1,mat,axes=(1,0)),v2,axes=(1,0)).transpose((0,-1,)+tuple(range(1,mat.ndim-1)))

    def get_HH_q(self,eig):
        assert (eig.NK,eig.NB)==(self.num_kpts,self.num_bands)
        HH_q=np.array([ self.wannier_gauge(E,ik,ik)  for ik,E in enumerate(eig.data) ])
        return 0.5*(HH_q+HH_q.transpose(0,2,1).conj())


    def get_SS_q(self,spn):
        assert (spn.NK,spn.NB)==(self.num_kpts,self.num_bands)
        SS_q=np.array([ self.wannier_gauge(S,ik,ik)  for ik,S in enumerate(spn.data) ]) 
        return 0.5*(SS_q+SS_q.transpose(0,2,1,3).conj())

    def get_AA_q(self,mmn,eig=None,transl_inv=False):  # if eig is present - it is BB_q 
        if transl_inv and (eig is not None):
           raise RuntimeError("transl_inv cannot be used to obtain BB")
        mmn.set_bk(self)
        AA_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3) ,dtype=complex)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb=mmn.neighbours[ik,ib]
                data=mmn.data[ik,ib]
                if eig is not None:
                    data*=eig.data[ik,:,None]
                AAW=self.wannier_gauge(data,ik,iknb)
                AA_q_ik=1.j*AAW[:,:,None]*mmn.wk[ik,ib]*mmn.bk_cart[ik,ib,None,None,:]
                if transl_inv:
                    AA_q_ik[range(self.num_wann),range(self.num_wann)]=-np.log(AAW.diagonal()).imag[:,None]*mmn.wk[ik,ib]*mmn.bk_cart[ik,ib,None,:]
                AA_q[ik]+=AA_q_ik
        if eig is None:
            AA_q=0.5*(AA_q+AA_q.transpose( (0,2,1,3) ).conj())
        return AA_q

    def get_CC_q(self,uhu,mmn):  # if eig is present - it is BB_q 
        mmn.set_bk(self)
        assert uhu.NNB==mmn.NNB
        CC_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3) ,dtype=complex)
        for ik in range(self.num_kpts):
          for ib1 in range(mmn.NNB):
            iknb1=mmn.neighbours[ik,ib1]
            for ib2 in range(mmn.NNB):
              iknb2=mmn.neighbours[ik,ib2]
              data=uhu.data[ik,ib1,ib2]
              CC_q[ik]+=1.j*self.wannier_gauge(data,iknb1,iknb2)[:,:,None]* (
                   mmn.wk[ik,ib1]*mmn.wk[ik,ib2]* (
               mmn.bk_cart[ik,ib1,alpha_A]* mmn.bk_cart[ik,ib2,beta_A ] - 
               mmn.bk_cart[ik,ib1,beta_A] * mmn.bk_cart[ik,ib2,alpha_A]  )  )[None,None,:]
        CC_q=0.5*(CC_q+CC_q.transpose( (0,2,1,3) ).conj())
        return CC_q

    def get_SA_q(self,siu,mmn):
        mmn.set_bk(self)
        SA_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3,3) ,dtype=complex)
        assert siu.NNB==mmn.NNB 
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb=mmn.neighbours[ik,ib]
                SAW=self.wannier_gauge(siu.data[ik,ib],ik,iknb)
                SA_q_ik=1.j*SAW[:,:,None,:]*mmn.wk[ik,ib]*mmn.bk_cart[ik,ib,None,None,:,None]
                SA_q[ik]+=SA_q_ik
        return SA_q

    def get_SHA_q(self,shu,mmn):
        mmn.set_bk(self)
        SHA_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3,3) ,dtype=complex)
        assert shu.NNB==mmn.NNB
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb=mmn.neighbours[ik,ib]
                SHAW=self.wannier_gauge(shu.data[ik,ib],ik,iknb)
                SHA_q_ik=1.j*SHAW[:,:,None,:]*mmn.wk[ik,ib]*mmn.bk_cart[ik,ib,None,None,:,None]
                SHA_q[ik]+=SHA_q_ik
        #SHA_q=0.5*(SHA_q+SHA_q.transpose( (0,2,1,3,4) ).conj())
        #for ik in range(self.num_kpts):
        #    print(ik, SHA_q[ik,17,16,2,0])
        return SHA_q

    def get_SR_q(self,spn,mmn):
        mmn.set_bk(self)
        SR_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3,3) ,dtype=complex)
        assert (spn.NK,spn.NB)==(self.num_kpts,self.num_bands)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb=mmn.neighbours[ik,ib]
                for i in range(3):
                    SM_i=spn.data[ik,:,:,i].dot(mmn.data[ik,ib,:,:])
                    SRW=self.wannier_gauge(SM_i,ik,iknb)-self.wannier_gauge(spn.data[ik,:,:,i],ik,ik)
                    SR_q[ik,:,:,:,i]+=1.j*SRW[:,:,None]*mmn.wk[ik,ib]*mmn.bk_cart[ik,ib,None,None,:]
        return SR_q
    
    def get_SH_q(self,spn,eig):
        SH_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3) ,dtype=complex)
        assert (spn.NK,spn.NB)==(self.num_kpts,self.num_bands)
        for ik in range(self.num_kpts):
            for i in range(3):
                SH_q[ik,:,:,i]=self.wannier_gauge(spn.data[ik,:,:,i]*eig.data[ik,None,:],ik,ik)
        return SH_q
        
    def get_SHR_q(self,spn,mmn,eig):
        mmn.set_bk(self)
        SHR_q=np.zeros( (self.num_kpts,self.num_wann,self.num_wann,3,3) ,dtype=complex)
        assert (spn.NK,spn.NB)==(self.num_kpts,self.num_bands)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb=mmn.neighbours[ik,ib]
                for i in range(3):
                    SH_i=spn.data[ik,:,:,i]*eig.data[ik,None,:]
                    SHM_i=SH_i.dot(mmn.data[ik,ib])
                    SHRW=self.wannier_gauge(SHM_i,ik,iknb)-self.wannier_gauge(SH_i,ik,ik)
                    SHR_q[ik,:,:,:,i]+=1.j*SHRW[:,:,None]*mmn.wk[ik,ib]*mmn.bk_cart[ik,ib,None,None,:]
        return SHR_q


class W90_data():
    @property
    def n_neighb(self):
        return 0

    @property 
    def  NK(self):
        return self.data.shape[0]

    @property 
    def  NB(self):
        return self.data.shape[1+self.n_neighb]

    @property 
    def  NNB(self):
        if self.n_neighb>0:
            return self.data.shape[1]
        else:
            return 0

def convert(A):
    return np.array([l.split() for l in A],dtype=float)

class MMN(W90_data):

    @property
    def n_neighb(self):
        return 1


    def __init__(self,seedname,npar=multiprocessing.cpu_count()):
        t0=time()
        f_mmn_in=open(seedname+".mmn","r")
        print ("reading {}.mmn: ".format(seedname)+f_mmn_in.readline())
        NB,NK,NNB=np.array(f_mmn_in.readline().split(),dtype=int)
        self.data=np.zeros( (NK,NNB,NB,NB), dtype=complex )
        block=1+self.NB*self.NB
        data=[]
        headstring=[]
        mult=4
        if npar>0 :
            pool=multiprocessing.Pool(npar)
        for j in range(0,NNB*NK,npar*mult):
            x=list(islice(f_mmn_in, int(block*npar*mult)))
            print ("mmn: read {} lines".format(len(x)))
            if len(x)==0 : break
            headstring+=x[::block]
            y=[x[i*block+1:(i+1)*block] for i in range(npar*mult) if (i+1)*block<=len(x)]
            if npar>0:
                data+=pool.map(convert,y)
            else:
                data+=[convert(z) for z in y]

        if npar>0 : 
            pool.close()
        f_mmn_in.close()
        t1=time()
        data=[d[:,0]+1j*d[:,1] for d in data]
        self.data=np.array(data).reshape(self.NK,self.NNB,self.NB,self.NB).transpose((0,1,3,2))
        headstring=np.array([s.split() for s in headstring  ] ,dtype=int).reshape(self.NK,self.NNB,5)
        assert np.all( headstring[:,:,0]-1==np.arange(self.NK)[:,None])
        self.neighbours=headstring[:,:,1]-1
        self.G=headstring[:,:,2:]
        t2=time()
        print ("Time for MMN.__init__() : {} , read : {} , headstring {}".format(t2-t0,t1-t0,t2-t1))

    def set_bk(self,chk):
      try :
        self.bk
        self.wk
        return
      except:
        bk_latt=np.array(np.round( [(chk.kpt_latt[nbrs]-chk.kpt_latt+G)*chk.mp_grid[None,:] for nbrs,G in zip(self.neighbours.T,self.G.transpose(1,0,2))] ).transpose(1,0,2),dtype=int)
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
#        print ("u,s,v=",u,s,v)
#        print("check svd : ",u.dot(np.diag(s)).dot(v)-shell_mat_line)
        s=1./s
        weight_shell=np.eye(3).reshape(1,-1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
        check_eye=sum(w*m for w,m in zip(weight_shell,shell_mat))
        tol=np.linalg.norm(check_eye-np.eye(3))
        if tol>1e-5 :
            raise RuntimeError("Error while determining shell weights. the following matrix :\n {} \n failed to be identity by an error of {} Further debug informstion :  \n bk_latt_unique={} \n bk_cart_unique={} \n bk_cart_unique_length={}\nshell_mat={}\weight_shell={}\n".format(
                      check_eye,tol, bk_latt_unique,bk_cart_unique,bk_cart_unique_length,shell_mat,weight_shell))
        weight=np.array([w for w,b1,b2 in zip(weight_shell,brd,brd[1:]) for i in range(b1,b2)])
        weight_dict  = {tuple(bk):w for bk,w in zip(bk_latt_unique,weight) }
        bk_cart_dict = {tuple(bk):bkcart for bk,bkcart in zip(bk_latt_unique,bk_cart_unique) }
        self.bk_cart=np.array([[bk_cart_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
        self.wk     =np.array([[ weight_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
        

class EIG(W90_data):
    def __init__(self,seedname):
        data=np.loadtxt(seedname+".eig")
        NB=int(round(data[:,0].max()))
        NK=int(round(data[:,1].max()))
        data=data.reshape(NK,NB,3)
        assert np.linalg.norm(data[:,:,0]-1-np.arange(NB)[None,:])<1e-15
        assert np.linalg.norm(data[:,:,1]-1-np.arange(NK)[:,None])<1e-15
        self.data=data[:,:,2]

            
class SPN(W90_data):
    def __init__(self,seedname='wannier90',formatted=False):
        print ("----------\n SPN  \n---------\n")
        spn_formatted_in=formatted
        if  spn_formatted_in:
            f_spn_in = open(seedname+".spn", 'r')
            SPNheader=f_spn_in.readline().strip()
            nbnd,NK=(int(x) for x in f_spn_in.readline().split())
        else:
            f_spn_in = FortranFile(seedname+".spn", 'r')
            SPNheader=(f_spn_in.read_record(dtype='c')) 
            nbnd,NK=f_spn_in.read_record(dtype=np.int32)
            SPNheader="".join(a.decode('ascii') for a in SPNheader)

        print ("reading {}.spn : {}".format(seedname,SPNheader))

        indm,indn=np.tril_indices(nbnd)
        self.data=np.zeros( (NK,nbnd,nbnd,3),dtype=complex)

        for ik in range(NK):
            A=np.zeros((3,nbnd,nbnd),dtype=np.complex)
            if spn_formatted_in:
                tmp=np.array( [f_spn_in.readline().split() for i in xrange (3*nbnd*(nbnd+1)/2)  ],dtype=float)
                tmp=tmp[:,0]+1.j*tmp[:,1]
            else:
                tmp=f_spn_in.read_record(dtype=np.complex)
            A[:,indn,indm]=tmp.reshape(3,nbnd*(nbnd+1)//2,order='F')
            check=np.einsum('ijj->',np.abs(A.imag))
            A[:,indm,indn]=A[:,indn,indm].conj()
            if check> 1e-10:
                raise RuntimeError ( "REAL DIAG CHECK FAILED : {0}".format(check) )
            self.data[ik]=A.transpose(1,2,0)
        print ("----------\n SPN OK  \n---------\n")


class UXU(W90_data):  # uHu or uIu
    @property
    def n_neighb(self):
        return 2

    #def __init__(self,seedname='wannier90',formatted=False,suffix='uHu'):
    def __init__(self,seedname='wannier90',formatted=False,suffix='uHu'):
        print ("----------\n  {0}   \n---------".format(suffix))
        print('formatted == {}'.format(formatted))
        if formatted:
            f_uXu_in = open(seedname+"."+suffix, 'r')
            header=f_uXu_in.readline().strip() 
            NB,NK,NNB =(int(x) for x in f_uXu_in.readline().split())
        else:
            f_uXu_in = FortranFile(seedname+"."+suffix, 'r')
            header=readstr(f_uXu_in)
            NB,NK,NNB=f_uXu_in.read_record('i4')

        print ("reading {}.{} : <{}>".format(seedname,suffix,header))

        
        self.data=np.zeros( (NK,NNB,NNB,NB,NB),dtype=complex )
        if formatted:
            tmp=np.array( [f_uXu_in.readline().split() for i in range(NK*NNB*NNB*NB*NB)  ],dtype=float)
            tmp_conj=tmp[:,0]+1.j*tmp[:,1]
            self.data=tmp_conj.reshape(NK,NNB,NNB,NB,NB)
        else:
            for ik in range(NK):
#            print ("k-point {} of {}".format( ik+1,NK))
                for ib2 in range(NNB):
                    for ib1 in range(NNB):
                        tmp=f_uXu_in.read_record('f8').reshape((2,NB,NB),order='F').transpose(2,1,0) 
                        self.data[ik,ib1,ib2]=tmp[:,:,0]+1j*tmp[:,:,1]
        print ("----------\n {0} OK  \n---------\n".format(suffix))
        f_uXu_in.close()


class UHU(UXU):  
    def __init__(self,seedname='wannier90',formatted=False):
        super(UHU, self).__init__(seedname=seedname,formatted=formatted,suffix='uHu' )

class UIU(UXU):  
    def __init__(self,seedname='wannier90',formatted=False):
        super(UIU, self).__init__(seedname=seedname,formatted=formatted,suffix='uIu' )


class SXU(W90_data):  # sHu or sIu
    @property
    def n_neighb(self):
        return 1

    def __init__(self,seedname='wannier90',formatted=False,suffix='sHu'):
        print ("----------\n  {0}   \n---------".format(suffix))

        if formatted:
            f_sXu_in = open(seedname+"."+suffix, 'r')
            header=f_sXu_in.readline().strip() 
            NB,NK,NNB =(int(x) for x in f_sXu_in.readline().split())
        else:
            f_sXu_in = FortranFile(seedname+"."+suffix, 'r')
            header=readstr(f_sXu_in)
            NB,NK,NNB=   f_sXu_in.read_record('i4')

        print ("reading {}.{} : <{}>".format(seedname,suffix,header))

        self.data=np.zeros( (NK,NNB,NB,NB,3),dtype=complex )

        for ik in range(NK):
#            print ("k-point {} of {}".format( ik+1,NK))
            for ib2 in range(NNB):
                for ipol in range(3):
                   tmp=f_sXu_in.read_record('f8').reshape((2,NB,NB),order='F').transpose(2,1,0)
                   self.data[ik,ib2,:,:,ipol]=tmp[:,:,0]+1j*tmp[:,:,1]
        print ("----------\n {0} OK  \n---------\n".format(suffix))
        f_sXu_in.close()


class SIU(SXU):
    def __init__(self,seedname='wannier90',formatted=False):
        super(SIU, self).__init__(seedname=seedname,formatted=formatted,suffix='sIu' )

class SHU(SXU):
    def __init__(self,seedname='wannier90',formatted=False):
        super(SHU, self).__init__(seedname=seedname,formatted=formatted,suffix='sHu' )



def set_bk(neighbours,G,NNB):
        bk_latt=np.array(np.round( [(chk.kpt_latt[nbrs]-chk.kpt_latt+G)*chk.mp_grid[None,:] for nbrs,G in zip(self.neighbours.T,self.G.transpose(1,0,2))] ).transpose(1,0,2),dtype=int)
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
#        print ("u,s,v=",u,s,v)
#        print("check svd : ",u.dot(np.diag(s)).dot(v)-shell_mat_line)
        s=1./s
        weight_shell=np.eye(3).reshape(1,-1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
        check_eye=sum(w*m for w,m in zip(weight_shell,shell_mat))
        tol=np.linalg.norm(check_eye-np.eye(3))
        if tol>1e-5 :
            raise RuntimeError("Error while determining shell weights. the following matrix :\n {} \n failed to be identity by an error of {} Further debug informstion :  \n bk_latt_unique={} \n bk_cart_unique={} \n bk_cart_unique_length={}\nshell_mat={}\weight_shell={}\n".format(
                      check_eye,tol, bk_latt_unique,bk_cart_unique,bk_cart_unique_length,shell_mat,weight_shell))
        weight=np.array([w for w,b1,b2 in zip(weight_shell,brd,brd[1:]) for i in range(b1,b2)])
        weight_dict  = {tuple(bk):w for bk,w in zip(bk_latt_unique,weight) }
        bk_cart_dict = {tuple(bk):bkcart for bk,bkcart in zip(bk_latt_unique,bk_cart_unique) }
        self.bk_cart=np.array([[bk_cart_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
        self.wk     =np.array([[ weight_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
