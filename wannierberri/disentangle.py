from .__w90_files import MMN,EIG,AMN,WIN
from copy import deepcopy
import numpy as np

class WannierModel():
# todo :  write the files
# todo :  rotatre uHu and spn
# todo : create a model from this
# todo : symmetry

    def __init__(self,seedname="wannier90"):
        win=WIN(seedname)
#        win.print_clean()
        self.mp_grid=win.get_param("mp_grid",dtype=int,size=3)
#        print ("mp_grid=",self.mp_grid)
        self.kpt_latt=win.get_param_block("kpoints",dtype=float,shape=(np.prod(self.mp_grid),3))
#        print ("kpoints=",self.kpt_latt)
        self.real_lattice=win.get_param_block("unit_cell_cart",dtype=float,shape=(3,3))
#        print ("real_lattice=",self.real_lattice)
#        exit()
#        real_lattice,mp_grid,kpt_latt=read_from_win(seedname,['real_lattice','mp_grid','kpoints'])
        eig=EIG(seedname)
        mmn=MMN(seedname)
        amn=AMN(seedname)
        assert eig.NK==amn.NK==mmn.NK
        assert eig.NB>=amn.NB
        assert eig.NB>=mmn.NB
        self.NK=eig.NK
        self.NW=amn.NW
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice)
        mmn.set_bk(self.mp_grid,self.kpt_latt,self.recip_lattice)
        self.bk=mmn.bk
        self.wb=mmn.wk
        self.neighbours=mmn.neighbours
        self.Mmn=mmn.data
        self.Amn=amn.data
        self.Eig=eig.data
        self.win_index=[np.arange(eig.NB)]*self.NK




    # TODO : allow k-dependent window (can it be useful?)
    def apply_outer_window(self,
                     win_min=-np.Inf,
                     win_max= np.Inf ):
        self.win_index=[ind[np.where( ( E<=win_max)*(E>=win_min) )[0]] for ind,E in zip (self.win_index,self.Eig)]
        self.Eig=[E[ind] for E, ind in zip(self.Eig,self.win_index)]
        self.Mmn=[[self.Mmn[ik][ib][self.win_index[ik],:][:,self.win_index[ikb]] for ib,ikb in enumerate(self.neighbours[ik])] for ik in range(self.NK)]
        self.Amn=[self.Amn[ik][self.win_index[ik],:] for ik in range(self.NK)]

    # TODO : allow k-dependent window (can it be useful?)
    def disentangle(self,
                 froz_min=np.Inf,
                 froz_max=-np.Inf,
                 num_iter=100,
                 conv_tol=1e-9,
                 mix_ratio=0.5
                 ):

        assert 0<mix_ratio<=1
        self.frozen=[ ( E<=froz_max)*(E>=froz_min) for E in self.Eig]
        self.free= [ np.logical_not(frozen) for frozen in self.frozen]
        self.nBfree=[ np.sum(free) for free in self.free ]
        self.nWfree=[ self.NW-np.sum(frozen) for frozen in self.frozen]
        print ("Bfree:",self.nBfree)
        print ("Wfree:",self.nWfree)
        # initial guess : eq 27 of SMV2001
        U_opt_free=self.get_max_eig(  [ A[free,:].dot(A[free,:].T.conj()) 
                        for A,free in zip(self.Amn,self.free)]  ,self.nWfree) # nBfee x nWfree marrices

        Mmn_FF=self.Mmn_Free_Frozen(self.Mmn,self.free,self.frozen,self.neighbours,self.wb,self.NW)
        def calc_Z(Mmn_loc,U=None):
            if U is None: 
               Mmn_loc_opt=Mmn_loc
            else:
               Mmn_loc_opt=[[Mmn[ib].dot(U[ikb]) for ib,ikb in enumerate(neigh)] for Mmn,neigh in zip(Mmn_FF('free','free'),self.neighbours)]
            return [sum(wb*mmn.dot(mmn.T.conj()) for wb,mmn in zip(wbk,Mmn)) for wbk,Mmn in zip(self.wb,Mmn_loc_opt) ]

        Z_frozen=calc_Z(Mmn_FF('free','frozen'))
#        del Mmn_free_frozen  # we do not need it anymore, I think ))
        

#        print ( '+---------------------------------------------------------------------+<-- DIS\n'+
#                '|  Iter     Omega_I(i-1)      Omega_I(i)      Delta (frac.)    Time   |<-- DIS\n'+
#                '+---------------------------------------------------------------------+<-- DIS'  )

        for i_iter in range(num_iter):
            Z=[(z+zfr)  for z,zfr in zip(calc_Z(Mmn_FF('free','free'),U_opt_free),Z_frozen)]
            if i_iter>0 and mix_ratio<1:
                Z=[ (mix_ratio*z + (1-mix_ratio)*zo) for z,zo in zip(Z,Z_old) ]
            U_opt_free=self.get_max_eig(Z,self.nWfree)
            Omega_I=Mmn_FF.Omega_I(U_opt_free)
            print ("iteration {:4d}".format(i_iter)+" Omega_I= "+"  ".join("{:15.10f}".format(x) for x in Omega_I)+" tot =","{:15.10f}".format(sum(Omega_I)))
            Z_old=deepcopy(Z)

        U_opt_full=[]
        for ik in range(self.NK):
           nband=self.Eig[ik].shape[0]
           U=np.zeros((nband,self.NW),dtype=complex)
           nfrozen=sum(self.frozen[ik])
           U[self.frozen[ik] , range( nfrozen) ] = 1.
           U[self.free[ik]   , nfrozen : ] = U_opt_free[ik]
           Z,D,V=np.linalg.svd(U.T.conj().dot(self.Amn[ik]))
           U=U.dot(Z.dot(V))
           U_opt_full.append(U)

       # now rotating to the optimized space
        for ik in range(self.NK):
            U=U_opt_full[ik]
            Ud=U.T.conj()
            self.Eig[ik]=(Ud.dot(np.diag(self.Eig[ik])).dot(U)).real
            self.Amn[ik]=Ud.dot(self.Amn[ik])
#            for ib,ikb in enumerate (self.neighbours[ik]):
            self.Mmn[ik]=[Ud.dot(M).dot(U_opt_full[ibk]) for M,ibk in zip (self.Mmn[ik],self.neighbours[ik])]
           
    def rotate(self,mat,ik1,ik2):
        # data should be of form NBxNBx ...   - any form later
        if len(mat.shape)==1:
            mat=np.diag(mat)
        assert mat.shape[:2]==(self.num_bands,)*2
        shape=mat.shape[2:]
        mat=mat.reshape(mat.shape[:2]+(-1,)).transpose(2,0,1)
        mat=mat[self.win_min[ik1]:self.win_max[ik1],self.win_min[ik2]:self.win_max[ik2]]
        v1=self.v_matrix[ik1].conj()
        v2=self.v_matrix[ik2].T
        return np.array( [v1.dot(m).dot(v2) for m in mat]).transpose( (1,2,0) ).reshape( (self.num_wann,)*2+shape )


    def get_max_eig(self,matrix,nvec):
        """ return the nvec column-eigenvectors of matrix with maximal eigenvalues. 
        Both matrix and nvec are lists by k-points,s with arbitrary size of matrices"""
#        print ("getting maximal vectors of: \n {}".format(matrix))
        assert len(matrix)==len(nvec)==self.NK
        assert np.all([m.shape[0]==m.shape[1] for m in matrix])
        assert np.all([m.shape[0]>=nv for m,nv in zip(matrix,nvec)]), "nvec={}, m.shape={}".format(nvec,[m.shape for m in matrix])
        EV=[np.linalg.eigh(M) for M in matrix]
        return [ ev[1][:,np.argsort(ev[0])[-nv:]] for ev,nv  in zip(EV,nvec) ] # nBfee x nWfree marrices

    class Mmn_Free_Frozen():
        def __init__(self,Mmn,free,frozen,neighbours,wb,NW):
           self.NK=len(Mmn)
           self.wb=wb
           self.neighbours=neighbours
           self.data={}
           self.spaces={'free':free,'frozen':frozen}
           for s1,sp1 in self.spaces.items():
               for s2,sp2 in self.spaces.items():
                   self.data[(s1,s2)]=[[Mmn[ik][ib][sp1[ik],:][:,sp2[ikb]] 
                             for ib,ikb in enumerate(neigh)] for ik,neigh in enumerate(self.neighbours)]
#           print ("wb=",self.wb)
#           print ( len(mmn) for mmn in self('frozen','frozen'))
           self.Omega_I_0=NW*self.wb[0].sum()
           self.Omega_I_frozen=-sum( sum( wb*np.sum(abs(mmn[ib])**2) for ib,wb in enumerate(WB)) for WB,mmn in zip(self.wb,self('frozen','frozen')))/self.NK

        def __call__(self,space1,space2):
            assert space1 in self.spaces
            assert space2 in self.spaces
            return self.data[(space1,space2)]

        def Omega_I_free_free(self,U_opt_free):
            U=U_opt_free
            Mmn=self('free','free')
            return -sum( self.wb[ik][ib]*np.sum(abs(   U[ik].T.conj().dot(Mmn[ib]).dot(U[ikb])  )**2) 
                        for ik,Mmn in enumerate(Mmn) for ib,ikb in enumerate(self.neighbours[ik])  )/self.NK

        def Omega_I_free_frozen(self,U_opt_free):
            U=U_opt_free
            Mmn=self('free','frozen')
            return -sum( self.wb[ik][ib]*np.sum(abs(   U[ik].T.conj().dot(Mmn[ib])  )**2) 
                        for ik,Mmn in enumerate(Mmn) for ib,ikb in enumerate(self.neighbours[ik])  )/self.NK*2

#        def Omega_I_frozen_free(self,U_opt_free):
#            U=U_opt_free
#            Mmn=self('frozen','free')
#            return -sum( self.wb[ik][ib]*np.sum(abs(  Mmn[ib].dot(U[ikb])  )**2) 
#                        for ik,Mmn in enumerate(Mmn) for ib,ikb in enumerate(self.neighbours[ik])  )/self.NK

        def Omega_I(self,U_opt_free):
            return self.Omega_I_0,self.Omega_I_frozen,self.Omega_I_free_frozen(U_opt_free),self.Omega_I_free_free(U_opt_free)

