from .__wannier_files inport MMN,EIG,AMN,WIN
from copy import deepcopy

class WannierModel():
# todo :  write the files
# todo :  rotatre uHu and spn
# todo : create a model from this
# todo : calculate Omega_I
# todo : symmetry

#todo :  define mp_grid from kpt_latt
#todo : read real_lattice and kpt_latt form .win
    def __init__(self,seedname="wannier90"):
        win=WIN(seedname)
        win.print_clean()
        exit()
#        real_lattice,mp_grid,kpt_latt=read_from_win(seedname,['real_lattice','mp_grid','kpoints'])
        eig=EIG(seedname)
        mmn=MMN(seedname)
        amn=AMN(seedname)
        assert eig.NK==amn.NK==mmn.NK
        self.NK=eig.NK
        self.NW=amn.NW
        self.real_lattice=real_lattice
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice)
        mmn.set_bk(mp_grid,kpt_latt,recip_lattice)
        self.bk=mmn.bk
        self.wk=mmn.wk
        self.mp_grid=mp_grid
        self.neighbours=mmn.neighbours
        self.Mmn=mmn.data
        self.Amn=amn.data
        self.Eig=eig.data

    # TODO : allow k-dependent window (can it be useful?)
    def apply_outer_window(self,
                     win_min=-np.Inf,
                     win_max=np.Inf)
        self.win_index=[np.where( ( E<=win_max)*(E>=win_min) )[0] for E in self.eig]
        self.Eig=[E[ind] for E, ind in zip(self.Eig,self.ind_outer)]
        self.Mmn=[[mmn[ik][ib][self.ind_outer[ik],:][:,self.ind_outer[ikb]] for ikb in self.neighbours[ik]] for ik in range(self.NK)]
        self.Amn=[amn[ik][self.ind_outer[ik],:] for ik in range(self.NK)]

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
        # initial guess : eq 27 of SMV2001
        U_opt_free=self.get_max_eig(  [np.linalg.eigh(A[free,:].dot(A[free,:].T.conj)) 
                        for A,free in zip(self.Amn,self.free)]  ,self.nfree) # nBfee x nWfree marrices

        def calc_Z(Mmn_loc,U=None):
            if U is None: 
               Mmn_loc_opt=Mmn_loc
            else:
               Mmn_loc_opt=[[Mmn[ib].dot(U[ikb]) for ib,ikb in enumerate(neigh)] for Mmn,neigh in zip(Mmn_free,self.neighbours)]
            return [sum(wb*mmn.dot(mmn.T.conj()) for wb,mmn in zip(self.wb,Mmn)) for Mmn in Mmn_loc_opt ]

        Mmn_free_frozen=[[self.Mmn[ik][ib][self.free[ik],:][:,self.frozen[ikb]] 
                             for ib,ikb in enumerate(self.neighbours[ik])] for ik in range(self.NK)] 
        Z_frozen=calc_Z(Mmn_free_frozen)
        del Mmn_free_frozen  # we do not need it anymore, I think ))
        Mmn_free=[[self.Mmn[ik][ib][self.free[ik],:][:,self.free[ikb]] for ib,ikb in enumerate(self.neighbours[ik])] for ik in range(self.NK)] 

        print ( '+---------------------------------------------------------------------+<-- DIS\n'+
                '|  Iter     Omega_I(i-1)      Omega_I(i)      Delta (frac.)    Time   |<-- DIS\n'+
                '+---------------------------------------------------------------------+<-- DIS'  )

        for i_iter in range(num_iter):
            Z=[(z+zfr)  for z,zfr in zip(calc_Z(Mmn_free,U_opt_free),Z_frozen)]
            if i_iter>0 and mix_ratio<1:
                Z=[ (mix_ratio*z + (1-mix_ratio)*zo) for z,zo in zip(Z,Z_old) ]
            U_opt_free=self.get_max_eig(Z,self.nWfree))
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
           self.Eig[ik]=(Ud.dot(np.diag(self.Eig[ik])).dot(U)).real
           self.Amn[ik]=Ud.dot(self.Amn[ik])
           for ib,ikb in enumerate (self.neighbours[ik]):
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
        assert len(matrix)==len(nvec)==self.NK
        assert np.all([m.shape[0]==m.shape[1] for m in matrix])
        assert np.all([m.shape[0]<nv for m,nv in zip(matrix,nvec)])
        EV=[np.linalg.eigh(M)) for M in matrix]
        return [ ev[1][:,np.argsort(ev[0])[-nv:]] for ev,nv  in zip(EV,nvec) ] # nBfee x nWfree marrices


