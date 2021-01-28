from .__w90_files import MMN,EIG,AMN,WIN,DMN
from copy import deepcopy
import numpy as np

DEGEN_THRESH=1e-2  # for safity - avoid splitting (almost) degenerate states between free/frozen  inner/outer subspaces  (probably too much)

class WannierModel():
    """A class to describe all input files of wannier90, and to construct the Wannier functions 
     via disentanglement procedure"""
# todo :  rotatre uHu and spn
# todo : create a model from this
# todo : symmetry

    def __init__(self,seedname="wannier90",sitesym=False):
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
#        eig.write(seedname+"-copy")
        mmn=MMN(seedname)
#        mmn.write(seedname+"-copy")
        amn=AMN(seedname)
#        amn.write(seedname+"-copy")
        assert eig.NK==amn.NK==mmn.NK
        assert eig.NB>=amn.NB
        assert eig.NB>=mmn.NB
        self.NK=eig.NK
        self.NW=amn.NW
        self.NB=mmn.NB
        self.recip_lattice=2*np.pi*np.linalg.inv(self.real_lattice)
        mmn.set_bk(mp_grid=self.mp_grid,kpt_latt=self.kpt_latt,recip_lattice=self.recip_lattice)
        self.bk_cart=mmn.bk_cart
        self.wb=mmn.wk
        self.G=mmn.G
        self.neighbours=mmn.neighbours
        self.Mmn=mmn.data
        self.Amn=amn.data
        self.Eig=eig.data
        self.win_index=[np.arange(eig.NB)]*self.NK
        if sitesym:
            self.Dmn=DMN(seedname,num_wann=self.NW)
        else:
            self.Dmn=DMN(None,num_wann=self.NW,num_bands=self.NB,nkpt=self.NK)




    # TODO : allow k-dependent window (can it be useful?)
    def apply_outer_window(self,
                     win_min=-np.Inf,
                     win_max= np.Inf ):
        "Excludes the bands from outside the outer window"

        def win_index_nondegen(ik,thresh=DEGEN_THRESH):
            "define the indices of the selected bands, making sure that degenerate bands were not split"
            E=self.Eig[ik]
            ind=np.where( ( E<=win_max)*(E>=win_min) )[0]
            while ind[0]>0 and E[ind[0]]-E[ind[0]-1]<thresh:
                ind=[ind[0]-1]+ind
            while ind[0]<len(E) and E[ind[-1]+1]-E[ind[-1]]<thresh:
                ind=ind+[ind[-1]+1]
            return ind

        win_index_irr=[win_index_nondegen(ik) for ik in self.Dmn.kptirr]
        self.Dmn.apply_outer_window(win_index_irr)
        win_index=[win_index_irr[ik] for ik in self.Dmn.kpt2kptirr]
        self.Eig=[E[ind] for E, ind in zip(self.Eig,win_index)]
        self.Mmn=[[self.Mmn[ik][ib][win_index[ik],:][:,win_index[ikb]] for ib,ikb in enumerate(self.neighbours[ik])] for ik in range(self.NK)]
        self.Amn=[self.Amn[ik][win_index[ik],:] for ik in range(self.NK)]

    # TODO : allow k-dependent window (can it be useful?)
    def disentangle(self,
                 froz_min=np.Inf,
                 froz_max=-np.Inf,
                 num_iter=100,
                 conv_tol=1e-9,
                 mix_ratio=0.5
                 ):

        assert 0<mix_ratio<=1
        def frozen_nondegen(ik,thresh=DEGEN_THRESH):
            """define the indices of the frozen bands, making sure that degenerate bands were not split 
            (unfreeze the degenerate bands together) """
            E=self.Eig[ik]
            ind=np.where( ( E<=froz_max)*(E>=froz_min) )[0]
            while len(ind)>0 and ind[0]>0 and E[ind[0]]-E[ind[0]-1]<thresh:
                del(ind[0])  
            while len(ind)>0 and ind[0]<len(E) and E[ind[-1]+1]-E[ind[-1]]<thresh:
                del(ind[-1])
            froz=np.zeros(E.shape,dtype=bool)
            froz[ind]=True
            return froz

        frozen_irr=[frozen_nondegen(ik) for ik in self.Dmn.kptirr]
        self.frozen=np.array([ frozen_irr[ik] for ik in self.Dmn.kpt2kptirr ])
        self.free= np.array([ np.logical_not(frozen) for frozen in self.frozen])
        self.Dmn.set_free(frozen_irr)
        self.nBfree=np.array([ np.sum(free) for free in self.free ])
        self.nWfree=np.array([ self.NW-np.sum(frozen) for frozen in self.frozen])
#        print ("Bfree:",self.nBfree)
#        print ("Wfree:",self.nWfree)
        # initial guess : eq 27 of SMV2001
        irr=self.Dmn.kptirr
#        print ('irr  = ',repr(irr),type(irr))
#        print ('free = ',repr(self.free))
        U_opt_free_irr=self.get_max_eig(  [ self.Amn[ik][free,:].dot(self.Amn[ik][free,:].T.conj()) 
                        for ik,free in zip(irr,self.free[irr])]  ,self.nWfree[irr],self.nBfree[irr]) # nBfee x nWfree marrices
        # initial guess : eq 27 of SMV2001
        U_opt_free=self.symmetrize_U_opt(U_opt_free_irr,free=True)


        Mmn_FF=self.Mmn_Free_Frozen(self.Mmn,self.free,self.frozen,self.neighbours,self.wb,self.NW)

        def calc_Z(Mmn_loc,U=None):
        # TODO : symmetrize (if needed) 
            if U is None: 
               Mmn_loc_opt=[Mmn_loc[ik] for ik in self.Dmn.kptirr]
            else:
               mmnff=Mmn_FF('free','free')
               mmnff=[mmnff[ik] for ik in self.Dmn.kptirr]
               Mmn_loc_opt=[[Mmn[ib].dot(U[ikb]) for ib,ikb in enumerate(neigh)] for Mmn,neigh in zip(mmnff,self.neighbours[irr])]
            return [sum(wb*mmn.dot(mmn.T.conj()) for wb,mmn in zip(wbk,Mmn)) for wbk,Mmn in zip(self.wb,Mmn_loc_opt) ]

        Z_frozen=calc_Z(Mmn_FF('free','frozen')) #  only for irreducible
        

#        print ( '+---------------------------------------------------------------------+<-- DIS\n'+
#                '|  Iter     Omega_I(i-1)      Omega_I(i)      Delta (frac.)    Time   |<-- DIS\n'+
#                '+---------------------------------------------------------------------+<-- DIS'  )

        for i_iter in range(num_iter):
            Z=[(z+zfr)  for z,zfr in zip(calc_Z(Mmn_FF('free','free'),U_opt_free),Z_frozen) ]  # only for irreducible
            if i_iter>0 and mix_ratio<1:
                Z=[ (mix_ratio*z + (1-mix_ratio)*zo) for z,zo in zip(Z,Z_old) ]  #  only for irreducible
            U_opt_free_irr=self.get_max_eig(Z,self.nWfree[irr],self.nBfree[irr]) #  only for irreducible
            U_opt_free=self.symmetrize_U_opt(U_opt_free_irr,free=True)
            Omega_I=Mmn_FF.Omega_I(U_opt_free)
            print ("iteration {:4d}".format(i_iter)+" Omega_I= "+"  ".join("{:15.10f}".format(x) for x in Omega_I)+" tot =","{:15.10f}".format(sum(Omega_I)))
            Z_old=deepcopy(Z)

        U_opt_full_irr=[]
        for ik in range(self.Dmn.kptirr):
           nband=self.Eig[ik].shape[0]
           U=np.zeros((nband,self.NW),dtype=complex)
           nfrozen=sum(self.frozen[ik])
           nfree=sum(self.free[ik])
#           print("ik={}, nfree={}, nfrozen={}, nband={}, \nfrozen={} \n".format(ik,nfree,nfrozen,nband,self.frozen[ik]))
           assert nfree+nfrozen==nband
           assert nfrozen<=self.NW, "number of frozen bands {} at k-point {} is greater than number of wannier functions {}".format(nfrozen,ik+1,self.NW)
           U[self.frozen[ik] , range( nfrozen) ] = 1.
#           print(U[self.free[ik]   , nfrozen : ].shape, U_opt_free[ik].shape)
           U[self.free[ik]   , nfrozen : ] = U_opt_free[ik]
#           print ("ik={}, U=\n{}\n".format(ik+1,U))
           Z,D,V=np.linalg.svd(U.T.conj().dot(self.Amn[ik]))
           U_opt_full_irr.append(U.dot(Z.dot(V)))
        U_opt_full=self.symmetrize_U_opt(U_opt_full_irr,free=False)


       # now rotating to the optimized space
        self.Hmn=[]
        for ik in range(self.NK):
            U=U_opt_full[ik]
            Ud=U.T.conj()
            # hamiltonian is not diagonal anymore
            self.Hmn.append(Ud.dot(np.diag(self.Eig[ik])).dot(U))
            self.Amn[ik]=Ud.dot(self.Amn[ik])
            self.Mmn[ik]=[Ud.dot(M).dot(U_opt_full[ibk]) for M,ibk in zip (self.Mmn[ik],self.neighbours[ik])]

    def symmetrize_U_opt(self,U_opt_free_irr,free=False):
        # TODO : first symmetrize by the little group
        # Now distribute to reducible points
        d_band=self.Dmn.d_band_free if free else self.Dmn.d_band
        print (self.Dmn.kpt2kptirr_sym.shape,self.Dmn.kpt2kptirr.shape)
#        for isym  in  self.Dmn.kpt2kptirr_sym:
#            print (isym,d_band[isym].shape,U_opt_free_irr.shape,self.Dmn.D_wann_dag.shape)

        for isym,ikirr in zip(self.Dmn.kpt2kptirr_sym,self.Dmn.kpt2kptirr)  :
            print (isym,ikirr)
            print ("    ",d_band[isym].shape)
            print ("    ",U_opt_free_irr[ikirr].shape)
            print ("    ",self.Dmn.D_wann_dag[isym].shape)
            print ("        ",d_band[isym][ikirr].shape)
#            print ("        ",U_opt_free_irr[ikirr].shape)
#            print ("        ",self.Dmn.D_wann_dag[isym][ikirr].shape)
        exit()
        U_opt_free=[d_band[isym][ikirr] @ U_opt_free_irr[ikirr] @ self.Dmn.D_wann_dag[isym][ikirr] for isym,ikirr in zip(self.Dmn.kpt2kptirr_sym,self.Dmn.kpt2kptirr)  ]

           
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


    def write_files(self,seedname="wannier90"):
        "Write the disentangled files , where num_wann==num_bands"
        Eig=[]
        Uham=[]
        Amn=[]
        Mmn=[]
        for H in self.Hmn:
            E,U=np.linalg.eigh(H)
            Eig.append(E)
            Uham.append(U)
        EIG(data=Eig).write(seedname)
        for ik in range(self.NK):
            U=Uham[ik]
            Ud=U.T.conj()
            Amn.append(Ud.dot(self.Amn[ik]))
            Mmn.append([Ud.dot(M).dot(Uham[ibk]) for M,ibk in zip (self.Mmn[ik],self.neighbours[ik])])
        MMN(data=Mmn,G=self.G,bk=self.bk_cart,wk=self.wb,neighbours=self.neighbours).write(seedname)
        AMN(data=Amn).write(seedname)

    def get_max_eig(self,matrix,nvec,nBfree):
        """ return the nvec column-eigenvectors of matrix with maximal eigenvalues. 
        Both matrix and nvec are lists by k-points with arbitrary size of matrices"""
#        print ("getting maximal vectors of: \n {}".format(matrix))
        assert len(matrix)==len(nvec)==len(nBfree)
        assert np.all([m.shape[0]==m.shape[1] for m in matrix])
        assert np.all([m.shape[0]>=nv for m,nv in zip(matrix,nvec)]), "nvec={}, m.shape={}".format(nvec,[m.shape for m in matrix])
        EV=[np.linalg.eigh(M) for M in matrix]
        return [ ev[1][:,np.argsort(ev[0])[nf-nv:nf]] for ev,nv,nf  in zip(EV,nvec,nBfree) ] 

    class Mmn_Free_Frozen():
        # TODO : make use of irreducible kpoints (maybe)
        """ a class to store and call the Mmn matrix between/inside the free and frozen subspaces, as well as to calculate the streads"""
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



