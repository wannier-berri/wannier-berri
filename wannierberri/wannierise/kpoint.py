from copy import deepcopy
import numpy as np
from .disentanglement import get_max_eig
from .sitesym import orthogonalize


class Kpoint_and_neighbours:
    """ a class to store the data on a single k-point

        Parameters
        ----------
        Mmn : numpy.ndarray(nnb,nb,nb)
            Mmn matrices
        frozen : numpy.ndarray(nb, dtype=bool)
            frozen bands at this k-point
        frozen_nb : list of numpy.ndarray(nnb,nb, dtype=bool)
            frozen bands at neighbours
        wb : numpy.ndarray(nnb, dtype=float)
            weights for each neighbour (b-vector)
        
        Attributes
        ----------
        data : dict((str,str),list of numpy.ndarray(nnb,nf,nf)
            the data for the Mmn matrix for each pair of subspaces (free/frozen)
        spaces : dict
            the spaces (free/frozen)
        neighbours : list of list of tuple
            list of neighbours for each k-point
        wk : list of numpy.ndarray(nnb)
            list of weights for each neighbour (b-vector)
        """

    def __init__(self, Mmn, frozen, frozen_nb,  wb, bk,
                 symmetrizer, ikirr,
                 amn,
                 weight=1,
                 ):
        nnb, nb = Mmn.shape[:2]
        self.nnb = nnb
        self.Mmn = Mmn
        assert Mmn.shape[2] == nb
        assert len(frozen) == nb
        assert frozen_nb.shape == (nnb, nb) 
        self.amn=amn
        self.num_wann = amn.shape[1]
        self.nband = amn.shape[0]
        self.wb = wb
        self.bk = bk
        self.weight=weight

        self.data = {}
        self.frozen = frozen
        self.nfrozen = sum(frozen)
        self.free = np.logical_not(self.frozen)
        self.free_nb = np.logical_not(frozen_nb)
        self.num_bands_free = sum(self.free)
        self.nWfree = self.num_wann - sum(frozen)
        self.NBfree = sum(self.free)
        self.spaces = {'free': self.free_nb, 'frozen': frozen_nb}
        self.freefree = [Mmn[ib][self.free, :][:, self.free_nb[ib]] for ib in range(nnb)]
        self.freefrozen = [Mmn[ib][self.free, :][:, frozen_nb[ib]] for ib in  range(nnb)]
        self.symmmetrize_Z = lambda Z: symmetrizer.symmetrize_Zk(Z, ikirr)
        self.symmetrize_U = lambda U: symmetrizer.symmetrize_U_kirr(U, ikirr)
        self.Zfrozen = self.calc_Z()
        self.Zold = None

        # initialize the U matrix with projections
        amn2 = amn[self.free, :].dot(amn[self.free, :].T.conj())
        self.U_opt_free = get_max_eig(amn2,  self.nWfree, self.NBfree)  # nBfee x nWfree marrices
        self.U_opt_full = self.rotate_to_projections(self.U_opt_free)

    def update(self, U_nb, localise=True, mix_ratio=1.0):
        """
        update the Z matrix

        Parameters
        ----------
        U_nb : numpy.ndarray(nnb, nBfree,nWfree) or None
            the U matrix at neighbouring k-points

        Returns
        -------
        numpy.ndarray(NB, nW)
            the updated U matrix
        """
        assert 0 <= mix_ratio <= 1
        self.U_nb = deepcopy(U_nb)
        U_nb_free = [self.U_nb[ib][f] for ib, f in enumerate(self.free_nb)]
        Z = self.calc_Z(U_nb_free) + self.Zfrozen
        if self.Zold is not None and mix_ratio != 1:
            Z = mix_ratio * Z + (1 - mix_ratio) * self.Zold
        self.Zold=Z
        self.U_opt_free = get_max_eig(Z, self.nWfree, self.num_bands_free)
        if localise:
            U_opt_full = np.zeros((self.nband, self.num_wann), dtype=complex)
            U_opt_full[self.frozen, range(self.nfrozen)] = 1.
            U_opt_full[self.free, self.nfrozen:] = self.U_opt_free
            Mmn_loc = np.array([U_opt_full.T.conj() @ self.Mmn[ib].dot(self.U_nb[ib]) 
                                for ib in range(self.nnb)])
            Mmn_loc_sumb = sum(mm*wb for mm, wb in zip(Mmn_loc, self.wb))/sum(self.wb)
            # print ("Mmn_loc_sumb", Mmn_loc_sumb.shape)
            # symmetrizer.symmetrize_Zk(Mmn_loc_sumb, ikirr)  # this actually makes thing worse, so not using it
            U=np.linalg.inv(Mmn_loc_sumb)
            U=U.T.conj()
            U = orthogonalize(U)
            U_opt_full = U_opt_full.dot(U)
            U_opt_full = orthogonalize(U_opt_full)
            self.U_opt_full = U_opt_full
        else:
            self.U_opt_full = self.rotate_to_projections(self.U_opt_free)
        # self.update_Mmn_opt()
        self.U_opt_full = self.symmetrize_U(self.U_opt_full)
        return self.U_opt_full

        
    def calc_Z(self, U_nb=None):
        """
        calculate the Z matrix for the given U matrix

        Z = \sum_{b} w_{b,k} M_{b,k} M_{b,k}^{\dagger}
        where M_{b,k} = M_{b,k}^{loc} U_{b,k}

        Parameters
        ----------
        U_nb : list of nnb matrices numpy.ndarray(nBfree,nWfree) or None
            the U matrix at neighbouring k-points

        Returns
        -------
        numpy.ndarray(nWfree,nWfree)
            the Z matrix
        """
        if U_nb is None:
            Mmn_loc_opt = self.freefrozen
        else:
            Mmn_loc_opt = [self.freefree[ib].dot(U_nb[ib]) for ib in range(len(self.wb))]
        Z = np.array(sum(wb * mmn.dot(mmn.T.conj()) for wb, mmn in zip(self.wb, Mmn_loc_opt)))
        self.symmmetrize_Z(Z)
        return Z
        

    def rotate_to_projections(self, U_opt_free):
        """
        rotate the U matrix to the projections of the bands
        to better match the initial guess

        Parameters
        ----------
        U_opt_free : numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands and wannier functions

        Returns
        -------
        numpy.ndarray(NB,nW)
            the rotated U matrix for full set of bands and WFs
        """
        U = np.zeros((self.nband, self.num_wann), dtype=complex)
        U[self.frozen, range(self.nfrozen)] = 1.
        U[self.free, self.nfrozen:] = U_opt_free
        ZV = orthogonalize(U.T.conj().dot(self.amn))
        U_out = U.dot(ZV)
        return U_out

#     def update_Mmn_opt(self):
#         """
#         update the Mmn matrix for the optimized U matrix
#         """
#         if self.U_opt_full is None or self.U_nb is None:
#             return
#         UT= self.U_opt_full.T.conj() 
#         self.Mmn_opt = np.array([UT @ mmn @ Ub for mmn, Ub in zip(self.Mmn, self.U_nb)])

#     def update_Unb(self, U_nb=None):
#         """
#         update the U matrix at neighbouring k-points

#         Parameters
#         ----------
#         U_nb : list of nnb matrices numpy.ndarray(nBfree,nWfree)
#             the U matrix at neighbouring k-points
#         """
#         if U_nb is not None:
#             self.U_nb = U_nb
#             self.update_Mmn_opt()

#     def get_centers(self, U_nb=None):
#         """
#         get the centers of the Wannier functions

#         Returns
#         -------
#         numpy.ndarray(nW,3)
#             the centers of the Wannier functions
#         """
#         self.update_Unb(U_nb)
#         rangew= np.arange(self.num_wann)
#         phinb = -np.angle( self.Mmn_opt[:, rangew, rangew]) 
#         return sum(w * phin[:,None]*bk for w, phin, bk in zip(self.wb, phinb, self.bk) )*self.weight 
        
        
#     def getSpreads(self, rn, U_nb=None):
#         """
#         calculate contributions of this k-point to the spread functional
#         and the Wannier Centers

#         Parameters
#         ----------
#         rn : numpy.ndarray(nW,3)
#             the centers of the Wannier functions
#         U_nb : list of nnb matrices numpy.ndarray(NB, NW) or None
#             the U matrix at neighbouring k-points. Also sets self.U_nb to the given value
#             if not provided - the ones from the previous iteration are used

#         Returns
#         -------
#         float
#             the spread functional
#         """
#         self.update_Unb(U_nb)
#         Mmn2 = abs(self.Mmn_opt)**2
#         rangew = np.arange(self.num_wann)
#         Mmn2[:, rangew, rangew] = 0
#         Mmn2 = Mmn2.sum(axis=(1,2))
#         Omega_OD = sum(self.wb * Mmn2)
#         phinb = -np.angle( self.Mmn_opt[:, rangew, rangew]) 
#         absnb =  np.abs( self.Mmn_opt[:, rangew, rangew]) 
#         Omega_D = sum(w * (phin[n] - bk @ rn[n])**2 for n in range(self.num_wann) for w, phin, bk in zip(self.wb, phinb, self.bk))
#         Omega_tot = sum(w * (-absn[n]**2+phin[n]**2)  for n in range(self.num_wann) for w, absn, phin in zip(self.wb, absnb, phinb))
#         Omega_tot += self.num_wann*np.sum(self.wb)
#         return np.array([Omega_D, Omega_OD, Omega_tot])*self.weight
    

# def getSpreads(kpoints, U_opt_full_BZ=None, neighbours_irreducible=None):
#     if U_opt_full_BZ is None:
#         U_nb_list = [None]*len(kpoints)
#     else:
#         print (neighbours_irreducible[0])
#         print (len(U_opt_full_BZ))
#         U_nb_list = [[U_opt_full_BZ[n] for n in neigh] for neigh in neighbours_irreducible]
#     rn = sum(kpoint.get_centers(U_nb) for kpoint, U_nb in zip(kpoints, U_nb_list))
#     print (rn)
#     print ("rn**2", np.linalg.norm(rn)**2)
#     print ("sum of weights", sum(kpoint.weight for kpoint in kpoints))
#     Omega_D, Omega_OD, Omega_tot = sum(kpoint.getSpreads(rn) for kpoint in kpoints)
#     Omega_tot -= np.linalg.norm(rn)**2
#     Omega_I  = Omega_tot - Omega_OD - Omega_D
#     return dict(Omega_D=Omega_D, Omega_OD=Omega_OD, Omega_I=Omega_I, Omega_tot=Omega_tot)


