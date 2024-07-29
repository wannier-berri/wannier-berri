import numpy as np


class SpreadFunctional:

    """
    calculate the spread functional

    Parameters
    ---------
    w : numpy.ndarray(NNB)
        the weights
    bk : numpy.ndarray(NNB, 3)
        the nearest neighbour vectors (cartesian)
    Mmn : numpy.ndarray(NK,NNB,nW,nW)
        the Mmn matrix
    neigh : list of list of int or numpy.ndarray(NK, NNB)
        the neighbours of each k-point (indices in the full BZ list)
    """

    def __init__(self, w, bk, neigh, Mmn):
        self.w = np.copy(w)
        self.bk = np.copy(bk)
        self.neigh = np.copy(neigh)
        self.Mmn = np.copy(Mmn)

    def get_wcc(self, U):
        NK = len(self.neigh)
        NW = U[0].shape[1]
        NNB = len(self.neigh[0])
        UT = [u.T.conj() for u in U]
        Mmn_loc = np.array([[UT[ik].dot(self.Mmn[ik][ib].dot(U[ikb])) for ib, ikb in enumerate(neigh)]
                            for ik, neigh in enumerate(self.neigh)])
        phinkb = -np.array([[[np.angle(mnnb[n, n]) for n in range(NW)] for mnnb in mnn] for mnn in Mmn_loc])
        return np.array(sum(self.w[ib] * phinkb[ik, ib][:, None] * self.bk[ib]  for ib in range(NNB) for ik in range(NK)))
        
    def __call__(self, U, wcc):
        """
        calculate the spread functional

        Parameters
        ----------
        U : list of numpy.ndarray(NK, NB, NW)
            the U matrix for all k-points

        Returns
        -------
        dict
            Omega_D : float
                the diagonal spread functional
            Omega_OD : float
                the off-diagonal spread functional
            Omega_I : float
                the total spread functional
            Omega_tot : float
                the total spread functional
            wannier_centers : numpy.ndarray(NW,3)
                the wannier centers
        """
        # return self.spread_wcc(U, wcc)
        return self.spread_MV(U, wcc)

    def spread_MV(self, U, wcc):
        """
        calculate the spread functional using the Marzari-Vanderbilt method
        """

        # NK = len(U)
        NK = len(self.neigh)
        NW = U[0].shape[1]
        NNB = len(self.neigh[0])
        UT = [u.T.conj() for u in U]
        Mmn_loc = np.array([[UT[ik].dot(self.Mmn[ik][ib].dot(U[ikb])) for ib, ikb in enumerate(neigh)]
                            for ik, neigh in enumerate(self.neigh)])
        phinkb = -np.array([[[np.angle(mnnb[n, n]) for n in range(NW)] for mnnb in mnn] for mnn in Mmn_loc])
        Mmn2 = abs(Mmn_loc)**2
        Mmn2[:, :, np.arange(NW), np.arange(NW)] = 0
        Mmn2 = Mmn2.sum(axis=(2, 3))
        Omega_OD = sum(self.w[ib] * Mmn2[ik, ib] for ib in range(NNB) for ik in range(NK))
        absnkb = np.array([[[abs(mnnb[n, n]) for n in range(NW)] for mnnb in mnn] for mnn in Mmn_loc])
        rm2sum = np.linalg.norm(wcc)**2
        # print ("wannier centers\n", rn)
        Omega_D = sum(self.w[ib] * (phinkb[ik, ib, n] - self.bk[ib] @ wcc[n])**2 for n in range(NW) for ib in range(NNB) for ik in range(NK))
        Omega_tot = sum(self.w[ib] * (-absnkb[ik, ib, n]**2 + phinkb[ik, ib, n]**2)  for n in range(NW) for ib in range(NNB) for ik in range(NK))
        Omega_tot += NW * np.sum(self.w[:, None], axis=(0, 1)) - rm2sum
        Omega_I = Omega_tot - Omega_OD - Omega_D
        return dict(Omega_D=Omega_D, Omega_OD=Omega_OD, Omega_I=Omega_I, Omega_tot=Omega_tot)
        # return Omega_D, Omega_OD, Omega_I, Omega_tot

    def spread_wcc(self, U, wcc):
        """
        calculate the spread functional

        Parameters
        ----------
        U : list of numpy.ndarray(NK, NB, NW)
            the U matrix for all k-points
        rn : numpy.ndarray(NW,3)
            the wannier centers

        Returns
        -------
        dict
            Omega_D : float
                the diagonal spread functional
            Omega_OD : float
                the off-diagonal spread functional
            Omega_I : float
                the total spread functional
            Omega_tot : float
                the total spread functional
            wannier_centers : numpy.ndarray(NW,3)
                the wannier centers
        """
        # NK = len(U)
        NK = len(self.neigh)
        NW = U[0].shape[1]
        NNB = len(self.neigh[0])
        wcc_phase = np.exp( 1j * wcc @ self.bk.T)[None,:,:]
        # wcc_phase = np.ones(wcc_phase.shape, dtype=complex)

        UT = [u.T.conj() for u in U]

        Mmn_loc = np.array([[UT[ik].dot(self.Mmn[ik][ib].dot(U[ikb]*wcc_phase[:,:,ib])) 
                            for ib, ikb in enumerate(neigh)]
                                for ik, neigh in enumerate(self.neigh)])
        bkwk = self.bk[:,:] * self.w[ :, None]
        bk_dot_bk = bkwk @ bkwk.T* NK
        # print ( "bk_dot_bk", bk_dot_bk, "Mmn_loc", Mmn_loc.shape)	
        print ( "wcc", np.round(wcc,5))
        print ("wcc_phase", np.round(wcc_phase,5))  
        spreads =  np.einsum('kann,ab,kbnn->n', Mmn_loc.conj(), bk_dot_bk, Mmn_loc)
        result = dict()
        result["Omega_tot"] = sum(spreads)
        for i, spread in enumerate(spreads):
            result[f"Omega_{i}"] = spread
        return result