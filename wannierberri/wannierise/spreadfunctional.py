import numpy as np


class SpreadFunctional:
    
    """
    calculate the spread functional

    Parameters
    ----------
    w : numpy.ndarray(NKirr, NNB)
        the weights
    bk : numpy.ndarray(NKirr, NNB, 3)
        the nearest neighbour vectors (cartesian)
    Mmn : numpy.ndarray(NKirr,NNB,nW,nW)
        the Mmn matrix
    U : numpy.ndarray(NKirr,nW,nW)
        the U matrix

    Returns
    -------
    float
        the spread functional
    """
    def __init__(self, w, bk, neigh, Mmn):
        self.w = np.copy(w)
        self.bk = np.copy(bk)
        self.neigh = np.copy(neigh)
        self.Mmn = np.copy(Mmn)

    def __call__(self, U):
        # NK = len(U)
        NK = len(self.neigh)
        NW = U[0].shape[1]
        NNB = len(self.neigh[0])
        UT = [u.T.conj() for u in U]
        Mmn_loc = np.array([[UT[ik].dot(self.Mmn[ik][ib].dot(U[ikb])) for ib, ikb in enumerate(neigh)] 
                            for ik, neigh in enumerate(self.neigh)])
        Mmn2 = abs(Mmn_loc)**2
        Mmn2[:,:,np.arange(NW),np.arange(NW)] = 0
        Mmn2 = Mmn2.sum(axis=(2,3))
        Omega_OD = sum(self.w[ik,ib] * Mmn2[ik,ib] for ib in range(NNB) for ik in range(NK))
        phinkb = -np.array([[[np.angle(mnnb[n,n]) for n in range(NW)] for mnnb in mnn] for mnn in Mmn_loc])
        absnkb = np.array([[[abs(mnnb[n,n]) for n in range(NW)] for mnnb in mnn] for mnn in Mmn_loc])
        rn = np.array(sum(self.w[ik,ib] * phinkb[ik,ib][:,None]*self.bk[ik,ib]  for ib in range(NNB) for ik in range(NK)))
        rm2sum =np.linalg.norm(rn)**2
        # print ("wannier centers\n", rn)
        Omega_D = sum(self.w[ik,ib] * (phinkb[ik,ib,n] - self.bk[ik][ib] @ rn[n])**2 for n in range(NW) for ib in range(NNB) for ik in range(NK))
        Omega_tot = sum(self.w[ik,ib] * (-absnkb[ik,ib,n]**2+phinkb[ik,ib,n]**2)  for n in range(NW) for ib in range(NNB) for ik in range(NK))
        Omega_tot += NW*np.sum(self.w,axis=(0,1))-rm2sum
        Omega_I  =Omega_tot - Omega_OD - Omega_D
        return dict(Omega_D=Omega_D, Omega_OD=Omega_OD, Omega_I=Omega_I, Omega_tot=Omega_tot)
        # return Omega_D, Omega_OD, Omega_I, Omega_tot