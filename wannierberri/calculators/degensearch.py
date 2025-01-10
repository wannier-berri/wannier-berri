import warnings
import numpy as np

from ..result.degenresult import DegenResult, DegenResultEmpty, clean_repeat

from .calculator import Calculator


class DegenSearcherKP:

    def __init__(self, H0, H1, H2=None,
                 kmax=1, kstep_max=0.1, mult_grad=1,
                 iband=0,
                 degen_thresh=1e-6,
                 grad_thresh=1e-14):
        self.num_bands = H0.shape[0]
        assert H0.shape == (self.num_bands, self.num_bands)
        self.ndim = H1.shape[2]
        assert self.ndim in (1, 2, 3)
        assert H1.shape == (self.num_bands, self.num_bands, self.ndim)
        self.H0 = H0[None, :, :]  # reserve 0th axes for kpoint index
        self.H1 = H1
        if H2 is not None:
            self.hasH2 = True
            assert H2.shape == (self.num_bands, self.num_bands, self.ndim, self.ndim)
            self.H2 = H2
        else:
            self.hasH2 = False
        assert iband >= 0, 'iband must be non-negative'
        assert iband + 1 < self.num_bands, f'iband={iband} must be less than number of bands-1 (num_bands={self.num_bands})'
        self.iband = iband
        assert mult_grad > 0, 'mult_grad must be positive'
        self.mult_grad = mult_grad
        assert kmax > 0, 'kmax must be positive'
        self.kmax = kmax
        assert kstep_max > 0, 'kstep_max must be positive'
        self.kstep_max = kstep_max
        assert degen_thresh > 0, 'degen_thresh must be positive'
        self.degen_tol = degen_thresh
        assert grad_thresh > 0, 'grad_thresh must be positive'
        self.grad_thresh = grad_thresh

    def grad(self, kpoints):
        grad = self.H1[None, :, :, :] * np.ones((kpoints.shape[0], 1, 1, 1))
        if self.hasH2:
            grad = grad + 2 * np.einsum("ka,mnab->kmna", kpoints, self.H2)
        return grad

    def hamiltonian(self, kpoints):
        H = self.H0 + np.einsum("ka,mna->kmn", kpoints, self.H1)
        if self.hasH2:
            H = H + np.einsum('ka,mnab,kb->kmn', kpoints, self.H2, kpoints)
        return H

    def step(self, kpoints):
        return_kpoints = []

        H = self.hamiltonian(kpoints)
        gradH = self.grad(kpoints)
        E, V = np.linalg.eigh(H)
        E = E[:, self.iband:self.iband + 2]
        dE = E[:, 1] - E[:, 0]
        Eav = (E[:, 1] + E[:, 0]) / 2
        del E
        # Select kpoints with degenerate bands
        sel = dE < self.degen_tol
        for k, e, de in zip(kpoints[sel], Eav[sel], dE[sel]):
            return_kpoints.append((k, e, de))  # for gradient we put None here
        sel = np.logical_not(sel)
        Eav = Eav[sel]
        V = V[sel, :, :]
        dE = dE[sel]
        kpoints = kpoints[sel]
        gradH = gradH[sel]

        # calculate gradient of dE
        V = V[:, :, self.iband:self.iband + 2]
        gradE = np.einsum('kml,kmna,knl->kal', V.conj(), gradH, V).real
        graddE = gradE[:, :, 1] - gradE[:, :, 0]
        graddElen = np.linalg.norm(graddE, axis=1)
        # remove poins with small gradient
        sel = graddElen > self.grad_thresh
        kpoints = kpoints[sel]
        graddE = graddE[sel]
        graddElen = graddElen[sel]
        dE = dE[sel]

        # now set the new kpoints
        dk = -graddE * dE[:, None] / graddElen[:, None]**2 * self.mult_grad
        # limit the step size
        dklen = np.linalg.norm(dk, axis=1)
        sel = dklen > self.kstep_max
        dk[sel] *= self.kstep_max / dklen[sel][:, None]
        kpoints_new = kpoints + dk
        sel = np.linalg.norm(kpoints_new, axis=1) < self.kmax
        return kpoints_new[sel], return_kpoints

    def find_degen_points(self, kpoints, max_num_iter=1000):
        return_points = []
        for i in range(max_num_iter):
            kpoints, _ = self.step(kpoints)
            return_points += _
            if len(kpoints) == 0:
                # print (f"Converged in {i} iterations")
                break
        if len(kpoints) > 0:
            warnings.warn(f"{len(kpoints)} kpoints did not converge")
        return np.array([list(k) + [e, de] for k, e, de in return_points])


    def start_random(self, num_start_points=100, include_zero=True):
        if include_zero:
            return np.vstack(([[0] * self.ndim], self.start_random(num_start_points=num_start_points, include_zero=False)))
        if self.ndim == 1:
            return (np.random.random(num_start_points) - 0.5) * 2 * self.kmax
        r = np.random.random(num_start_points) * self.kmax
        phi = np.random.random(num_start_points) * 2 * np.pi
        kx, ky = np.cos(phi), np.sin(phi)
        if self.ndim == 2:
            return np.array([kx, ky]).T * r
        elif self.ndim == 3:
            theta = np.random.random(num_start_points) * np.pi
            sintheta = np.sin(theta)
            return r[:, None] * np.array([sintheta * kx, sintheta * ky, np.cos(theta)]).T


class DegenSearcher(Calculator):

    def __init__(self, iband, thresh=1e-9, gap=1,
                 kmax=1, kstep_max=0.1,
                 resolution = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.iband = iband
        self.thresh = thresh
        self.gap = gap
        self.kmax = kmax
        self.kstep_max = kstep_max
        self.resolution = resolution

    def call_1k(self, ik, iband, E_K, H1, k0, r):
        E = E_K[ik]
        # first, find the group of bands well separated by a gap
        for i in range(iband - 1, -1, -1):
            if E[i + 1] - E[i] > self.gap:
                imin = i + 1
                break
        else:
            imin = 0
        for i in range(iband + 2, len(E)):
            if E[i] - E[i - 1] > self.gap:
                imax = i
                break
        else:
            imax = len(E)
        # print (f"out of energies {E} for bands {iband},{iband+1} included {imin}:{imax}")
        inn = np.arange(imin, imax)
        out = np.concatenate((np.arange(0, imin), np.arange(imax, len(E))))
        h0 = np.diag(E[inn])
        h1 = H1.nn(ik, inn, out)
        searcher = DegenSearcherKP(h0, h1, iband=iband - imin, degen_thresh=self.thresh,
                                   kmax=r, kstep_max=self.kstep_max * r,)
        degen_kpoints = searcher.find_degen_points(searcher.start_random())
        if len(degen_kpoints) == 0:
            return None
        else:
            # print (f"degenerate points found: {degen_kpoints}")
            degen_kpoints[:, :3] += k0[None, :]
            return clean_repeat(degen_kpoints, resolution=self.resolution)


    def __call__(self, data_K):
        H1 = data_K.covariant("Ham", gender=1)
        E_K = data_K.E_K
        degeneracies = []
        kp = data_K.kpoints_all
        r = data_K.Kpoint.radius
        for ik in range(len(E_K)):
            newk = self.call_1k(ik, self.iband, E_K=E_K, H1=H1, k0=kp[ik], r=r)
            if newk is not None:
                degeneracies.append(newk)
        if len(degeneracies) > 0:
            degeneracies = np.vstack(degeneracies)
            # Transform kpoints from cartesian to the reciprocal lattice
            degeneracies[:, :3] = degeneracies[:, :3].dot(np.linalg.inv(data_K.Kpoint.pointgroup.recip_lattice))
            return DegenResult(dic={self.iband: degeneracies}, recip_lattice=data_K.Kpoint.pointgroup.recip_lattice, 
                               save_mode=self.save_mode,
                               resolution=self.resolution)
        else:
            return DegenResultEmpty(save_mode=self.save_mode)
        

