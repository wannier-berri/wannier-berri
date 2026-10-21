"""
Optical conductivity at a finite wave vector q, in the tight-binding approximation.

The Wannier functions are treated as points at their centres (the position operator is diagonal in the Wannier
basis), and the vector potential enters through the Peierls phases of the hoppings. Then sigma_ab(q, omega) follows
from H(R) and the Wannier centres alone, at any q, without an expansion in q.
"""

import numpy as np
from scipy.constants import elementary_charge, hbar, angstrom

from .calculator import Calculator
from ..data_K.data_K_R import Data_K_R
from ..data_K.data_K_soc import Data_K_soc
from ..result import EnergyResult
from ..symmetry.point_symmetry import transform_ident, transform_trans
from ..utility import cached_einsum


def _bond_average(x):
    """(e^{ix} - 1) / (ix) = e^{ix/2} sin(x/2) / (x/2): the average of e^{i q.r} along a bond, with x = q.d (real)."""
    return np.exp(0.5j * x) * np.sinc(x / (2 * np.pi))


def _fermi(E, mu, kBT):
    """occupation, E (k, n), mu (nEF) -> (k, n, nEF)."""
    x = E[:, :, None] - mu[None, None, :]
    if kBT == 0:
        return 1.0 * (x <= 0)
    return 0.5 * (1 - np.tanh(x / (2 * kBT)))


def _rotate(U1, X, U2):
    """matrix elements U1^dagger X U2, X (k, i, j, a) -> (k, n, m, a)."""
    return cached_einsum("kin,kija,kjm->knma", U1.conj(), X, U2)


class OpticalConductivity_q(Calculator):
    r"""Optical conductivity :math:`\sigma_{ab}(\mathbf{q}, \omega)` at a finite wave vector, in the tight-binding
    approximation (S/m).

    The current responds to the field :math:`\mathbf{E}\, e^{i(\mathbf{q}\cdot\mathbf{r} - \omega t)}` as
    :math:`j_a = \sigma_{ab}(\mathbf{q}, \omega) E_b`. The Wannier functions are treated as points at their centres:
    the position operator is diagonal in the Wannier basis, as with ``force_internal_terms_only=True``, and the vector
    potential enters through the Peierls phases of the hoppings :math:`H_{ij}(\mathbf{R})`, along the straight bond
    :math:`\mathbf{d} = \boldsymbol{\tau}_i - \mathbf{R} - \boldsymbol{\tau}_j`. The Kubo formula (paramagnetic and
    diamagnetic terms) is then evaluated exactly in :math:`\mathbf{q}`, from the states at :math:`\mathbf{k}` on the
    grid and at :math:`\mathbf{k}+\mathbf{q}`. The matrix elements of the position operator between different
    Wannier functions (external terms) are not included.

    Take :math:`\mathbf{q}` on the k grid (its reduced coordinates times ``NKFFT*NKdiv`` integers): then
    :math:`\mathbf{k}+\mathbf{q}` is also on the grid, and the result is invariant under electromagnetic gauge
    transformations: a static longitudinal vector potential :math:`\mathbf{A} \propto \mathbf{q}`, which is a pure
    gauge, drives no current (:math:`\sigma_{ab}(\mathbf{q}, \omega) q_b \to 0` for :math:`\omega, \eta \to 0`,
    i.e. charge is conserved). The result also satisfies
    :math:`\sigma_{ab}(\mathbf{q}, \omega)^* = \sigma_{ab}(-\mathbf{q}, -\omega)` on any grid. For other
    :math:`\mathbf{q}`, :math:`\mathbf{k}+\mathbf{q}` is not among the k-points summed over, the paramagnetic and
    diamagnetic terms do not cancel exactly, and these hold only as the grid gets denser; the calculation then stops
    with an error, unless ``allow_q_off_grid=True``. The same applies to K-points added by adaptive refinement and to
    tetrahedron grids. Along non-periodic directions (with no hoppings along them) any component of :math:`\mathbf{q}`
    is allowed.

    At :math:`\mathbf{q} = 0` the result is the optical conductivity in this approximation, including the Drude part
    of a metal. Its first derivative in :math:`\mathbf{q}` gives the spatially dispersive conductivity.

    Parameters
    ----------
    q : array(float), shape (3,)
        wave vector in Cartesian coordinates (1/Angstrom)
    Efermi : array(float)
        Fermi levels (eV)
    omega : array(float)
        frequencies (eV)
    kBT : float
        temperature (eV)
    smr_fixed_width : float
        broadening :math:`\eta` (eV): the frequency is :math:`\omega + i\eta` in all terms (a Lorentzian
        broadening; there is no other smearing type)
    allow_q_off_grid : bool
        allow a :math:`\mathbf{q}` that is not on the k grid (see above)

    Notes
    -----
    The system must be given by its real-space matrix elements (``System_R``), for electrons; systems built from
    spin blocks with spin-orbit coupling (``SystemSOC``) and phonons are not supported. Bands are not selected by
    energy (``Emin``, ``Emax`` are not used): all Wannier bands enter. At a finite ``q`` the result is not symmetric
    under the point group, so ``wannierberri.run`` must be called with ``use_irred_kpt=False`` and
    ``symmetrize=False``. The memory grows as the number of k-points of the FFT grid times the square of the number
    of Wannier functions (several arrays of the current vertices on the whole grid) and times the number of Wannier
    functions and Fermi levels (the occupations), and as the number of R vectors times the square of the number of
    Wannier functions (the vertices in real space). The temporary arrays of the sum over pairs of bands are bounded by
    ``k_chunk_elements``, as long as one k-point fits.

    Output:
        ``[Efermi, omega, a, b]``
    """

    # size of the temporary arrays of the sum over pairs of bands (number of complex elements), which is done in
    # chunks of k-points (at least one)
    k_chunk_elements = 2 ** 22

    def __init__(self, q=(0, 0, 0), Efermi=None, omega=None, kBT=0, smr_fixed_width=0.1, allow_q_off_grid=False,
                 **kwargs):
        super().__init__(**kwargs)
        if Efermi is None or omega is None:
            raise ValueError("Efermi and omega should be given")
        self.q = np.array(q, dtype=float).reshape(3)
        self.Efermi = np.array(Efermi, dtype=float).reshape(-1)
        self.omega = np.array(omega, dtype=float).reshape(-1)
        self.kBT = float(kBT)
        self.eta = float(smr_fixed_width)
        for name in ("q", "Efermi", "omega"):
            x = getattr(self, name)
            if x.size == 0 or not np.all(np.isfinite(x)):
                raise ValueError(f"{name} should be a non-empty array of finite numbers, found {x}")
        if not (np.isfinite(self.kBT) and self.kBT >= 0):
            raise ValueError(f"kBT should be zero or positive, found {kBT}")
        if not (np.isfinite(self.eta) and self.eta > 0):
            raise ValueError(f"smr_fixed_width should be positive, found {smr_fixed_width}")
        self.allow_q_off_grid = bool(allow_q_off_grid)

    def _check_q_on_grid(self, data_K):
        """raise an error if q is not on the k grid, unless allowed (an error, unlike a warning, also reaches the
        user from parallel workers)."""
        if self.allow_q_off_grid or not np.any(self.q):
            return
        hint = ("k+q would not be among the k-points summed over, and gauge invariance would hold only as the grid gets"
                " denser. Take q on the grid, or set allow_q_off_grid=True")
        dense = getattr(getattr(data_K, "grid", None), "dense", None)
        if dense is None:
            raise ValueError(f"OpticalConductivity_q at finite q needs a regular grid (wannierberri.Grid): {hint}")
        if getattr(getattr(data_K, "Kpoint", None), "refinement_level", 0) > 0:
            raise ValueError(f"K-points added by adaptive refinement are not on the k grid: {hint}")
        q_grid = data_K.real_lattice @ self.q / (2 * np.pi) * np.array(dense)
        # along non-periodic directions without hoppings, H(k) does not depend on k, and q is free
        free = ~np.array(data_K.system.periodic, dtype=bool) & np.all(data_K.rvec.iRvec == 0, axis=0)
        if not np.allclose(q_grid[~free], np.round(q_grid[~free]), rtol=0, atol=1e-6):
            raise ValueError(f"q is not on the k grid {tuple(dense)} (in units of its spacing, q = {q_grid}): {hint}")

    @property
    def allow_symmetrize(self):
        return not np.any(self.q)

    def __call__(self, data_K):
        if not isinstance(data_K, Data_K_R):
            raise ValueError("OpticalConductivity_q needs a system given by real-space matrix elements (System_R)")
        if isinstance(data_K, Data_K_soc) or data_K.is_phonon:
            raise ValueError("OpticalConductivity_q does not support systems with spin-orbit blocks (SystemSOC) "
                             "or phonons")
        self._check_q_on_grid(data_K)
        rvec = data_K.rvec
        Ham_R = data_K.Ham_R  # including the phase of the grid shift
        nw = Ham_R.shape[1]
        R = rvec.cRvec
        # the Wannier centres tau, as in the R vectors and in the derivatives of data_K
        tau = rvec.shifts_right_cart
        d = -rvec.cRvec_shifted  # bond tau_i - R - tau_j, (R, i, j, 3)
        qd = d @ self.q
        exp_qR = np.exp(1j * R @ self.q)[:, None, None]
        exp_qtau = np.exp(1j * tau @ self.q)[None, None, :]

        def R_to_k(XX_R):
            """XX_R may be overwritten (as allowed by Rvectors.R_to_k)."""
            return rvec.R_to_k(np.asarray(XX_R, dtype=complex), hermitian=False)

        def eigh(XX_R):
            H = R_to_k(XX_R)
            return np.linalg.eigh(0.5 * (H + H.conj().swapaxes(1, 2)))

        # states at k and k+q, in the gauge of data_K: H(k) = sum_R H(R) e^{ik.R}. Those at k are computed here too,
        # with all bands (data_K.E_K and UU_K may select bands by energy)
        E0, U0 = eigh(Ham_R.copy())
        E1, U1 = eigh(Ham_R * exp_qR)
        # current vertices (Peierls coupling, in units of e/hbar) between k and k+q, and the diamagnetic vertex at k
        X = R_to_k(Ham_R[..., None] * 1j * d * (_bond_average(-qd) / exp_qtau)[..., None])
        Y = R_to_k(Ham_R[..., None] * -1j * d * (_bond_average(qd) * exp_qR * exp_qtau)[..., None])
        X = _rotate(U0, X, U1)
        Y = _rotate(U1, Y, U0)
        # diagonal of the diamagnetic vertex in the bands at k, one component at a time
        D = np.zeros(E0.shape + (3, 3))
        bond2 = np.abs(_bond_average(qd)) ** 2
        for a in range(3):
            for b in range(a, 3):
                Dab = R_to_k(Ham_R * -d[..., a] * d[..., b] * bond2)
                D[..., a, b] = D[..., b, a] = np.diagonal(_rotate(U0, Dab[..., None], U0)[..., 0], axis1=1, axis2=2).real

        f0 = _fermi(E0, self.Efermi, self.kBT)  # (k, n, EF)
        f1 = _fermi(E1, self.Efermi, self.kBT)
        K = np.zeros((len(self.Efermi), len(self.omega), 3, 3), dtype=complex)
        K -= np.einsum("kne,knab->eab", f0, D)[:, None]
        # paramagnetic term: sum over k, n, m of (f_n(k) - f_m(k+q)) / (omega + i eta + E_n(k) - E_m(k+q))
        # X_nm^a Y_mn^b, in chunks of k-points to bound the size of its temporary arrays
        nEF = len(self.Efermi)
        Y = Y.swapaxes(1, 2)  # Y_mn at [k, n, m]
        dE = E0[:, :, None] - E1[:, None, :]
        nchunk = max(1, self.k_chunk_elements // (nw * nw * max(nEF, 9)))
        for start in range(0, len(dE), nchunk):
            sl = slice(start, start + nchunk)
            XY = (X[sl, :, :, :, None] * Y[sl, :, :, None, :]).reshape(-1, 9)
            df = f0[sl, :, None, :] - f1[sl, None, :, :]
            for iw, w in enumerate(self.omega):
                K[:, iw] += ((df / (w + 1j * self.eta + dE[sl, :, :, None])).reshape(-1, nEF).T @ XY).reshape(-1, 3, 3)
        # j = K A, with K in eV Angstrom^2 / Angstrom^3 and A in units of hbar / (e Angstrom); E = i omega A
        K *= elementary_charge ** 2 / hbar / angstrom / (data_K.nk * data_K.cell_volume)
        sigma = K / (1j * (self.omega[None, :, None, None] + 1j * self.eta))
        return EnergyResult([self.Efermi, self.omega], sigma, transformTR=transform_trans,
                            transformInv=transform_ident, save_mode=self.save_mode)
