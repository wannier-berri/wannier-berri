#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file 'LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------

import numpy as np

from .system import System_k
from .__finite_differences import find_shells, Derivative3D


class SystemKP(System_k):
    r"""
    A system to describe k.p Hamiltonians.
    Technically, it is concodered as a periodic system with k-vector limited to the box defined by the reciprocal lattice.
    a k-vector is always translated to have reduced coordinates between [-1/2,1/2)
    (In future : translate to the 1BZ for non-simple-cubic lattices)

    Parameters
    -----------
    Ham : function
        The Hamiltonian - a function of 3D k-vector that returns a (num_waan x num_wann) Hermitean matrix
    derHam : function
        The cartesian k-derivative of the Hamiltonian - a function of 3D k-vector that returns a (num_waan x num_wann x 3) Hermitean (in mn) matrix.
        If not specified, it will be evaluated numerically from `Ham` with a finite-difference scheme using the `finite_diff_dk` parameter.
    der2Ham : function
        The cartesian second k-derivative of the Hamiltonian - a function of 3D k-vector that returns a (num_waan x num_wann x 3 x 3) Hermitean (in mn) matrix
        If not specified, it will be evaluated numerically from `derHam` with a finite-difference scheme using the `finite_diff_dk` parameter.
    der3Ham : function
        The cartesian second k-derivative of the Hamiltonian - a function of 3D k-vector that returns a (num_waan x num_wann x 3 x 3 x 3) Hermitean (in mn) matrix
        If not specified, it will be evaluated numerically from `der2Ham` with a finite-difference scheme using the `finite_diff_dk` parameter.
    kmax : float
        maximal k-vector (in :math:`\mathring{\rm A}^{-1}`)  In this case the reciprocal lattice is cubic with size  `2*kmax`
    real_lattice : array(3,3)
        the lattice vectors of the model (iif `kmax` is not set)
    recip_lattice : array(3,3)
        the reciprocal lattice vectors of the model (if 'kmax','real_lattice' are not set)
    k_vector_cartesian : bool
        if True, the k-vector in `Ham`, `derHar`, `der2Ham` is given in cartesian coordinates. if `False` - it is in reduced coordinates (Note : the derivatives are always assumed w.r.t. cartesian coordinates)
    finite_diff_dk : float
        defines the dk in taking derivatives (in fraction of the reciprocal lattice)

    Notes
    -----
    * if derivatives of hamiltonian are not provided, they are computed with finite diifferences
    * internally, `self.Ham` and derivatives (`self.Ham_cart`, `self_derHam_cart` ...) accept k in reduced coordinated.
    * the derivatives are always assumed w.r.t. cartesian coordinates


    """

    def __init__(self, Ham, derHam=None, der2Ham=None, der3Ham=None, kmax=1., real_lattice=None, recip_lattice=None,
                 k_vector_cartesian=True, finite_diff_dk=1e-4, **parameters):
        if "name" not in parameters:
            parameters["name"] = "kp"
        super().__init__(force_internal_terms_only=True, **parameters)
        if kmax is not None:
            assert real_lattice is None, "kmax and real_lattice should not be set tigether"
            assert recip_lattice is None, "kmax and recip_lattice should not be set tigether"
            recip_lattice = np.eye(3) * 2 * kmax
        self.set_real_lattice(real_lattice=real_lattice, recip_lattice=recip_lattice)
        self.recip_lattice_inv = np.linalg.inv(self.recip_lattice)
        self.num_wann = Ham([0, 0, 0]).shape[0]

        self.k_to_1BZ = lambda k: (np.array(k) + 0.5) % 1 - 0.5
        # in is not actually 1BZ, rather a box [-0.5:0.5), but for simple cubic its the same.
        # TODO : make a real 1BZ

        self.k_red2cart = lambda k: np.dot(k, self.recip_lattice)
        self.k_cart2red = lambda k: np.dot(k, self.recip_lattice_inv)

        if k_vector_cartesian:
            self.k_ham_from_red = lambda k: self.k_red2cart(self.k_to_1BZ(k))
        else:
            self.k_ham_from_red = lambda k: np.array(self.k_to_1BZ(k))

        self.Ham = lambda k: Ham(self.k_ham_from_red(k))
        assert self.Ham([0, 0, 0]).shape == (self.num_wann,
                                             self.num_wann), f"the shape of Hamiltonian is {self.Ham([0, 0, 0]).shape} not a square ({self.num_wann})"

        self.wk, bki = find_shells(self.recip_lattice * finite_diff_dk)
        self.bk_red = bki * finite_diff_dk
        self.bk_cart = self.bk_red.dot(self.recip_lattice)

        if derHam is not None:
            self.derHam = lambda k: derHam(self.k_ham_from_red(k))
        else:
            self.derHam = Derivative3D(self.Ham, bk_red=self.bk_red, bk_cart=self.bk_cart, wk=self.wk)
        assert self.derHam([0, 0, 0]).shape == (self.num_wann, self.num_wann, 3)

        if der2Ham is not None:
            self.der2Ham = lambda k: der2Ham(self.k_ham_from_red(k))
        else:
            self.der2Ham = Derivative3D(self.derHam, bk_red=self.bk_red, bk_cart=self.bk_cart, wk=self.wk)
        assert self.der2Ham([0, 0, 0]).shape == (self.num_wann, self.num_wann, 3, 3)

        if der3Ham is not None:
            self.der3Ham = lambda k: der3Ham(self.k_ham_from_red(k))
        else:
            self.der3Ham = Derivative3D(self.der2Ham, bk_red=self.bk_red, bk_cart=self.bk_cart, wk=self.wk)
        assert self.der3Ham([0, 0, 0]).shape == (self.num_wann, self.num_wann, 3, 3, 3)

        self.Ham_cart = lambda k: self.Ham(self.k_cart2red(k))
        self.derHam_cart = lambda k: self.derHam(self.k_cart2red(k))
        self.der2Ham_cart = lambda k: self.der2Ham(self.k_cart2red(k))
        self.der3Ham_cart = lambda k: self.der3Ham(self.k_cart2red(k))

        self.set_pointgroup()
        print("Number of wannier functions:", self.num_wann)

    @property
    def NKFFT_recommended(self):
        return np.array([1, 1, 1])
