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
#                                                            #
#------------------------------------------------------------

import numpy as np
import lazy_property
from .sym_wann import SymWann
from wannierberri.__utility import alpha_A, beta_A, iterate3dpm
from wannierberri.symmetry import Symmetry, Group, TimeReversal
from termcolor import cprint
import functools
import multiprocessing


class System():

    default_parameters = {
        'seedname': 'wannier90',
        'frozen_max': -np.Inf,
        'berry': False,
        'morb': False,
        'spin': False,
        'SHCryoo': False,
        'SHCqiao': False,
        'use_ws': True,
        'mp_grid': None,
        'periodic': (True, True, True),
        'use_wcc_phase': False,
        'wannier_centers_cart': None,
        'wannier_centers_reduced': None,
        'npar': None,
        '_getFF': False,
    }

    __doc__ = """
    The base class for describing a system. Does not have its own constructor,
    please use the child classes, e.g  :class:`System_w90` or :class:`System_tb`


    Parameters
    -----------
    seedname : str
        the seedname used in Wannier90. Default: ``{seedname}``
    berry : bool
        set ``True`` to enable evaluation of external term in  Berry connection or Berry curvature and their
        derivatives. Default: ``{berry}``
    spin : bool
        set ``True`` if quantities derived from spin  will be used. Default:``{spin}``
    morb : bool
        set ``True`` to enable calculation of external terms in orbital moment and its derivatives.
        Requires the ``.uHu`` file. Default: ``{morb}``
    periodic : [bool,bool,bool]
        set ``True`` for periodic directions and ``False`` for confined (e.g. slab direction for 2D systems). If less then 3 values provided, the rest are treated as ``False`` . Default : ``{periodic}``
    SHCryoo : bool
        set ``True`` if quantities derived from Ryoo's spin-current elements will be used. (RPS 2019) Default: ``{SHCryoo}``
    SHCqiao : bool
        set ``True`` if quantities derived from Qiao's approximated spin-current elements will be used. (QZYZ 2018). Default: ``{SHCqiao}``
    use_ws : bool
        minimal distance replica selection method :ref:`sec-replica`.  equivalent of ``use_ws_distance`` in Wannier90. Default: ``{use_ws}``
    mp_grid : [nk1,nk2,nk3]
        size of Monkhorst-Pack frid used in ab initio calculation. Needed when `use_ws=True`, and only if it cannot be read from input file, i.e.
        like :class:`.System_tb`, :class:`.System_PythTB`, :class:`.System_TBmodels` ,:class:`.System_fplo`, but only if
        the data originate from ab initio data, not from toy models.
        In contrast, for :class:`.System_w90` and :class:`.System_ase` it is not needed,  but can be provided and will override the original value
        (if you know what and why you are doing)
        Default: ``{mp_grid}``
    frozen_max : float
        position of the upper edge of the frozen window. Used in the evaluation of orbital moment. But not necessary. Default: ``{frozen_max}``
    _getFF : bool
        generate the FF_R matrix based on the uIu file. May be used for only testing so far. Default : ``{_getFF}``
    use_wcc_phase: bool
        using wannier centers in Fourier transform. Correspoinding to Convention I (True), II (False) in Ref."Tight-binding formalism in the context of the PythTB package". Default: ``{use_wcc_phase}``
    wannier_centers_cart :  array-like(num_wann,3)
        use the given wannier_centers (cartesian) instead of those determined automatically. Incompatible with `wannier_centers_reduced`
    wannier_centers_reduced :  array-like(num_wann,3)
        use the given wannier_centers (reduced) instead of those determined automatically. Incompatible with `wannier_centers_cart`
    npar : int
        number of nodes used for parallelization in the `__init__` method. Default: `multiprocessing.cpu_count()`

    Notes
    -----
    + for tight-binding models it is recommended to use `use_wcc_phase = True`. In this case the external terms vanish, and
    + one can safely use `berry=False, morb=False`, and also set `'external_terms':False` in the parameters of the calculation

    """.format(**default_parameters)

    def set_parameters(self, **parameters):

        for param in self.default_parameters:
            if param in parameters:
                vars(self)[param] = parameters[param]
            else:
                vars(self)[param] = self.default_parameters[param]

        for param in parameters:
            if param not in self.default_parameters:
                print(f"WARNING: parameter {param} was passed to data_K, which is not recognised")

        if self.npar is None:
            self.npar = multiprocessing.cpu_count()
        if self.mp_grid is not None:
            self.mp_grip = np.array(self.mp_grid)

        periodic = np.zeros(3, dtype=bool)
        periodic[:len(self.periodic)] = self.periodic
        self.periodic = periodic

    def symmetrize(self, proj, positions, atom_name, soc=False, magmom=None, DFT_code='qe'):
        """
        proj:
            Should be the same with projections card in Wannier90.win.
        positions: list
            Positions of each atom.
        atom_name: list
            Name of each atom.
        magmom: array
            Magnetic moment of each atom.
        """
        XX_R = {'Ham': self.Ham_R}
        for X in ['AA', 'BB', 'CC', 'SS', 'FF', 'SA', 'SHA', 'SR', 'SH', 'SHR']:
            try:
                XX_R[X] = vars(self)[X + '_R']
            except KeyError:
                pass
        symmetrize_wann = SymWann(
            num_wann=self.num_wann,
            lattice=self.real_lattice,
            positions=positions,
            atom_name=atom_name,
            proj=proj,
            iRvec=self.iRvec,
            XX_R=XX_R,
            soc=soc,
            magmom=magmom,
            DFT_code=DFT_code)
        XX_R, self.iRvec = symmetrize_wann.symmetrize()
        for X in XX_R.keys():
            vars(self)[X + '_R'] = XX_R[X]

    def check_periodic(self):
        exclude = np.zeros(self.nRvec, dtype=bool)
        for i, per in enumerate(self.periodic):
            if not per:
                sel = (self.iRvec[:, i] != 0)
                if np.any(sel):
                    print(
                        """WARNING : you declared your system as non-periodic along direction {i}, but there are {nrexcl} of total {nr} R-vectors with R[{i}]!=0.
        They will be excluded, please make sure you know what you are doing """.format(
                            i=i, nrexcl=sum(sel), nr=self.nRvec))
                    exclude[sel] = True
        if np.any(exclude):
            notexclude = np.logical_not(exclude)
            self.iRvec = self.iRvec[notexclude]
            for X in ['Ham', 'AA', 'BB', 'CC', 'SS', 'FF']:
                XR = X + '_R'
                if hasattr(self, XR):
                    vars(self)[XR] = vars(self)[XR][:, :, notexclude]

    @property
    def getAA(self):
        return self.morb or self.berry or self.SHCryoo or self.SHCqiao

    @property
    def getBB(self):
        return self.morb

    @property
    def getCC(self):
        return self.morb

    @property
    def getSS(self):
        return self.spin or self.SHCryoo or self.SHCqiao

    @property
    def getFF(self):
        return self._getFF

    @property
    def getSA(self):
        return self.SHCryoo

    @property
    def getSHA(self):
        return self.SHCryoo

    @property
    def getSHC(self):
        return self.SHCqiao

    def getXX_only_wannier_centers(self, getSS=False):
        """return AA_R, BB_R, CC_R containing only the diagonal matrix elements, evaluated from
        the wannier_centers_cart_auto cariable (tight-binding approximation).
        In practice, it is useless because corresponding terms vanish with use_wcc_phase = True.
        but for testing may be used
        Used with pythtb, tbmodels, and also fplo, ASE until proper evaluation of matrix elements is implemented for them.
        """

        iR0 = self.iR0
        if self.getAA:
            self.AA_R = np.zeros((self.num_wann, self.num_wann, self.nRvec0, 3), dtype=complex)
            if not self.use_wcc_phase:
                for i in range(self.num_wann):
                    self.AA_R[i, i, iR0, :] = self.wannier_centers_cart_auto[i]

        if self.getBB:
            self.BB_R = np.zeros((self.num_wann, self.num_wann, self.nRvec0, 3), dtype=complex)
            if not self.use_wcc_phase:
                for i in range(self.num_wann):
                    self.BB_R[i, i, iR0, :] = self.AA_R[i, i, iR0, :] * self.Ham_R[i, i, iR0]

        if self.getCC:
            self.CC_R = np.zeros((self.num_wann, self.num_wann, self.nRvec0, 3), dtype=complex)

        if self.getSS and getSS:
            raise NotImplementedError()

        if self.getSA:
            raise NotImplementedError()

        if self.getSHA:
            raise NotImplementedError()


    def do_at_end_of_init(self):
        self.set_symmetry()
        self.check_periodic()
        self.set_wannier_centers()
        self.do_ws_dist()
        print("Number of wannier functions:", self.num_wann)
        print("Number of R points:", self.nRvec)
        print("Recommended size of FFT grid", self.NKFFT_recommended)

    def do_ws_dist(self):
        if self.use_ws and (self.mp_grid is not None):
            print("using ws_distance")
            ws_map = ws_dist_map(
                self.iRvec, self.wannier_centers_cart_ws, self.mp_grid, self.real_lattice, npar=self.npar)
            for X in ['Ham', 'AA', 'BB', 'CC', 'SS', 'FF', 'SA', 'SHA', 'SR', 'SH', 'SHR']:
                XR = X + '_R'
                if hasattr(self, XR):
                    print("using ws_dist for {}".format(XR))
                    vars(self)[XR] = ws_map(vars(self)[XR])
            self.iRvec = np.array(ws_map._iRvec_ordered, dtype=int)
        else:
            print("NOT using ws_dist")

    @property
    def wannier_centers_cart_ws(self):
        "to prefer the values read from .chk over the values provided in the input"
        if hasattr(self, "wannier_centers_cart_auto"):
            return self.wannier_centers_cart_auto
        else:
            return self.wannier_centers_cart

    def to_tb_file(self, tb_file=None):
        if tb_file is None:
            tb_file = self.seedname + "_fromchk_tb.dat"
        f = open(tb_file, "w")
        f.write("written by wannier-berri form the chk file\n")
        cprint(f"writing TB file {tb_file}", 'green', attrs=['bold'])
        np.savetxt(f, self.real_lattice)
        f.write("{}\n".format(self.num_wann))
        f.write("{}\n".format(self.nRvec))
        for i in range(0, self.nRvec, 15):
            a = self.Ndegen[i:min(i + 15, self.nRvec)]
            f.write("  ".join("{:2d}".format(x) for x in a) + "\n")
        for iR in range(self.nRvec):
            f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
            f.write(
                "".join(
                    "{0:3d} {1:3d} {2:15.8e} {3:15.8e}\n".format(
                        m + 1, n + 1, self.Ham_R[m, n, iR].real * self.Ndegen[iR], self.Ham_R[m, n, iR].imag
                        * self.Ndegen[iR]) for n in range(self.num_wann) for m in range(self.num_wann)))
        if hasattr(self, 'AA_R'):
            for iR in range(self.nRvec):
                f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
                f.write(
                    "".join(
                        "{0:3d} {1:3d} ".format(m + 1, n + 1) + " ".join(
                            "{:15.8e} {:15.8e}".format(a.real, a.imag)
                            for a in self.AA_R[m, n, iR] * self.Ndegen[iR]) + "\n" for n in range(self.num_wann)
                        for m in range(self.num_wann)))
        f.close()

    def _FFT_compatible(self, FFT, iRvec):
        "check if FFT is enough to fit all R-vectors"
        return np.unique(iRvec % FFT, axis=0).shape[0] == iRvec.shape[0]

    @property
    def NKFFT_recommended(self):
        "finds a minimal FFT grid on which different R-vectors do not overlap"
        if self.mp_grid is not None:
            return self.mp_grid
        NKFFTrec = np.ones(3, dtype=int)
        for i in range(3):
            R = self.iRvec[:, i]
            if len(R[R > 0]) > 0:
                NKFFTrec[i] += R.max()
            if len(R[R < 0]) > 0:
                NKFFTrec[i] -= R.min()
        assert self._FFT_compatible(NKFFTrec, self.iRvec)
        return NKFFTrec

    def set_symmetry(self, symmetry_gen=[]):
        """
        Set the symmetry group of the :class:`System`

        Parameters
        ----------
        symmetry_gen : list of :class:`symmetry.Symmetry` or str
            The generators of the symmetry group.

        Notes
        -----
        + Only the generators of the symmetry group are essential. However, no problem if more symmetries are provided.
          The code further evaluates all possible products of symmetry operations, until the full group is restored.
        + Providing `Identity` is not needed. It is included by default
        + Operations are given as objects of :class:`symmetry.Symmetry` or by name as `str`, e.g. ``'Inversion'`` , ``'C6z'``, or products like ``'TimeReversal*C2x'``.
        + ``symetyry_gen=[]`` is equivalent to not calling this function at all
        + Only the **point group** operations are important. Hence, for non-symmorphic operations, only the rotational part should be given, neglecting the translation.

        """
        self.symgroup = Group(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)

    @lazy_property.LazyProperty
    def cRvec(self):
        return self.iRvec.dot(self.real_lattice)

    @lazy_property.LazyProperty
    def cRvec_p_wcc(self):
        """
        With self.use_wcc_phase=True it is R+tj-ti. With self.use_wcc_phase=False it is R. [i,j,iRvec,a] (Cartesian)
        """
        if self.use_wcc_phase:
            return self.cRvec[None, None, :, :] + self.diff_wcc_cart[:, :, None, :]
        else:
            return self.cRvec[None, None, :, :]

    @lazy_property.LazyProperty
    def diff_wcc_cart(self):
        """
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [i,j,a] (Cartesian)
        """
        wannier_centers = self.wannier_centers_cart
        return wannier_centers[None, :, :] - wannier_centers[:, None, :]

    @lazy_property.LazyProperty
    def diff_wcc_red(self):
        """
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [i,j,a] (Reduced)
        """
        wannier_centers = self.wannier_centers_reduced
        return wannier_centers[None, :, :] - wannier_centers[:, None, :]

    @property
    def wannier_centers_cart_wcc_phase(self):
        "returns zero array if use_wcc_phase = False"
        if self.use_wcc_phase:
            return self.wannier_centers_cart
        else:
            return np.zeros_like(self.wannier_centers_cart)

    def set_wannier_centers(self):
        """
        set self.wannier_centers_cart and self.wannier_centers_reduced. Also, if
        use_wcc_phase=True, modify the relevant real-space matrix elements .
        """
        if self.wannier_centers_cart is not None:
            if self.wannier_centers_reduced is not None:
                raise ValueError(
                    "one should not specify both wannier_centers_cart and wannier_centers_reduced,"
                    "or, set_wannier_centers should not be called twice")
            else:
                self.wannier_centers_reduced = self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))
        elif self.wannier_centers_reduced is not None:
            self.wannier_centers_cart = self.wannier_centers_reduced.dot(self.real_lattice)
        elif hasattr(self, "wannier_centers_cart_auto"):
            self.wannier_centers_cart = self.wannier_centers_cart_auto
            self.wannier_centers_reduced = self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))
        if self.use_wcc_phase:
            if self.wannier_centers_cart is None:
                raise ValueError("use_wcc_phase = True, but the wannier centers could not be determined")
            if hasattr(self, 'AA_R'):
                AA_R_new = np.copy(self.AA_R)
                AA_R_new[np.arange(self.num_wann), np.arange(self.num_wann), self.iR0, :] -= self.wannier_centers_cart
            if hasattr(self, 'BB_R'):
                print("WARNING: orbital moment does not work with wcc_phase so far")
                BB_R_new = self.BB_R.copy() - self.Ham_R[:, :, :, None] * self.wannier_centers_cart[None, :, None, :]
            if hasattr(self, 'CC_R'):
                print("WARNING: orbital moment does not work with wcc_phase so far")
                norm = np.linalg.norm(self.CC_R - self.conj_XX_R(self.CC_R))
                assert norm < 1e-10, f"CC_R is not Hermitian, norm={norm}"
                assert hasattr(self, 'BB_R'), "if you use CC_R and use_wcc_phase=True, you need also BB_R"
                T = self.wannier_centers_cart[:, None, None, :, None] * self.BB_R[:, :, :, None, :]
                CC_R_new = self.CC_R.copy() + 1.j * sum(
                    s * (
                        -T[:, :, :, a, b]  # -t_i^a * B_{ij}^b(R)
                        - self.conj_XX_R(T[:, :, :, b, a])  # - B_{ji}^a(-R)^*  * t_j^b
                        + self.wannier_centers_cart[:, None, None, a] * self.Ham_R[:, :, :, None]
                        * self.wannier_centers_cart[None, :, None, b]  # + t_i^a*H_ij(R)t_j^b
                    ) for (s, a, b) in [(+1, alpha_A, beta_A), (-1, beta_A, alpha_A)])
                norm = np.linalg.norm(CC_R_new - self.conj_XX_R(CC_R_new))
                assert norm < 1e-10, f"CC_R after applying wcc_phase is not Hermitian, norm={norm}"

            if (hasattr(self, "SA_R") or hasattr(self, "SHA_R") or hasattr(self, "SR_R") or hasattr(self, "SH")
                    or hasattr(self, "SHR_R")):
                raise NotImplementedError("use_wcc_phase=True for spin current matrix elements not implemented")

            for X in ['AA', 'BB', 'CC']:
                if hasattr(self, X + '_R'):
                    vars(self)[X + '_R'] = locals()[X + '_R_new']

    @property
    def iR0(self):
        return self.iRvec.tolist().index([0, 0, 0])

    @lazy_property.LazyProperty
    def reverseR(self):
        """indices of R vectors that has -R in irvec, and the indices of the corresponding -R vectors."""
        mapping = np.all(self.iRvec[:, None, :] + self.iRvec[None, :, :] == 0, axis=2)
        # check if some R-vectors do not have partners
        notfound = np.where(np.logical_not(mapping.any(axis=1)))[0]
        for ir in notfound:
            print(f"WARNING : R[{ir}] = {self.iRvec[ir]} does not have a -R partner")
        # check if some R-vectors have more then 1 partner
        morefound = np.where(np.sum(mapping, axis=1) > 1)[0]
        if len(morefound > 0):
            raise RuntimeError(
                f"R vectors number {morefound} have more then one negative partner : "
                f"\n{self.iRvec[morefound]} \n{np.sum(mapping,axis=1)}")
        lst_R, lst_mR = [], []
        for ir1 in range(self.nRvec):
            ir2 = np.where(mapping[ir1])[0]
            if len(ir2) == 1:
                lst_R.append(ir1)
                lst_mR.append(ir2[0])
        lst_R = np.array(lst_R)
        lst_mR = np.array(lst_mR)
        # Check whether the result is correct
        assert np.all(self.iRvec[lst_R] + self.iRvec[lst_mR] == 0)
        return lst_R, lst_mR

    def conj_XX_R(self, XX_R):
        """ reverses the R-vector and takes the hermitian conjugate """
        XX_R_new = np.zeros_like(XX_R)
        lst_R, lst_mR = self.reverseR
        XX_R_new[:, :, lst_R] = XX_R[:, :, lst_mR]
        return XX_R_new.swapaxes(0, 1).conj()

    @property
    def nRvec(self):
        return self.iRvec.shape[0]

    @lazy_property.LazyProperty
    def cell_volume(self):
        return abs(np.linalg.det(self.real_lattice))

    def check_hermitian(self, XX):
        if hasattr(self, XX):
            XX_R = np.copy(vars(self)[XX])
            assert (np.max(abs(XX_R - self.conh_XX_R(XX_R))) < 1e-8), f"{XX} should obey X(-R) = X(R)^+"
        else:
            print(f"{XX} is missing,nothing to check")

    def set_structure(self, positions, atom_labels, magnetic_moments=None):
        """
        Set atomic structure of the system.

        Parameters
        ----------
        positions : (num_atom, 3) array_like of float
            Atomic positions in fractional coordinates.
        atom_labels: (num_atom,) list
            labels (integer, string, etc.) to distinguish species.
        magnetic_moments: (num_atom, 3) array_like of float (optional)
            Magnetic moment vector of each atom.
        """
        if len(positions) != len(atom_labels):
            raise ValueError("length of positions and atom_labels must be the same")
        if magnetic_moments is not None:
            if len(magnetic_moments) != len(positions):
                raise ValueError("length of positions and magnetic_moments must be the same")
            if not all([len(x) for x in magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
        self.positions = positions
        self.atom_labels = atom_labels
        self.magnetic_moments = magnetic_moments

    def get_spglib_cell(self):
        """Returns the atomic structure as a cell tuple in spglib format"""
        try:
            # assign integer to self.atom_labels
            atom_labels_unique = list(set(self.atom_labels))
            atom_numbers = [atom_labels_unique.index(label) for label in self.atom_labels]
            if self.magnetic_moments is None:
                return (self.real_lattice, self.positions, atom_numbers)
            else:
                return (self.real_lattice, self.positions, atom_numbers, self.magnetic_moments)
        except AttributeError:
            raise AttributeError("set_structure must be called before get_spglib_cell")

    def set_symmetry_from_structure(self):
        """
        Set the symmetry group of the :class:`System`. Requires spglib to be installed.
        :meth:`System.set_structure` must be called in advance.

        For magnetic systems, symmetries involving time reversal are not detected because
        spglib does not support time reversal symmetry for noncollinear systems.
        """
        import spglib

        spglib_symmetry = spglib.get_symmetry(self.get_spglib_cell())
        symmetry_gen = []
        for isym in range(spglib_symmetry["rotations"].shape[0]):
            # spglib gives real-space rotations in reduced coordinates. Here,
            # 1) convert to Cartesian coordinates, and
            # 2) take transpose to go to reciprocal space.
            W = spglib_symmetry["rotations"][isym]
            Wcart = self.real_lattice.T @ W @ np.linalg.inv(self.real_lattice).T
            R = Wcart.T
            symmetry_gen.append(Symmetry(R))

        if self.magnetic_moments is None:
            symmetry_gen.append(TimeReversal)
        else:
            if not all([len(x) for x in self.magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
            print(
                "Warning: spglib does not find symmetries including time reversal "
                "operation.\nTo include such symmetries, use set_symmetry.")

        self.symgroup = Group(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)


class ws_dist_map():

    def __init__(self, iRvec, wannier_centers, mp_grid, real_lattice, npar=multiprocessing.cpu_count()):
        # Find the supercell translation (i.e. the translation by a integer number of
        # supercell vectors, the supercell being defined by the mp_grid) that
        # minimizes the distance between two given Wannier functions, i and j,
        # the first in unit cell 0, the other in unit cell R.
        # I.e., we find the translation to put WF j in the Wigner-Seitz of WF i.
        # We also look for the number of equivalent translation, that happen when w_j,R
        # is on the edge of the WS of w_i,0. The results are stored
        # a dictionary shifts_iR[(iR,i,j)]
        ws_search_size = np.array([2] * 3)
        ws_distance_tol = 1e-5
        cRvec = iRvec.dot(real_lattice)
        mp_grid = np.array(mp_grid)
        shifts_int_all = np.array([ijk for ijk in iterate3dpm(ws_search_size + 1)]) * np.array(mp_grid[None, :])
        self.num_wann = wannier_centers.shape[0]
        self._iRvec_new = dict()
        param = (shifts_int_all, wannier_centers, real_lattice, ws_distance_tol, wannier_centers.shape[0])
        p = multiprocessing.Pool(npar)
        irvec_new_all = p.starmap(functools.partial(ws_dist_stars, param=param), zip(iRvec, cRvec))
        p.close()
        p.join()
        print('irvec_new_all shape', np.shape(irvec_new_all))
        for ir, iR in enumerate(iRvec):
            for ijw, irvec_new in irvec_new_all[ir].items():
                self._add_star(ir, irvec_new, ijw[0], ijw[1])
        self._iRvec_ordered = sorted(self._iRvec_new)
        for ir, R in enumerate(iRvec):
            chsum = 0
            for irnew in self._iRvec_new:
                if ir in self._iRvec_new[irnew]:
                    chsum += self._iRvec_new[irnew][ir]
            chsum = np.abs(chsum - np.ones((self.num_wann, self.num_wann))).sum()
            if chsum > 1e-12: print("WARNING: Check sum for {0} : {1}".format(ir, chsum))

    def __call__(self, matrix):
        ndim = len(matrix.shape) - 3
        num_wann = matrix.shape[0]
        reshaper = (num_wann, num_wann) + (1, ) * ndim
        matrix_new = np.array(
            [
                sum(
                    matrix[:, :, ir] * self._iRvec_new[irvecnew][ir].reshape(reshaper)
                    for ir in self._iRvec_new[irvecnew]) for irvecnew in self._iRvec_ordered
            ]).transpose((1, 2, 0) + tuple(range(3, 3 + ndim)))
        assert (np.abs(matrix_new.sum(axis=2) - matrix.sum(axis=2)).max() < 1e-12)
        return matrix_new

    def _add_star(self, ir, irvec_new, iw, jw):
        weight = 1. / irvec_new.shape[0]
        for irv in irvec_new:
            self._add(ir, irv, iw, jw, weight)

    def _add(self, ir, irvec_new, iw, jw, weight):
        irvec_new = tuple(irvec_new)
        if irvec_new not in self._iRvec_new:
            self._iRvec_new[irvec_new] = dict()
        if ir not in self._iRvec_new[irvec_new]:
            self._iRvec_new[irvec_new][ir] = np.zeros((self.num_wann, self.num_wann), dtype=float)
        self._iRvec_new[irvec_new][ir][iw, jw] += weight


def ws_dist_stars(iRvec, cRvec, param):
    shifts_int_all, wannier_centers, real_lattice, ws_distance_tol, num_wann = param
    irvec_new = {}
    for jw in range(num_wann):
        for iw in range(num_wann):
            # function JW translated in the Wigner-Seitz around function IW
            # and also find its degeneracy, and the integer shifts needed
            # to identify it
            R_in = -wannier_centers[iw] + cRvec + wannier_centers[jw]
            dist = np.linalg.norm(R_in[None, :] + shifts_int_all.dot(real_lattice), axis=1)
            irvec_new[(iw, jw)] = iRvec + shifts_int_all[dist - dist.min() < ws_distance_tol].copy()
    return irvec_new
