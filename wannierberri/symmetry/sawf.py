from functools import cached_property, lru_cache
import warnings
from irrep.bandstructure import BandStructure
from irrep.spacegroup import SpaceGroupBare
import numpy as np


from ..utility import cached_einsum, clear_cached
from ..w90files.amn import AMN
from .utility import get_inverse_block, rotate_block_matrix
from .projections import Projection, ProjectionsSet

from .Dwann import Dwann
from .orbitals import OrbitalRotator

from packaging import version
import irrep
irrep_version = version.parse(irrep.__version__)


IRREP_IRREDUCIBLE_VERSION = version.parse("2.2.0")  # the version of irrep that supports irreducible band structure


class SymmetrizerSAWF:
    """
    A class to handle the symmetry of the Wannier functions and the ab initio bands

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.sawf`
        if None, the object is initialized with void values (zeroes)
    num_wann : int
        the number of Wannier functions (in the case of void initialization)
    num_bands : int
        the number of ab initio bands (in the case of void initialization)
    nkpt : int
        the number of kpoints (in the case of void initialization)

    Attributes
    ----------
    comment : str
        the comment at the beginning of the file
    NB : int
        the number of ab initio bands
    Nsym : int
        the number of symmetries
    NKirr : int
        the number of irreducible kpoints
    NK : int
        the number of kpoints
    num_wann : int
        the number of Wannier functions
    kptirr : numpy.ndarray(int, shape=(NKirr,))
        the list of irreducible kpoints
    kpt2kptirr : numpy.ndarray(int, shape=(NK,))
        the mapping from kpoints to irreducible kpoints (each number denotes the index of the irreducible kpoint in kptirr)
    kptirr2kpt : numpy.ndarray(int, shape=(NKirr, Nsym))
        the mapping from irreducible kpoints to all kpoints
    kpt2kptirr_sym : numpy.ndarray(int, shape=(NK,))
        the symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question
    d_band_blocks : list(list(numpy.ndarray(complex, shape=(NB, NB))))
        the ab initio band transformation matrices in the block form (between almost degenerate bands)
    d_band_blocks_inverse : list(list(numpy.ndarray(complex, shape=(NB, NB))))
        the inverse matrices of d_band_blocks
    d_band_block_indices : list(np.array(int, shape=(Nblocks, 2)))
        the indices of the blocks in the ab initio band transformation matrices
    D_wann_blocks : list(list(numpy.ndarray(complex, shape=(num_wann, num_wann))))
        the Wannier function transformation matrices in the block form. Within the same irreducible representation of WFs
    D_wann_blocks_inverse : list(list(numpy.ndarray(complex, shape=(num_wann, num_wann))))
        the inverse matrices of D_wann_blocks
    D_wann_block_indices : np.array(int, shape=(Nblocks, 2))
        the indices of the blocks in the Wannier function transformation matrices
    time_reversals : array(bool, shape=(Nsym,))
        weather the symmetry operation includes time reversal or not
    """

    npz_tags = ['D_wann_block_indices', '_NB',
                'kpt2kptirr', 'kptirr', 'kptirr2kpt', 'kpt2kptirr_sym',
                '_NK', 'num_wann', 'comment', 'NKirr', 'Nsym', 'time_reversals',]
    npz_tags_optional = ['eig_irr', 'kpoints_all', 'kpt_from_kptirr_isym', 'grid']


    def __init__(self):
        self._NB = 0
        self.num_wann = 0
        self.D_wann_block_indices = np.zeros((0, 2), dtype=int)
        self.NKirr = 0
        self.Nsym = 0
        self.kpoints_all = np.zeros((0, 3), dtype=float)
        self.kpt2kptirr = np.zeros(0, dtype=int)
        self.kptirr = np.zeros(0, dtype=int)
        self.kptirr2kpt = np.zeros((0, 0), dtype=int)


    def from_irrep(self, bandstructure: BandStructure,
                 grid=None, degen_thresh=1e-2, store_eig=True,
                 ecut=None,  # not used, but kept for compatibility
                 irreducible=False,
                 unitary_params=None):
        """
        Initialize the object from the BandStructure object

        Parameters
        ---------- 
        bandstructure : irrep.bandstructure.BandStructure
            the object containing the band structure
        grid : tuple(int), optional
            the grid of kpoints (3 integers), if None, the grid is determined from the kpoints
            may be used to reduce the grid (by an integer factor) for the symmetry analysis
        degen_thresh : float, optional
            the threshold for the degeneracy of the bands. Only transformations between bands
             with energy difference smaller than this value are considered
        irreducible : bool, optional
            if True, the symmetrizer will use only the irreducible kpoints, the rest will be ignored

        """
        if unitary_params is None:
            unitary_params = {}
        if irrep_version >= IRREP_IRREDUCIBLE_VERSION:
            data = bandstructure.get_dmn(grid=grid,
                                         Ecut=ecut,
                                         irreducible=irreducible,
                                         degen_thresh=degen_thresh,
                                         unitary=True,
                                         unitary_params=unitary_params)
        else:
            if irreducible:
                raise ImportError("The irreducible option requires irrep version >= 2.2.0, please update irrep")
            data = bandstructure.get_dmn(grid=grid,
                                         degen_thresh=degen_thresh,
                                         unitary=True,
                                         unitary_params=unitary_params)
        self.grid = data["grid"]
        self.kpoints_all = data["kpoints"]
        self.kpt2kptirr = data["kpt2kptirr"]
        self.kptirr = data["kptirr"]
        self.kptirr2kpt = data["kptirr2kpt"]
        self.d_band_blocks = data["d_band_blocks"]
        self.d_band_block_indices = data["d_band_block_indices"]
        if irrep_version >= IRREP_IRREDUCIBLE_VERSION:
            self.selected_kpoints = data["selected_kpoints"]
            self.kpt_from_kptirr_isym = data["kpt_from_kptirr_isym"]

        self.comment = "Generated by wannierberri with irrep"
        self.D_wann = []
        self.spacegroup = bandstructure.spacegroup
        self.Nsym = bandstructure.spacegroup.size
        self.time_reversals = np.array([symop.time_reversal for symop in self.spacegroup.symmetries])
        self.NKirr = len(self.kptirr)
        self._NK = len(self.kpoints_all)
        self._NB = bandstructure.num_bands
        self.clear_inverse()
        if store_eig:
            self.set_eig([bandstructure.kpoints[ik].Energy_raw for ik in self.kptirr])
        return self

    @cached_property
    def d_band_blocks_inverse(self):
        return get_inverse_block(self.d_band_blocks)

    @cached_property
    def D_wann_blocks_inverse(self):
        return get_inverse_block(self.D_wann_blocks)

    @property
    def NK(self):
        return self._NK

    @property
    def NB(self):
        if hasattr(self, '_NB'):
            return self._NB
        else:
            return 0

    @cached_property
    def isym_little(self):
        return [np.where(self.kptirr2kpt[ik] == self.kptirr[ik])[0] for ik in range(self.NKirr)]

    @cached_property
    def kpt2kptirr_sym(self):
        return np.array([np.where(self.kptirr2kpt[self.kpt2kptirr[ik], :] == ik)[0][0] for ik in range(self.NK)])

    @cached_property
    def kptirr_weights(self):
        """
        Returns the weights of the irreducible kpoints in the full BZ
        """
        return np.array([len(set(self.kptirr2kpt[ikirr])) for ikirr in range(self.NKirr)])




    @cached_property
    def orbitalrotator(self):
        return OrbitalRotator()
        # return OrbitalRotator([symop.rotation_cart for symop in self.spacegroup.symmetries])


    def set_D_wann_from_projections(self,
                                    projections,
                                    ):
        """
        Parameters
        ----------
        projections : ProjectionsSet or list(Projection)
            alternative way to provide the projections. Will be appended to the projections list
        kpoints : np.array(float, shape=(npoints,3,))
            the kpoints in fractional coordinates (neede only if the kpoints are not stored in the object yet) 
        """
        # a list of tuples (positions, orbital_string, basis_list)
        projections_list = []

        if isinstance(projections, Projection):
            projections = [projections]
        elif isinstance(projections, ProjectionsSet):
            projections = projections.projections
        for proj in projections:
            orbitals = proj.orbitals
            basis_list = proj.basis_list
            print(f"orbitals = {orbitals}")
            if len(orbitals) > 1:
                warnings.warn(f"projection {proj} has more than one orbital. it will be split into separate blocks, please order them in the win file consistently")
            for orb in orbitals:
                projections_list.append((proj.positions, orb, basis_list))

        D_wann_list = []
        self.T_list = []
        self.atommap_list = []
        self.rot_orb_list = []
        for positions, proj, basis_list in projections_list:
            print(f"calculating Wannier functions for {proj} at {positions}")
            _Dwann = Dwann(spacegroup=self.spacegroup, positions=positions, orbital=proj, orbitalrotator=self.orbitalrotator,
                           spinor=self.spacegroup.spinor,
                           basis_list=basis_list)
            _dwann = _Dwann.get_on_points_all(kpoints=self.kpoints_all, ikptirr=self.kptirr, ikptirr2kpt=self.kptirr2kpt)
            D_wann_list.append(_dwann)
            self.T_list.append(_Dwann.T)
            self.atommap_list.append(_Dwann.atommap)
            self.rot_orb_list.append(_Dwann.rot_orb)
        self.set_D_wann(D_wann_list)
        return self

    @cached_property
    def rot_orb_dagger_list(self):
        return [rot_orb.swapaxes(-2, -1).conj()
            for rot_orb in self.rot_orb_list]

    def set_D_wann(self, D_wann):
        """
        set the D_wann matrix

        Parameters
        ----------
        D_wann : np.array(complex, shape=(NKirr, Nsym, num_wann, num_wann))
                the Wannier function transformation matrix (conjugate transpose)
        atommap : np.array(int, shape=(num_points,Nsym))
            the mapping of atoms under the symmetry operations
        T : np.array(int, shape=(num_points,Nsym))
            the translation lattice vector that brings the original atom in position atommap[ip,isym] to the transform of atom ip

        Notes
        -----
        the parameters can be given as lists, in that case the lengths of lists must be equal
          also updates the num_wann attribute
        """
        self.clear_inverse(d=False, D=True)
        if not isinstance(D_wann, list):
            D_wann = [D_wann]
        print("D.shape", [D.shape for D in D_wann])
        self.D_wann_block_indices = []
        num_wann = 0
        self.D_wann_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        for D in D_wann:
            assert D.shape[0] == self.NKirr
            assert D.shape[1] == self.Nsym
            assert D.shape[2] == D.shape[3]
            self.D_wann_block_indices.append((num_wann, num_wann + D.shape[2]))
            num_wann += D.shape[2]
            for ik in range(self.NKirr):
                for isym in range(self.Nsym):
                    self.D_wann_blocks[ik][isym].append(D[ik, isym])
        self.D_wann_block_indices = np.array(self.D_wann_block_indices)
        self.num_wann = num_wann
        print("num_wann", num_wann)
        print("D_wann_block_indices", self.D_wann_block_indices)


    def symmetrize_wannier_property(self, wannier_property):
        """
        Symmetrizes a property of the Wannier functions (single-WF property) over the spacegroup

        Parameters
        ----------
        wannier_property : np.ndarray(dtype=float, shape=(num_wann, ...))
            The property of the Wannier functions to be symmetrized. The first axis should be the Wannier function index,
            the second [optional] should be the cartesian coordinates
        """
        ncart = (wannier_property.ndim - 1)
        if ncart == 0:
            wcc_red_in = wannier_property
        elif ncart == 1:
            wcc_red_in = wannier_property @ self.spacegroup.lattice_inv
        else:
            raise ValueError("The input should be either a vector or a matrix")
        WCC_red_out = np.zeros((self.num_wann,) + (3,) * ncart, dtype=float)
        for isym, symop in enumerate(self.spacegroup.symmetries):
            for block, (ws, _) in enumerate(self.D_wann_block_indices):
                norb = self.rot_orb_list[block][0, 0].shape[0]
                T = self.T_list[block][:, isym]
                num_points = T.shape[0]
                atom_map = self.atommap_list[block][:, isym]
                for atom_a in range(num_points):
                    start_a = ws + atom_a * norb
                    atom_b = atom_map[atom_a]
                    start_b = ws + atom_b * norb
                    XX_L = wcc_red_in[start_a:start_a + norb]
                    if ncart > 0:
                        XX_L = symop.transform_r(XX_L) + T[atom_a]
                    transformed = cached_einsum("ij,j...,ji->i...", self.rot_orb_dagger_list[block][atom_a, isym].T, XX_L, self.rot_orb_list[block][atom_a, isym].T).real
                    WCC_red_out[start_b:start_b + norb] += transformed
        if ncart > 0:
            WCC_red_out = WCC_red_out @ self.spacegroup.lattice
        return WCC_red_out / self.spacegroup.size

    @lru_cache
    def d_band_diagonal(self, ikirr, isym):
        if ikirr is None:
            return np.array([self.d_band_diagonal(ikirr, isym) for ikirr in range(self.NKirr)])
        if isym is None:
            return np.array([self.d_band_diagonal(ikirr, isym) for isym in range(self.Nsym)])

        result = np.zeros(self.NB, dtype=complex)
        for (start, end), block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
            result[start:end] = block.diagonal()
        return result

    def set_eig(self, eig):
        eig = np.array(eig, dtype=float)
        assert eig.ndim == 2
        assert eig.shape[1] == self.NB
        if eig.shape[0] == self.NK:
            self.eig_irr = eig[self.kptirr]
        elif eig.shape[0] == self.NKirr:
            self.eig_irr = eig
        else:
            raise ValueError(f"The shape of eig should be either ({self.NK}, {self.NB}) or ({self.NKirr}, {self.NB}), not {eig.shape}")

    def symmetrize_WCC(self, wannier_centers_cart):
        return self.symmetrize_wannier_property(wannier_centers_cart)

    def symmetrize_spreads(self, wannier_spreads):
        return self.symmetrize_wannier_property(wannier_spreads)

    def set_spacegroup(self, spacegroup):
        self.spacegroup = spacegroup
        self.time_reversals = np.array([symop.time_reversal for symop in self.spacegroup.symmetries])
        self.Nsym = spacegroup.size
        return self

    def as_dict(self):
        dic = {k: self.__getattribute__(k) for k in self.__class__.npz_tags}
        for k in self.__class__.npz_tags_optional:
            if hasattr(self, k):
                dic[k] = self.__getattribute__(k)

        for ik in range(self.NKirr):
            dic[f'd_band_block_indices_{ik}'] = self.d_band_block_indices[ik]
            for i in range(len(self.d_band_block_indices[ik])):
                dic[f'd_band_blocks_{ik}_{i}'] = np.array([self.d_band_blocks[ik][isym][i] for isym in range(self.Nsym)])
        for i in range(len(self.D_wann_block_indices)):
            dic[f'D_wann_blocks_{i}'] = np.array([[self.D_wann_blocks[ik][isym][i] for isym in range(self.Nsym)]
                                            for ik in range(self.NKirr)])

        for k, val in self.spacegroup.as_dict().items():
            dic["spacegroup_" + k] = val
        for attrname in ["T", "atommap", "rot_orb"]:
            if hasattr(self, attrname + "_list"):
                for i, t in enumerate(self.__getattribute__(attrname + "_list")):
                    dic[f'{attrname}_{i}'] = t
        return dic

    def from_dict(self, dic=None, return_obj=True, **kwargs):
        dic_loc = {}
        cls = self.__class__
        for k in self.__class__.npz_tags:
            dic_loc[k] = dic[k]

        for k in cls.npz_tags_optional:
            if k in dic:
                dic_loc[k] = dic[k]


        for k, v in dic_loc.items():
            self.__setattr__(k, v)

        self.d_band_block_indices = [dic[f'd_band_block_indices_{ik}'] for ik in range(self.NKirr)]
        self.d_band_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        self.D_wann_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        d_band_num_blocks = [self.d_band_block_indices[ik].shape[0] for ik in range(self.NKirr)]
        D_wann_num_blocks = self.D_wann_block_indices.shape[0]


        d_band_blocks_tmp = [[dic[f"d_band_blocks_{ik}_{i}"] for i in range(nblock)]
                             for ik, nblock in enumerate(d_band_num_blocks)]
        D_wann_blocks_tmp = [dic[f"D_wann_blocks_{i}"]
                             for i in range(D_wann_num_blocks)]

        for ik in range(self.NKirr):
            for isym in range(self.Nsym):
                self.d_band_blocks[ik][isym] = [np.ascontiguousarray(d_band_blocks_tmp[ik][i][isym])
                                                for i in range(d_band_num_blocks[ik])]
                self.D_wann_blocks[ik][isym] = [np.ascontiguousarray(D_wann_blocks_tmp[i][ik, isym])
                                                for i in range(D_wann_num_blocks)]
        prefix = "spacegroup_"
        l = len(prefix)
        dic_spacegroup = {k[l:]: v for k, v in dic.items() if k.startswith(prefix)}
        if len(dic_spacegroup) > 0:
            self.spacegroup = SpaceGroupBare(**dic_spacegroup)
        for prefix in ["T", "atommap", "rot_orb"]:
            keys = sorted([k for k in dic.keys() if k.startswith(prefix)])
            lst = [dic[k] for k in keys]
            self.__setattr__(prefix + "_list", lst)
        return self


    @lru_cache
    def ndegen(self, ikirr):
        return len(set(self.kptirr2kpt[ikirr]))

    def __call__(self, *args, **kwds):
        raise RuntimeError("The SymmetrizerSAWF is not callable, only its child classes are")


    def U_to_full_BZ(self, U, include_k=None):
        """
        Expands the U matrix from the irreducible to the full BZ

        Parameters
        ----------
        U : list of NKirr np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The input matrix to be expanded
        include_k : array(NK, bool)
            indicates which k-points to include

        Returns
        -------
        U : list of NK np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The expanded matrix. if all_k is False, the U matrices at the kpoints not included in self.include_k are set to None
        """
        all_k = include_k is None
        Ufull = [None for _ in range(self.NK)]
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                iRk = self.kptirr2kpt[ikirr, isym]
                if Ufull[iRk] is None and (all_k or include_k[iRk]):
                    Ufull[iRk] = self.rotate_U(U[ikirr], ikirr, isym, forward=True)
        return Ufull

    def rotate_U(self, U, ikirr, isym, forward=True):
        """
        Rotates the umat matrix at the irreducible kpoint
        U = D_band^+ @ U @ D_wann
        """
        d_indices = self.d_band_block_indices[ikirr]
        D_indices = self.D_wann_block_indices
        # forward = not forward
        U1 = np.zeros(U.shape, dtype=complex)
        Uloc = U.copy()
        if forward:
            if self.time_reversals[isym]:
                Uloc = Uloc.conj()
            Uloc = rotate_block_matrix(Uloc,
                                       lblocks=self.d_band_blocks[ikirr][isym],
                                       lindices=d_indices,
                                       rblocks=self.D_wann_blocks_inverse[ikirr][isym],
                                       rindices=D_indices,
                                    #    inv_left=False, inv_right=True,
                                       result=U1)

            # return d @ U @ D.conj().T
        else:
            Uloc = rotate_block_matrix(Uloc,
                                     lblocks=self.d_band_blocks_inverse[ikirr][isym],
                                     lindices=d_indices,
                                     rblocks=self.D_wann_blocks[ikirr][isym],
                                     rindices=D_indices,
                                     result=U1)
            if self.time_reversals[isym]:
                Uloc = Uloc.conj()

        return Uloc

    def clear_inverse(self, d=True, D=True):
        if d:
            clear_cached(self, 'd_band_blocks_inverse')
            # if hasattr(self, 'd_band_blocks_inverse'):
            #     del self.d_band_blocks_inverse
        if D:
            clear_cached(self, 'D_wann_blocks_inverse')
            # if hasattr(self, 'D_wann_blocks_inverse'):
            #     del self.D_wann_blocks_inverse

    def select_bands(self, selected_bands):
        """select the bands to be used in the calculation, the rest are excluded

            Parameters
            ----------
            selected_bands : list of int or bool
                the indices of the bands to be used, NOT a boolean mask
        """
        if selected_bands is None:
            return
        selected_bands_bool = np.zeros(self.NB, dtype=bool)
        selected_bands_bool[selected_bands] = True
        # print(f"applying window to select {sum(selected_bands_bool)} bands from {self.NB}\n", selected_bands_bool)
        for ikirr in range(self.NKirr):
            self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr] = self.select_bands_in_blocks(self.d_band_blocks[ikirr], self.d_band_block_indices[ikirr], selected_bands_bool)
        for i, block_ind in enumerate(self.d_band_block_indices):
            if i == 0:
                self._NB = block_ind[-1, -1]
            assert block_ind[0, 0] == 0
            assert np.all(block_ind[1:, 0] == block_ind[:-1, 1])
            assert block_ind[-1, -1] == self.NB
        # print(f"new NB = {self.NB}")



    def select_bands_in_blocks(self, d_band_blocks_ik, d_band_block_indices_ik, selected_bands_bool):
        if selected_bands_bool is None:
            return d_band_blocks_ik, d_band_block_indices_ik

        new_block_indices = []
        new_blocks = [[] for _ in range(self.Nsym)]
        st = 0
        for iblock, (start, end) in enumerate(d_band_block_indices_ik):
            select = selected_bands_bool[start:end]
            nsel = np.sum(select)
            if nsel > 0:
                new_block_indices.append((st, st + nsel))
                st += nsel
                for isym in range(self.Nsym):
                    new_blocks[isym].append(
                        np.ascontiguousarray(d_band_blocks_ik[isym][iblock][:, select][select, :]))
        return np.array(new_block_indices), new_blocks



    def check_eig(self, eig):
        """
        Check the symmetry of the eigenvlues

        Parameters
        ----------
        eig : EIG object
            the eigenvalues

        Returns
        -------
        float
            the maximum error
        """
        maxerr = 0
        for ik in range(self.NK):
            ikirr = self.kpt2kptirr[ik]
            e1 = eig.data[ik]
            e2 = eig.data[self.kptirr[ikirr]]
            maxerr = max(maxerr, np.linalg.norm(e1 - e2))

        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                e1 = eig.data[self.kptirr[ikirr]]
                e2 = eig.data[self.kptirr2kpt[ikirr, isym]]
                maxerr = max(maxerr, np.linalg.norm(e1 - e2))
        return maxerr

    def check_amn(self, amn, warning_precision=1e-5,
                  ignore_upper_bands=None,
                  ignore_lower_bands=None,
                  verbose=False):
        """
        Check the symmetry of the amn

        Parameters
        ----------
        amn : AMN object of np.ndarray(complex, shape=(NK, NB, NW))
            the amn

        Returns
        -------
        float
            the maximum error

        Note:
        -----
        Works only when ALL k-points are included in the amn.
        """
        if isinstance(amn, AMN):
            amn = amn.data
        if isinstance(amn, dict):
            amn = np.array([amn[ik] for ik in range(self.NK)])
        maxerr = 0
        assert amn.shape == (self.NK, self.NB, self.num_wann), f"amn.shape = {amn.shape} != (NK={self.NK}, NB={self.NB}, num_wann={self.num_wann}) "
        if ignore_lower_bands is not None:
            assert abs(ignore_lower_bands) < self.NB
            ignore_lower_bands = abs(int(ignore_lower_bands))
        else:
            ignore_lower_bands = 0
        if ignore_upper_bands is not None:
            assert abs(ignore_upper_bands) < self.NB - ignore_lower_bands
            ignore_upper_bands = -abs(int(ignore_upper_bands))

        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                ik = self.kptirr2kpt[ikirr, isym]
                a1 = amn[ik]
                a2 = amn[self.kptirr[ikirr]]
                a1p = self.rotate_U(a1, ikirr, isym, forward=False)
                a1 = a1[ignore_lower_bands:ignore_upper_bands]
                a1p = a1p[ignore_lower_bands:ignore_upper_bands]
                a2 = a2[ignore_lower_bands:ignore_upper_bands]
                diff = a2 - a1p
                diff = np.max(abs(diff))
                maxerr = max(maxerr, np.linalg.norm(diff))
                if diff > warning_precision:
                    print(f"ikirr={ikirr}, isym={isym} : {diff}")
                    if verbose:
                        for aaa in zip(a1, a1p, a2, a1p - a2, a1p / a2):
                            string = ""
                            for a in aaa:
                                _abs = ", ".join(f"{np.abs(_):.4f}" for _ in a)
                                _angle = ", ".join(f"{np.angle(_) / np.pi * 180:7.2f}" for _ in a)
                                string += f"[{_abs}] [{_angle}]   |    "
                            print(string)
        return maxerr

    def symmetrize_amn(self, amn):
        """
        Symmetrize the amn

        Parameters
        ----------
        amn : AMN object of np.ndarray(complex, shape=(NK, NB, NW))
            the amn

        Returns
        -------
        AMN
            the symmetrized amn
        """
        if not isinstance(amn, np.ndarray):
            amn = amn.data
        assert amn.shape == (self.NK, self.NB, self.num_wann)

        amn_sym_irr = np.zeros((self.NKirr, self.NB, self.num_wann), dtype=complex)
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                amn_sym_irr[ikirr] += self.rotate_U(amn[self.kptirr2kpt[ikirr, isym]], ikirr, isym, forward=False)
        amn_sym_irr /= self.Nsym
        lfound = np.zeros(self.NK, dtype=bool)
        amn_sym = np.zeros((self.NK, self.NB, self.num_wann), dtype=complex)
        for ikirr in range(self.NKirr):
            ik = self.kptirr[ikirr]
            amn_sym[ik] = amn_sym_irr[ikirr]
            lfound[ik] = True
            for isym in range(self.Nsym):
                ik = self.kptirr2kpt[ikirr, isym]
                if not lfound[ik]:
                    amn_sym[ik] = self.rotate_U(amn_sym_irr[ikirr], ikirr, isym, forward=True)
                    lfound[ik] = True
        return amn_sym

    def get_random_amn(self):
        """ generate a random amn array that is comaptible with the symmetries of the Wanier functions in thesymmetrizer object

        Returns
        -------
        np.ndarray(complex, shape=(NK, NB, num_wann))
            the random amn, respecting the symmetries of the DMN object
        """
        shape = (self.NK, self.NB, self.num_wann)
        amn = np.random.random(shape) + 1j * np.random.random(shape)
        return self.symmetrize_amn(amn)

    def to_npz(self, f_npz):
        dic = self.as_dict()
        print(f"saving to {f_npz} : ")
        np.savez_compressed(f_npz, **dic)
        return self

    def from_npz(self, f_npz):
        dic = np.load(f_npz)
        return self.from_dict(dic)


    #
    # def check_mmn(self, mmn, f1, f2):
    #     """
    #     Check the symmetry of data in the mmn file (not working)

    #     Parameters
    #     ----------
    #     mmn : MMN object
    #         the mmn file data

    #     Returns
    #     -------
    #     float
    #         the maximum error
    #     """
    #     assert mmn.NK == self.NK
    #     assert mmn.NB == self.NB

    #     maxerr = 0
    #     neighbours_irr = np.array([self.kpt2kptirr[neigh] for neigh in mmn.neighbours])
    #     for i in range(self.NKirr):
    #         ind1 = np.where(self.kpt2kptirr == i)[0]
    #         kirr1 = self.kptirr[i]
    #         neigh_irr = neighbours_irr[ind1]
    #         for j in range(self.NKirr):
    #             kirr2 = self.kptirr[j]
    #             ind2x, ind2y = np.where(neigh_irr == j)
    #             print(f"rreducible kpoints {kirr1} and {kirr2} are equivalent to {len(ind2x)} points")
    #             ref = None
    #             for x, y in zip(ind2x, ind2y):
    #                 k1 = ind1[x]
    #                 k2 = mmn.neighbours[k1][y]
    #                 isym1 = self.kpt2kptirr_sym[k1]
    #                 isym2 = self.kpt2kptirr_sym[k2]
    #                 d1 = self.d_band[i, isym1]
    #                 d2 = self.d_band[j, isym2]
    #                 assert self.kptirr2kpt[i, isym1] == k1
    #                 assert self.kptirr2kpt[j, isym2] == k2
    #                 assert self.kpt2kptirr[k1] == i
    #                 assert self.kpt2kptirr[k2] == j
    #                 ib = np.where(mmn.neighbours[k1] == k2)[0][0]
    #                 assert mmn.neighbours[k1][ib] == k2
    #                 data = mmn.data[k1, ib]
    #                 data = f1(d1) @ data @ f2(d2)
    #                 if ref is None:
    #                     ref = data
    #                     err = 0
    #                 else:
    #                     err = np.linalg.norm(data - ref)
    #                 print(f"   {k1} -> {k2} : {err}")
    #                 maxerr = max(maxerr, err)
    #     return maxerr



class VoidSymmetrizer(SymmetrizerSAWF):

    """
    A fake symmetrizer that does nothing
    Just to be able to use the same code with and without site-symmetry
    """

    def __init__(self, *args, NK=1, **kwargs):
        self.NKirr = NK
        self._NK = NK
        self.kptirr = np.arange(NK)
        self.kptirr2kpt = self.kptirr[:, None]
        self.kpt2kptirr = np.arange(NK)
        self.Nsym = 1

    def U_to_full_BZ(self, U, include_k=None):
        return np.copy(U)

    def __call__(self, X):
        return np.copy(X)

    def symmetrize_wannier_property(self, wannier_property):
        return wannier_property
