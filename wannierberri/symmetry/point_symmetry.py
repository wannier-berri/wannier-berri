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
# ------------------------------------------------------------
""" module to define the Pointgroup Symmetry operations, acting on the tensors in the reciprocal space

Contains a general class for Rotation, Mirror, and also some pre-defined shortcuts:

+ Identity =Symmetry( np.eye(3))

+ Inversion=Symmetry(-np.eye(3))

+ TimeReversal=Symmetry( np.eye(3),True)

+ Mx=Mirror([1,0,0])

+ My=Mirror([0,1,0])

+ Mz=Mirror([0,0,1])

+ C2z=Rotation(2,[0,0,1])

+ C3z=Rotation(3,[0,0,1])

+ C4x=Rotation(4,[1,0,0])

+ C4y=Rotation(4,[0,1,0])

+ C4z=Rotation(4,[0,0,1])

+ C6z=Rotation(6,[0,0,1])

+ C2x=Rotation(2,[1,0,0])

+ C2y=Rotation(2,[0,1,0])

+ dict_sym = {"C2x": C2x, ...} - a dictionary with the pre-defined symmetries

"""

import numpy as np
import scipy
import scipy.spatial
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as rotmat
from packaging import version as pversion
import warnings
from copy import deepcopy
from ..utility import real_recip_lattice
from collections.abc import Iterable

SYMMETRY_PRECISION = 1e-6


class PointSymmetry:
    """
    Symmetries that acts on reciprocal space objects, in Cartesian coordinates.
    A k-point vector ``k`` transform as ``self.iTR * self.iInv * (sym.R @ k)``.

    Parameters
    ------------
    R : (3, 3) ndarray
        Rotation matrix.  ``det(R) = 1 or -1``.
    TR : bool
        True if symmetry involves time reversal.


    Attributes
    ----------
    R : (3, 3) ndarray
        Proper rotation matrix. Always satisfy ``det(R) = 1``.
    TR : bool
        True if symmetry involves time reversal.
    Inv : bool
        True if symmetry involves spatial inversion. (i.e. if on input det(R) was -1)
    """

    def __init__(self, R, TR=False):
        self.TR = TR
        self.Inv = np.linalg.det(R) < 0
        self.R = R * (-1 if self.Inv else 1)
        self.iTR = -1 if self.TR else 1
        self.iInv = -1 if self.Inv else 1

    def as_dict(self):
        return dict(R=self.R * (-1 if self.Inv else 1), TR=self.TR)

    def show(self):
        print(self)

    def __str__(self):
        return f"rotation:\n{np.round(self.R, decimals=4)} , TR: {self.TR} , I: {self.Inv}"

    def __mul__(self, other):
        return PointSymmetry((self.R @ other.R) * (self.iInv * other.iInv), self.TR != other.TR)

    def __eq__(self, other):
        return np.linalg.norm(self.R - other.R) < 1e-12 and self.TR == other.TR and self.Inv == other.Inv

    def copy(self):
        return deepcopy(self)

    def transform_reduced_vector(self, vec, basis):
        return vec @ (basis @ self.R.T @ np.linalg.inv(basis)) * (self.iTR * self.iInv)

    def rotate(self, res):
        return res @ self.R.T

    def transform_tensor(self, data, rank, transformTR, transformInv):
        res = np.copy(data)
        dim = len(res.shape)
        if rank > 0:
            if not np.all(np.array(res.shape[dim - rank:dim]) == 3):
                raise RuntimeError(
                    f"all dimensions of rank-{rank} tensor should be 3, found: {res.shape[dim - rank:dim]}")
        for i in range(dim - rank, dim):
            res = self.rotate(
                res.transpose(tuple(range(i)) + tuple(range(i + 1, dim)) +
                              (i,))).transpose(tuple(range(i)) + (dim - 1,) + tuple(range(i, dim - 1)))
        if self.TR:
            transformTR(res)
        #            res = res.conj()
        #        if (self.TR and TRodd) != (self.Inv and Iodd):
        #            res = -res
        #        if self.TR and TRtrans:
        #            res = res.swapaxes(dim - rank, dim - rank + 1).conj()
        if self.Inv:
            transformInv(res)
        return res


class Rotation(PointSymmetry):
    r""" n-fold rotation around the ``axis``

    Parameters
    ----------
    n : int
        1,2,3,4 or 6. Defines the rotation angle :math:`2\pi/n`
    axis : Iterable of 3 float numbers
        the rotation axis in Cartesian coordinates. Length of vector does not matter, but should not be zero.
    """

    def __init__(self, n, axis=[0, 0, 1]):
        if not isinstance(n, int):
            raise ValueError("Only integer rotations are supported")
        if n == 0:
            raise ValueError("rotations with n=0 are nonsense")
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            raise ValueError(f"the axis vector is too small : {norm}. do you know what you are doing?")
        axis = np.array(axis) / norm
        if pversion.parse(scipy.__version__) < pversion.parse("1.4.0"):
            R = rotmat.from_rotvec(2 * np.pi / n * axis / np.linalg.norm(axis)).as_dcm()
        else:
            R = rotmat.from_rotvec(2 * np.pi / n * axis / np.linalg.norm(axis)).as_matrix()
        super().__init__(R)


class Mirror(PointSymmetry):
    r""" mirror plane perpendicular to ``axis``

    Parameters
    ----------
    axis : Iterable of 3 float numbers
        the normal of the mirror plane in Cartesian coordinates. Length of vector does not matter, but should not be zero
    """

    def __init__(self, axis=[0, 0, 1]):
        super().__init__(-Rotation(2, axis).R)


# some typically used symmetries
Identity = PointSymmetry(np.eye(3))
Inversion = PointSymmetry(-np.eye(3))
TimeReversal = PointSymmetry(np.eye(3), True)
Mx = Mirror([1, 0, 0])
My = Mirror([0, 1, 0])
Mz = Mirror([0, 0, 1])
C2x = Rotation(2, [1, 0, 0])
C2y = Rotation(2, [0, 1, 0])
C2z = Rotation(2, [0, 0, 1])
C3z = Rotation(3, [0, 0, 1])
C4x = Rotation(4, [1, 0, 0])
C4y = Rotation(4, [0, 1, 0])
C4z = Rotation(4, [0, 0, 1])
C6z = Rotation(6, [0, 0, 1])

dict_sym = {"Identity": Identity,
            "Inversion": Inversion,
            "TimeReversal": TimeReversal,
            "Mx": Mx,
            "My": My,
            "Mz": Mz,
            "C2x": C2x,
            "C2y": C2y,
            "C2z": C2z,
            "C3z": C3z,
            "C4x": C4x,
            "C4y": C4y,
            "C4z": C4z,
            "C6z": C6z
            }


def product(lst):
    assert isinstance(lst, Iterable)
    assert len(lst) > 0
    res = Identity
    for op in lst[-1::-1]:
        res = op * res
    return res


def from_string(string):
    try:
        res = dict_sym[string]
        if not isinstance(res, PointSymmetry):
            raise RuntimeError(f"string '{string}' produced not a Symmetry, but {res} of type {type(res)}")
        return res
    except KeyError:
        raise ValueError(
            f"The symmetry {string} is not defined. Use classes Rotation(n,axis) or Mirror(axis) from wannierberri.symmetry"
        )


def from_string_prod(string):
    try:
        return product([from_string(s) for s in string.split("*")])
    except Exception as e:
        raise ValueError(f"The symmetry {string} could not be recognized:  {e}")


class PointGroup():
    r"""Class to store a symmetry point group.

    Parameters
    ----------
    generator_list : list of :class:`~Symmetry` or str
        The generators of the symmetry group.
    recip_lattice : `~numpy.array`
        3x3 array with rows giving the reciprocal lattice basis
    real_lattice : `~numpy.array`
        3x3 array with rows giving the real lattice basis
    dictionary : dict
        dictionary with the symmetry operations and the lattice vectors
    spacegroup : irrep.SpaceGroup
        the spacegroup to which the pointgroup belongs. If provided, other parameters are ignored.

    Notes
    ----------

      + need to provide either `recip_lattice` or `real_latice`, not both

      + if you only want to generate a symmetric tensor, or to find independent components,  `recip_lattice` and `real_latice`, are not needed
    """

    def __init__(self, generator_list=[], recip_lattice=None, real_lattice=None, dictionary=None,
                 spacegroup=None):
        if spacegroup is not None:
            point_symmetries = []
            for symop in spacegroup.symmetries:
                R = symop.rotation_cart
                TR = symop.time_reversal
                point_symmetries.append(PointSymmetry(R, TR))
            self.__init__(generator_list=point_symmetries,
                          real_lattice=spacegroup.Lattice)
        elif dictionary is not None:
            nsym = dictionary['nsym']
            gen_lst = []
            for i in range(nsym):
                d = {}
                for k, v in dictionary.items():
                    l = self._symm_dict_prefix(i)
                    if k.startswith(l):
                        d[k[len(l):]] = v
                gen_lst.append(PointSymmetry(**d))
            self.__init__(generator_list=gen_lst, real_lattice=dictionary['real_lattice'], dictionary=None)
        else:
            self.real_lattice, self.recip_lattice = real_recip_lattice(
                real_lattice=real_lattice, recip_lattice=recip_lattice)
            sym_list = [(op if isinstance(op, PointSymmetry) else from_string_prod(op)) for op in generator_list]
            if len(sym_list) == 0:
                sym_list = [Identity]

            while True:
                lenold = len(sym_list)
                for s1 in sym_list:
                    for s2 in sym_list:
                        s3 = s1 * s2
                        if s3 not in sym_list:
                            sym_list.append(s3)
                            if len(sym_list) > 1000:
                                raise RuntimeError("Cannot define a finite group")
                if len(sym_list) == lenold:
                    break

            self.symmetries = sym_list
            msg_not_symmetric = (
                " : please check if  the symmetries are consistent with the lattice vectors," +
                " and that  enough digits were written for the lattice vectors (at least 6-7 after coma)")
            if real_lattice is not None:
                assert self.check_basis_symmetry(self.real_lattice), "real_lattice is not symmetric" + msg_not_symmetric
            if real_lattice is not None:
                assert self.check_basis_symmetry(self.recip_lattice), "recip_lattice is not symmetric" + msg_not_symmetric

    def _symm_dict_prefix(self, i):
        return f'symm{i}_'

    def as_dict(self):
        nsym = len(self.symmetries)
        ret = dict(real_lattice=self.recip_lattice,
                   nsym=nsym)
        for i, s in enumerate(self.symmetries):
            for k, v in s.as_dict().items():
                ret[self._symm_dict_prefix(i) + k] = v
        return ret

    def __str__(self):
        s = f"Real_lattice:\n{np.round(self.real_lattice, decimals=4)}\n Recip. Lattice:\n {np.round(self.recip_lattice, decimals=4)}\n size:{self.size}\nOperations:\n"
        for i, sym in enumerate(self.symmetries):
            s += f"{i}:\n{sym}\n"
        return s

    def check_basis_symmetry(self, basis, tol=1e-6, rel_tol=None):
        "returns True if the basis is symmetric"
        if rel_tol is not None:
            tol = rel_tol * tol
        eye = np.eye(3)
        for sym in self.symmetries:
            basis_rot = sym.transform_reduced_vector(eye, basis)
            if np.abs(np.round(basis_rot) - basis_rot).max() > tol:
                return False
        return True

    def symmetric_grid(self, nk):
        return self.check_basis_symmetry(self.recip_lattice / np.array(nk)[:, None], rel_tol=10)

    @property
    def size(self):
        return len(self.symmetries)

    def symmetrize_axial_vector(self, res):
        return sum(s.transform_axial_vector(res) for s in self.symmetries) / self.size

    def symmetrize_polar_vector(self, res):
        return sum(s.transform_polar_vector(res) for s in self.symmetries) / self.size

    def symmetrize(self, result):
        return sum(result.transform(s) for s in self.symmetries) / self.size

    def gen_symmetric_tensor(self, rank, TRodd, Iodd):
        r"""generates a random tensor, which respects the given symmetry pointgroup. May be used to get an idea, what components of the tensr are allowed by the symmetry.

        Parameters
        ----------
        rank : int
            rank of the tensor
        TRodd : bool
            True if the tensor is odd under time-reversal, False otherwise
        Iodd : bool
            True if the tensor is odd under inversion, False otherwise

        Returns
        --------
        `numpy.array(float)`
             :math:`3 \times 3\times \ldots` array respecting the symmetry
        """
        transform_TR = transform_odd if TRodd else transform_ident
        transform_I = transform_odd if Iodd else transform_ident
        A = self.symmetrize_tensor(np.random.random((3,) * rank), transformTR=transform_TR, transformInv=transform_I)
        A[abs(A) < 1e-14] = 0
        return A

    def get_symmetric_components(self, rank, TRodd, Iodd):
        r"""writes which components of a tensor nonzero, and which are equal (or opposite)

        Parameters
        ----------
        rank : int
            rank of the tensor
        TRodd : bool
            True if the tensor is odd under time-reversal, False otherwise
        Iodd : bool
            True if the tensor is odd under inversion, False otherwise

        Returns
        -------
        a list of str
            list of eualities, e.g. ['0=xxy=xxz=...', 'xxx=-xyy=-yxy=-yyx', 'xyz=-yxz', 'xzy=-yzx', 'zxy=-zyx']
        """
        assert rank >= 0
        A = self.gen_symmetric_tensor(rank, TRodd, Iodd)
        indices = [()]
        indices_xyz = [""]
        for i in range(A.ndim):
            indices = [(j,) + ind for j in (0, 1, 2) for ind in indices]
            indices_xyz = [a + ind for a in "xyz" for ind in indices_xyz]
        equalities = {0: ["0"]}
        tol = 1e-14
        for ind, ind_xyz in zip(indices, indices_xyz):
            value = A[ind]
            found = False
            for val, comp in equalities.items():
                if abs(val - value) < tol:
                    equalities[val].append(ind_xyz)
                    found = True
                    break
            if not found:
                for val, comp in equalities.items():
                    if abs(val + value) < tol:
                        equalities[val].append('-' + ind_xyz)
                        found = True
                        break
            if not found:
                equalities[value] = [ind_xyz]
        return ["=".join(val) for val in equalities.values()]

    def symmetrize_tensor(self, data, transformTR, transformInv, rank=None):
        dim = data.ndim
        if rank is None:
            rank = dim
        shape = np.array(data.shape)
        assert np.all(shape[dim - rank:dim] == 3), f"the last rank={rank} dimensions should be 3, found : {shape}"
        return sum(s.transform_tensor(data, rank=rank,
                                      transformTR=transformTR, transformInv=transformInv)
                   for s in self.symmetries) / self.size

    def star(self, k):
        st = [S.transform_reduced_vector(k, self.recip_lattice) for S in self.symmetries]
        for i in range(len(st) - 1, 0, -1):
            diff = np.array(st[:i]) - np.array(st[i])[None, :]
            if np.linalg.norm(diff - diff.round(), axis=-1).min() < SYMMETRY_PRECISION:
                del st[i]
        return np.array(st)


########
# transformation of tensors (inplace)
#########


class Transform:
    r"""Describes transformation of a tensor under inversion or time-reversal

    Parameters
    ----------
    factor : int
        multiplication factor (+1 or -1)
    conj : bool
        apply complex conjugation
    transpose_axes : tuple
        how to permute the axes of the tensor.

    Note
    -----
    * Pre-defined transforms are :
        + `transform_ident = Transform()`
        + `transform_odd   = Transform(factor=-1)`
        + `transform_trans = Transform(transpose_axes=(1,0))`
        """

    def __init__(self, factor=1, conj=False, transpose_axes=None):
        self.conj = conj
        self.factor = factor
        assert factor in (1, -1), f"factor is {factor}"
        self.transpose_axes = transpose_axes

    def as_dict(self):
        return {k: self.__getattribute__(k) for k in ["conj", "factor", "transpose_axes"]}

    def __str__(self):
        return f"Transform(factor={self.factor}, conj={self.conj}, transpose_axes={self.transpose_axes}"

    def __call__(self, res):
        if self.transpose_axes is not None:
            dim0 = res.ndim - len(self.transpose_axes)
            trans = tuple(i for i in range(dim0)) + tuple(dim0 + a for a in self.transpose_axes)
            res[:] = res.transpose(trans)
        if self.conj:
            res[:] = res[:].conj()
        res[:] *= self.factor
        return res

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return False
        for key in "factor", "conj", "transpose_axes":
            if getattr(self, key) != getattr(other, key):
                return False
        return True


class TransformProduct(Transform):
    """Constructs a :class:`~Transform`
    from a list of :class:`~Transform`
    """

    def __init__(self, transform_list):
        transform_list = list(transform_list)
        conj_list = list([t.conj for t in transform_list])
        if len(set(conj_list)) != 1:
            raise ValueError(
                "either ALL of NONE of the transformations in a product should have conjugation .  {conj_list}")
        if np.any([t.transpose_axes is not None for t in transform_list]):
            raise NotImplementedError("Product of transformations including transposing is not implemented")
        super().__init__(factor=np.prod([t.factor for t in transform_list]), conj=conj_list[0])


transform_ident = Transform()
transform_odd = Transform(factor=-1)
transform_odd_conj = Transform(factor=-1, conj=True)
transform_odd_trans_021 = Transform(factor=-1, transpose_axes=(0, 2, 1))
transform_trans = Transform(transpose_axes=(1, 0))


def transform_from_dict(dic, key):
    """Finds a transform in a dictionary and returns it.
    if not found, returns None"""
    if key not in dic:
        return None
    elif dic[key] is None:
        return None
    else:
        d = dic[key].item()
        if isinstance(d, dict):
            return Transform(**d)
        elif isinstance(d, str):
            warnings.warn("transform read as string from file, recognized as None")
            return None
        else:
            return ValueError(f"wrong type of transform[{key}] in the npz file:{type(d)}")
