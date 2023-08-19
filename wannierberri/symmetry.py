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
""" module to define the Symmetry operations. Contains a general class for Rotation, Mirror, and also some pre-defined shortcuts:

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

"""

import numpy as np
import scipy
import scipy.spatial
import scipy.spatial.transform
from packaging import version as pversion

from scipy.spatial.transform import Rotation as rotmat
from copy import deepcopy
from .__utility import real_recip_lattice
from collections.abc import Iterable

import abc

SYMMETRY_PRECISION = 1e-6


class Symmetry():
    """
    Symmetries that acts on reciprocal space objects, in Cartesian coordinates.
    A k-point vector ``k`` transform as ``self.iTR * self.iInv * (sym.R @ k)``.

    Attributes
    ----------
    R : (3, 3) ndarray
        Proper rotation matrix. Always satisfy ``det(R) = 1``.
    TR : bool
        True if symmetry involves time reversal.
    Inv : bool
        True if symmetry involves spatial inversion.
    """

    def __init__(self, R, TR=False):
        self.TR = TR
        self.Inv = np.linalg.det(R) < 0
        self.R = R * (-1 if self.Inv else 1)
        self.iTR = -1 if self.TR else 1
        self.iInv = -1 if self.Inv else 1

    def show(self):
        print(self)

    def __str__(self):
        return f"rotation: {self.R} , TR: {self.TR} , I: {self.Inv}"

    def __mul__(self, other):
        return Symmetry((self.R @ other.R) * (self.iInv * other.iInv), self.TR != other.TR)

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
                    "all dimensions of rank-{} tensor should be 3, found: {}".format(rank, res.shape[dim - rank:dim]))
        for i in range(dim - rank, dim):
            res = self.rotate(
                res.transpose(tuple(range(i)) + tuple(range(i + 1, dim))
                              + (i, ))).transpose(tuple(range(i)) + (dim - 1, ) + tuple(range(i, dim - 1)))
        if self.TR:
            transformTR(res)
        if self.Inv:
            transformInv(res)
        return res


class Rotation(Symmetry):
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
            raise ValueError("the axis vector is too small : {0}. do you know what you are doing?".format(norm))
        axis = np.array(axis) / norm
        if pversion.parse(scipy.__version__) < pversion.parse("1.4.0"):
            R = rotmat.from_rotvec(2 * np.pi / n * axis / np.linalg.norm(axis)).as_dcm()
        else:
            R = rotmat.from_rotvec(2 * np.pi / n * axis / np.linalg.norm(axis)).as_matrix()
        super().__init__(R)


class Mirror(Symmetry):
    r""" mirror plane perpendicular to ``axis``

    Parameters
    ----------
    axis : Iterable of 3 float numbers
        the normal of the mirror plane in Cartesian coordinates. Length of vector does not matter, but should not be zero
    """

    def __init__(self, axis=[0, 0, 1]):
        super().__init__(-Rotation(2, axis).R)


#some typically used symmetries
Identity = Symmetry(np.eye(3))
Inversion = Symmetry(-np.eye(3))
TimeReversal = Symmetry(np.eye(3), True)
Mx = Mirror([1, 0, 0])
My = Mirror([0, 1, 0])
Mz = Mirror([0, 0, 1])
C2z = Rotation(2, [0, 0, 1])
C3z = Rotation(3, [0, 0, 1])
C4x = Rotation(4, [1, 0, 0])
C4y = Rotation(4, [0, 1, 0])
C4z = Rotation(4, [0, 0, 1])
C6z = Rotation(6, [0, 0, 1])
C2x = Rotation(2, [1, 0, 0])
C2y = Rotation(2, [0, 1, 0])


def product(lst):
    assert isinstance(lst, Iterable)
    assert len(lst) > 0
    res = Identity
    for op in lst[-1::-1]:
        res = op * res
    return res


def from_string(string):
    try:
        res = globals()[string]
        if not isinstance(res, Symmetry):
            raise RuntimeError("string '{}' produced not a Symmetry, but {} of type {}".format(string, res, type(res)))
        return res
    except KeyError:
        raise ValueError(
            f"The symmetry {string} is not defined. Use classes Rotation(n,axis) or Mirror(axis) from wannierberri.symmetry"
        )


def from_string_prod(string):
    try:
        return product([globals()[s] for s in string.split("*")])
    except Exception:
        raise ValueError(f"The symmetry {string} could not be recognized")


class Group():
    r"""Class to store a symmetry point group.

    Parameters
    ----------
    generator_list : list of :class:`~Symmetry` or str
        The generators of the symmetry group.
    recip_lattice : `~numpy.array`
        3x3 array with rows giving the reciprocal lattice basis
    real_lattice : `~numpy.array`
        3x3 array with rows giving the real lattice basis

    Notes
    ----------

      + need to provide either `recip_lattice` or `real_latice`, not both

      + if you only want to generate a symmetric tensor, or to find independent components,  `recip_lattice` and `real_latice`, are not needed
    """

    def __init__(self, generator_list=[], recip_lattice=None, real_lattice=None):
        self.real_lattice, self.recip_lattice = real_recip_lattice(
            real_lattice=real_lattice, recip_lattice=recip_lattice)
        sym_list = [(op if isinstance(op, Symmetry) else from_string_prod(op)) for op in generator_list]
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
        MSG_not_symmetric = (
            " : please check if  the symmetries are consistent with the lattice vectors,"
            + " and that  enough digits were written for the lattice vectors (at least 6-7 after coma)")
        if real_lattice is not None:
            assert self.check_basis_symmetry(self.real_lattice), "real_lattice is not symmetric" + MSG_not_symmetric
        if real_lattice is not None:
            assert self.check_basis_symmetry(self.recip_lattice), "recip_lattice is not symmetric" + MSG_not_symmetric

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
        A = self.symmetrize_tensor(np.random.random((3, ) * rank), TRodd=TRodd, Iodd=Iodd)
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
            indices = [(j, ) + ind for j in (0, 1, 2) for ind in indices]
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
        assert np.all(shape[dim - rank:dim] == 3), "the last rank={} dimensions should be 3, found : {}".format(
            rank, shape)
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



class Transform(abc.ABC):


    def __init__(self,dimension=None):
        """if dimension is not None - it is the only number of cartesian indices
            in the tensor alowed for the transfoprmation
        """
        self.dimension=dimension

    @abc.abstractmethod
    def __call__(self,res):
        pass

    def __eq__(self,other):
        verbose = False  # True for testing
        if verbose:
            print (f"checking equaality of {self} and {other}")
        if not isinstance(other,Transform):
            if verbose: print (f"the other is not Transform, it is {other}")
            return False
        elif self.dimension!=other.dimension :
            if verbose: print (f"dimensions do not fit : {self.dimension} and {other.dimension}")
            return False
        else:
            dim = self.dimension
            if dim is None:
                dim=2
            res1 = random_tensors[dim]
            res2 = np.copy(res1)
            self(res1)
            other(res2)
            check = np.max( abs(res1-res2) )
            if check> 1e-8:
                if verbose:
                    print ("the tensors after transformation are: \n{res1}\n and \n {res2} \n they differ by {check}")
                return False
            else:
                if verbose:
                    print (f"are equal : {self} and {other}")
                return True


class TransformIdent(Transform):

    def __call__(self,res):
        pass

    def __str__(self):
        return ("TransformIdent")

class TransformOdd(Transform):

    def __call__(self,res):
        res[:]=-res[:]

    def __str__(self):
        return ("TransformOdd")

class TransformTrans(Transform):


    def __init__(self,axes=(1,0)):
        """if dimension is not None - it is the only number of cartesian indices
            in the tensor alowed for the transfoprmation
        """
        self.axes=tuple(axes)
        self.dimension=len(axes)

    def __call__(self,res):
        dim0=res.ndim-self.dimension
        trans = tuple(i for i in range(dim0))+tuple(dim0+a for a in self.axes)
        res[:]=res.transpose(trans)

    def __str__(self):
        return (f"TransformTrans({self.axes})")


transform_ident = TransformIdent()
transform_odd   = TransformOdd()
transform_trans = TransformTrans()


class TransformProduct(Transform):

    def __init__(self, transform_list):
        self.transform_list=list(transform_list)
        dimensions = list([t.dimension for t in self.transform_list])
        if None in dimensions:
            if len(set(dimensions))!=1 :
                raise ValueError("either ALL of NONE of the transformatrions in a product should have dimension . found {dimensions}")
            self.dimension = None
        else:
            self.dimension = sum(dimensions)

    def __call__(self,res):
        for tr in self.transform_list:
            tr(res)

    def __str__(self):
        return "Product of [" + ", ".join(str(t) for t in self.transform_list) + " ]"


random_tensors = { 0:np.array([0.80596238+0.34296245j]),
        1: np.array([0.25071929+0.34491236j, 0.04877657+0.17022553j,
         0.39690414+0.50507555j]), 
        2: np.array([[0.470442  +0.95357786j, 0.49501982+0.9856977j ,
         0.70029476+0.22566716j],
        [0.14017942+0.0417328j , 0.67810481+0.91156402j,
         0.99808451+0.2213261j ],
        [0.65149989+0.96146194j, 0.01377349+0.01784413j,
         0.80596238+0.34296245j]]), 
        3: np.array([[[0.72843698+0.60452792j, 0.26848089+0.5645207j ,
         0.5894163 +0.62908298j],
        [0.70418612+0.65981844j, 0.50711601+0.2101012j ,
         0.97623005+0.94589939j],
        [0.34623831+0.42267871j, 0.13230829+0.60132712j,
         0.17269701+0.66593089j]],

       [[0.62252721+0.25972703j, 0.38588518+0.44633076j,
         0.26748758+0.82460679j],
        [0.20973642+0.87918606j, 0.35768832+0.36513199j,
         0.35634033+0.32618218j],
        [0.41633665+0.30948491j, 0.38232737+0.61776951j,
         0.68191988+0.28190326j]],

       [[0.82881472+0.19917847j, 0.5504073 +0.4946653j ,
         0.07215705+0.97850916j],
        [0.99621333+0.38722484j, 0.97585631+0.5840961j ,
         0.22429663+0.14430257j],
        [0.84854584+0.27526566j, 0.03537387+0.13132689j,
         0.98373851+0.06963218j]]]), 
        4: np.array([[[[0.30031749+0.68469709j, 0.58659937+0.24525988j,
          0.0597766 +0.05075574j],
         [0.12479131+0.90183041j, 0.80639707+0.66720799j,
          0.51740499+0.04500371j],
         [0.29795983+0.66040221j, 0.60433079+0.65185754j,
          0.3298667 +0.60153711j]],

        [[0.7112296 +0.9846345j , 0.95713785+0.20224548j,
          0.99630118+0.37702615j],
         [0.97142677+0.97842682j, 0.8466332 +0.40146803j,
          0.89224514+0.78101255j],
         [0.05634177+0.45136221j, 0.3366071 +0.04759113j,
          0.18878582+0.60663714j]],

        [[0.76642002+0.3875563j , 0.2775941 +0.48162047j,
          0.52805749+0.0800852j ],
         [0.44986335+0.83752967j, 0.3317307 +0.87730627j,
          0.00941376+0.76739382j],
         [0.21629705+0.63123142j, 0.53396577+0.88771443j,
          0.13835306+0.76693498j]]],


       [[[0.74012236+0.59049934j, 0.87205772+0.53257876j,
          0.53048185+0.75441839j],
         [0.90906743+0.36159884j, 0.85192814+0.00565488j,
          0.2228864 +0.54438124j],
         [0.05848741+0.23302882j, 0.07905387+0.57442505j,
          0.07725599+0.94095147j]],

        [[0.6836054 +0.12826987j, 0.79610868+0.02788082j,
          0.37195177+0.26364243j],
         [0.64980615+0.23827312j, 0.27448158+0.68271081j,
          0.52409896+0.64693385j],
         [0.87922886+0.17594415j, 0.75541579+0.19611622j,
          0.07322409+0.42616649j]],

        [[0.26586886+0.38057158j, 0.04297439+0.04077598j,
          0.01259778+0.94651108j],
         [0.40609943+0.8285334j , 0.45170852+0.71981179j,
          0.32755724+0.74186577j],
         [0.93301137+0.78309037j, 0.94662744+0.14698388j,
          0.73885154+0.80563412j]]],


       [[[0.85501232+0.52963992j, 0.78767093+0.34986533j,
          0.52119841+0.78410604j],
         [0.39439026+0.56389553j, 0.36449086+0.4192848j ,
          0.86797015+0.19484826j],
         [0.97773862+0.2100186j , 0.10800484+0.48067086j,
          0.19862437+0.90569614j]],

        [[0.10677531+0.57929921j, 0.12707442+0.67470644j,
          0.4516258 +0.62156642j],
         [0.13300833+0.55650137j, 0.40436296+0.5847801j ,
          0.45804265+0.71206061j],
         [0.10762864+0.77615143j, 0.90595254+0.55383336j,
          0.68232845+0.6914805j ]],

        [[0.17390251+0.84810437j, 0.78534032+0.55692283j,
          0.39799596+0.16364989j],
         [0.80922176+0.38966705j, 0.44245086+0.34459141j,
          0.43973888+0.46929441j],
         [0.40355698+0.93692636j, 0.78189567+0.84883019j,
          0.02322222+0.28217111j]]]])}

