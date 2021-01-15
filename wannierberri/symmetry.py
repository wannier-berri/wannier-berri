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
from lazy_property import LazyProperty as Lazy
from .__utility import real_recip_lattice, is_round
from collections import Iterable

SYMMETRY_PRECISION=1e-6

class Symmetry():

    def __init__(self,R,TR=False):
        self.TR=TR
        self.Inv=np.linalg.det(R)<0
        self.R=R*(-1 if self.Inv else 1)
            
    def show(self):
        print (self)

    def __str__(self):
        return ("rotation: {0}, TR:{1} , I:{1}".format(self.R,self.TR,self.Inv))

    @Lazy
    def iTR(self):
        return -1 if self.TR else 1

    @Lazy
    def iInv(self):
        return -1 if self.Inv else 1

    def __mul__(self,other):
        return Symmetry(self.R.dot(other.R)*(self.iInv*other.iInv),self.TR!=other.TR)

    def __eq__(self,other):
        return np.linalg.norm(self.R-other.R)<1e-14 and self.TR==other.TR and self.Inv==other.Inv

    def copy(self):
        return deepcopy(self)


    def transform_reduced_vector(self,vec,basis):
        return np.dot(vec, basis.dot(self.R.T).dot(np.linalg.inv(basis)))*(self.iTR*self.iInv)

    def rotate(self,res):
        return np.dot(res,self.R.T)


    def transform_tensor(self,data,rank,TRodd=False,Iodd=False):
        res=np.copy(data)
        dim=len(res.shape)
        if rank>0:
          if not  np.all( np.array(res.shape[dim-rank:dim])==3):
            raise RuntimeError("all dimensions of rank-{} tensor should be 3, found: {}".format(rank,res.shape[dim-rank:dim]))
        for i in range(dim-rank,dim):
            res=self.rotate(res.transpose( tuple(range(i))+tuple(range(i+1,dim))+(i,) ) ).transpose( 
                    tuple(range(i))+(dim-1,)+tuple(range(i,dim-1))  )
        if (self.TR and TRodd)!=(self.Inv and Iodd):
            res=-res
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
    def __init__(self,n,axis=[0,0,1]):
        if not isinstance(n,int):
            raise ValueError("Only integer rotations are supported")
        if n==0:
            raise ValueError("rotations with n=0 are nonsense")
        norm=np.linalg.norm(axis)
        if norm<1e-10:
            raise ValueError("the axis vector is too small : {0}. do you know what you are doing?".format(norm))
        axis=np.array(axis)/norm
        if pversion.parse(scipy.__version__)<pversion.parse("1.6.0"):
            R=rotmat.from_rotvec(2*np.pi/n*axis/np.linalg.norm(axis)).as_dcm()
        else:
            R=rotmat.from_rotvec(2*np.pi/n*axis/np.linalg.norm(axis)).as_matrix()
        super(Rotation, self).__init__(R )



class Mirror(Symmetry):
    r""" mirror plane perpendicular to ``axis``  

    Parameters
    ----------
    axis : Iterable of 3 float numbers
        the normal of the mirror plane in Cartesian coordinates. Length of vector does not matter, but should not be zero
    """
    def __init__(self,axis=[0,0,1]):
         super(Mirror, self).__init__( -Rotation(2,axis).R )




#some typically used symmetries
Identity =Symmetry( np.eye(3))
Inversion=Symmetry(-np.eye(3))
TimeReversal=Symmetry( np.eye(3),True)
Mx=Mirror([1,0,0])
My=Mirror([0,1,0])
Mz=Mirror([0,0,1])
C2z=Rotation(2,[0,0,1])
C3z=Rotation(3,[0,0,1])
C4x=Rotation(4,[1,0,0])
C4y=Rotation(4,[0,1,0])
C4z=Rotation(4,[0,0,1])
C6z=Rotation(6,[0,0,1])
C2x=Rotation(2,[1,0,0])
C2y=Rotation(2,[0,1,0])

def product(lst):
    assert isinstance(lst,Iterable) 
    assert len(lst)>0
    res=Identity
    for op in lst[-1::-1]:
        res=op*res
    return res

def from_string(string):
    try:
        res=globals()[string]
        if not isinstance(res,Symmetry):
           raise RuntimeError("string '{}' produced not a Symmetry, but {} of type {}".format(string,res,type(res)))
        return res
    except KeyError as err:
        raise ValueError("The symmetry {} is not defined. Use classes Rotation(n,axis) or Mirror(axis) from wannierberri.symmetry ".format(string))


def from_string_prod(string):
    try:
        return product([globals()[s] for s in string.split("*")])
    except Exception as err:
        raise ValueError("The symmetry {string} could not be recognuized : {}".format(err))

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

    def __init__(self,generator_list=[],recip_lattice=None,real_lattice=None):
        self.real_lattice,self.recip_lattice=real_recip_lattice(real_lattice=real_lattice,recip_lattice=recip_lattice)
        sym_list=[(op if isinstance(op,Symmetry) else from_string_prod(op) )    for op in generator_list  ]
        print (sym_list)
        if len(sym_list)==0:
            sym_list=[Identity]
        cnt=0
        while True:
            cnt+=1
            if cnt>1000:
                raise RuntimeError("Cannot define a finite group")
            lenold=len(sym_list)
            for s1 in sym_list:
              for s2 in sym_list:
                s3=s1*s2
                new = True
                for s4 in sym_list:
                   if s3==s4:
                      new=False
                      break
                if new:
                    sym_list.append(s3)
            lennew=len(sym_list)
            if lenold==lennew:
                break
        self.symmetries=sym_list
        MSG_not_symmetric=(" : please check if  the symmetries are consistent with the lattice vectors,"+
                    " and that  enough digits were written for the lattice vectors (at least 6-7 after coma)" )
        if real_lattice is not None:
            assert self.check_basis_symmetry(self.real_lattice) , "real_lattice is not symmetric" + MSG_not_symmetric
        if real_lattice is not None:
            assert self.check_basis_symmetry(self.recip_lattice) , "recip_lattice is not symmetric" + MSG_not_symmetric
#        print ("BASIS={}".format(self.basis))

    def check_basis_symmetry(self,basis,tol=1e-6,rel_tol=None):
        "returns True if the basis is symmetric"
        if rel_tol is not None:
            tol=rel_tol*tol
        eye=np.eye(3)
        for sym in self.symmetries:
            basis_rot=sym.transform_reduced_vector(eye,basis)
            if np.abs(np.round(basis_rot)-basis_rot).max()> tol :
               return False
        return True

    def symmetric_grid(self,nk):
        return self.check_basis_symmetry(self.recip_lattice/np.array(nk)[:,None],rel_tol=10) 
        
    @property
    def size(self):
        return len(self.symmetries)
    
    def symmetrize_axial_vector(self,res):
        return sum(s.transform_axial_vector(res) for s in self.symmetries)/self.size

    def symmetrize_polar_vector(self,res):
        return sum(s.transform_polar_vector(res) for s in self.symmetries)/self.size

    def symmetrize(self,result):
        return sum(result.transform(s) for s in self.symmetries)/self.size

    def gen_symmetric_tensor(self,rank,TRodd,Iodd):
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
        A=self.symmetrize_tensor(np.random.random((3,)*rank),TRodd=TRodd,Iodd=Iodd)
        A[abs(A)<1e-14]=0
        return A



    def get_symmetric_components(self,rank,TRodd,Iodd):
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
        assert rank>=0
        A=self.gen_symmetric_tensor(rank,TRodd,Iodd)
        indices=[ () ]
        indices_xyz=[""]
        for i in range(A.ndim):
            indices=[ (j,)+ind for j in (0,1,2) for ind in indices ]
            indices_xyz=[ a+ind for a in "xyz" for ind in indices_xyz ]
#        print (indices)
#        print (indices_xyz)
        equalities={0:["0"]}
        tol=1e-14
        for ind,ind_xyz in zip(indices,indices_xyz):
            value=A[ind]
            found=False
            for val,comp in equalities.items():
                if abs(val-value)<tol:
                    equalities[val].append(ind_xyz)
                    found=True
                    break
            if not found:
              for val,comp in equalities.items():
                if abs(val+value)<tol:
                    equalities[val].append('-'+ind_xyz)
                    found=True
                    break
            if not found:
              equalities[value]=[ind_xyz]
        return ["=".join(val) for val in equalities.values()]






    def symmetrize_tensor(self,data,TRodd,Iodd,rank=None):
        dim=data.ndim
        if rank is None:
            rank=dim
        shape=np.array(data.shape)
        assert np.all(shape[dim-rank:dim]==3), "the last rank={} dimensions should be 3, found : {}".format(rank,shape)
        return sum( s.transform_tensor(data,TRodd=TRodd,Iodd=Iodd,rank=rank)  for s in self.symmetries)/self.size
    
    def star(self,k):
        st=[S.transform_reduced_vector(k,self.recip_lattice) for S in self.symmetries]
        for i in range(len(st)-1,0,-1):
           diff=np.array(st[:i])-np.array(st[i])[None,:]
           if np.linalg.norm (diff-diff.round() ,axis=-1).min()<SYMMETRY_PRECISION:
               del st[i]
        return np.array(st)


#    def star_int(self,k):
#        k=np.array(k)
#        st=[S.transform_reduced_vector(k,self.recip_lattice) for S in self.symmetries]
#        return  set([ tuple(np.array(np.round(k),dtype=int)) for k in st if is_round(k,prec=1e-6) ])

#    def split_kpts_to_shells(self,kpts):
#        kpts=tuple(k for k in kpts)


if __name__ == '__main__':
    s=Rotation(4)
    basis=np.array([[1,0,-0.3],[0,1,-0.3],[0,0,0.6]])
    group=Group([s],basis)
#    v=[[1,0,0],[0,1,0],[0,0,1]]    
    v=[-0.375,-0.375,0.375]
#    basis=np.array([[0.5,np.sqrt(3)/2,0],[0.5,-np.sqrt(3)/2,0],[0,0,1]])
#    print (s.transform_vector(v,basis))
    print (group.star(v))
    
    

