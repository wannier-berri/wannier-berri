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
import scipy
import scipy.spatial
import scipy.spatial.transform

from scipy.spatial.transform import Rotation as rotmat
from copy import deepcopy
from lazy_property import LazyProperty as Lazy

SYMMETRY_PRECISION=1e-6

class Symmetry():

    def __init__(self,R,TR=False):
        self.TR=TR
        self.Inv=np.linalg.det(R)<0
        self.R=R*(-1 if self.Inv else 1)
            
    def show(self):
        print ("rotation: {0}, TR:{1} , I:{1}".format(self.R,self.TR,self.Inv))

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


    def transform_k_vector(self,vec,basis=np.eye(3)):
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





Identity =Symmetry( np.eye(3))
Inversion=Symmetry(-np.eye(3))
TimeReversal=Symmetry( np.eye(3),True)
        

class Rotation(Symmetry):
    def __init__(self,n,axis=[0,0,1]):
        if not isinstance(n,int):
            raise ValueError("Only integer rotations are supported")
        if n==0:
            raise ValueError("rotations with n=0 are nonsense")
        norm=np.linalg.norm(axis)
        if norm<1e-10:
            raise ValueError("the axis vector is too small : {0}. do you know what you are doing?".format(norm))
        axis=np.array(axis)/norm
        R=rotmat.from_rotvec(2*np.pi/n*axis/np.linalg.norm(axis)).as_dcm()
        super(Rotation, self).__init__(R )



class Mirror(Symmetry):
    def __init__(self,axis=[0,0,1]):
         super(Mirror, self).__init__( (Rotation(2,axis)*Inversion).R )




#some typically used symmetries
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




class Group():

    def __init__(self,generator_list=[],basis=np.eye(3)):
        sym_list=generator_list
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
        self.basis=basis
        print ("BASIS={}".format(self.basis))
        
    @property
    def size(self):
        return len(self.symmetries)
    
    def symmetrize_axial_vector(self,res):
        return sum(s.transform_axial_vector(res) for s in self.symmetries)/self.size

    def symmetrize_polar_vector(self,res):
        return sum(s.transform_polar_vector(res) for s in self.symmetries)/self.size

    def symmetrize(self,result):
        return sum(result.transform(s) for s in self.symmetries)/self.size
    
    def star(self,k):
        st=[S.transform_k_vector(k,self.basis) for S in self.symmetries]
        for i in range(len(st)-1,0,-1):
           diff=np.array(st[:i])-np.array(st[i])[None,:]
           if np.linalg.norm (diff-diff.round() ,axis=-1).min()<SYMMETRY_PRECISION:
               del st[i]
        return np.array(st)


if __name__ == '__main__':
    s=Rotation(4)
    basis=np.array([[1,0,-0.3],[0,1,-0.3],[0,0,0.6]])
    group=Group([s],basis)
#    v=[[1,0,0],[0,1,0],[0,0,1]]    
    v=[-0.375,-0.375,0.375]
#    basis=np.array([[0.5,np.sqrt(3)/2,0],[0.5,-np.sqrt(3)/2,0],[0,0,1]])
#    print (s.transform_vector(v,basis))
    print (group.star(v))
    
    

