import numpy as np
from scipy.spatial.transform import Rotation as rotmat
from copy import deepcopy





class Symmetry():

    def __init__(self,R):
        self.R=R
        
    def show(self):
        print (self.R)

    def dot(self,other):
        return Symmetry(self.R.dot(other.R))
        
    def __eq__(self,other):
        return np.linalg.norm(self.R-other.R)<1e-14
        
    def copy(self):
        return deepcopy(self)
    
    def power(self):
        if self==Identity:
            return 1
        a=self.copy()
        for i in range(100):
            a=a.dot(self)
            if a==Identity:
                return i+2
        raise RuntimeError("for symmetry {0} the power is greater than 100".format(self.R))
        
    def transform_result(self,res):
        return np.dot(res,self.R.T)*np.linalg.det(self.R)

#    def transform_vector(self,vec,basis=np.eye(3)):
#        return basis.T.dot(vec).dot(self.R.T).dot(np.linalg.inv(basis))
    
    def transform_vector(self,vec,basis=np.eye(3)):
        return np.dot(vec, basis.dot(self.R.T).dot(np.linalg.inv(basis)))

    
Identity =Symmetry( np.eye(3))
Inversion=Symmetry(-np.eye(3))

#class Identity(Symmetry):
#    def __init__(self):
#        super(Identity, self).__init__(np.eye(3))
    
        
#class Inversion(Symmetry):
#    def __init__(self):
#        super(Inversion, self).__init__(-np.eye(3))
        

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
         super(Mirror, self).__init__(Rotation(2,axis).dot(Inversion).R)



class Group():

    def __init__(self,generator_list=[Identity],basis=np.eye(3)):
        sym_list=generator_list
        cnt=0
        while True:
            cnt+=1
            if cnt>1000:
                raise RuntimeError("Cannot define a finite group")
            lenold=len(sym_list)
            for s1 in sym_list:
              for s2 in sym_list:
                s3=s1.dot(s2)
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
        
    @property
    def size(self):
        return len(self.symmetries)
    
    def symmetrize(self,res):
        return sum(s.transform_result(res) for s in self.symmetries)/self.size

    def star(self,k):
        return np.array([S.transform_vector(k) for S in self.symmetries])


if __name__ == '__main__':
    s=Rotation(3)
    v=[1,1,0]
    v=[[0,1,0],[1,0,0]]
    basis=np.array([[0.5,np.sqrt(3)/2,0],[0.5,-np.sqrt(3)/2,0],[0,0,1]])
#    print (s.transform_vector(v,basis))
    print (s.transform_vector_2(v,basis))
    
    


#for s in findAll([Inversion,Rotation(4),Rotation(3,[0,0,1])]):
#    s.show()


#basis=np.array([[0.5,np.sqrt(3)/2,0],[0.5,-np.sqrt(3)/2,0],[0,0,1]])
#print(basis.T.dot([1,0,0]))
#print(Rotation(3).transform_vector([1,0,0],basis=basis) )



#exit()

#Inversion.show()
#Identity.show()
#Rotation(1).show()
#Rotation(3).show()
#Rotation(3,axis=[1e-5,0,0]).show()
#print ( Mirror(axis=[1,0,0]).dot(Mirror(axis=[-1,np.sqrt(3),0])).power())
#print ( Mirror([1,2,3])==Inversion.dot(Rotation(2,[-1,-2,-3])))
