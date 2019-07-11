import numpy as np


def _check_eq_approx(a,b,thresh=1e-10):
    if np.any(abs(a-b)>thresh ):
        raise RuntimeError("symmetry: check of equality failed for\n {0} \nand\n {1}\n with threshold {2}".format(a,b,thresh))
    
    


class symmetry():

    def __init__(self,lattice,rot_cart=None, rot_cryst=None,ident=False):
# rotation defined by a matrix, which has the transformed basis vectors in rows 
        if ident and not (rot_cart is None):
            raise ValueError(" symmetry is specified as identity, but rot_cart  is specified")
        if ident and not (rot_cryst is None):
            raise ValueError(" symmetry is specified as identity, but rot_cryst is specified")
        if ident: 
            rot_cart=np.eye(3)

        if (rot_cart is None) == (rot_cryst is None):
            raise RuntimeError("symmetry should be defined by rotation of cartesian basis or crysyal basis. One of them. Never both")
        if not (rot_cart is None):
            self._rot_cart=rot_cart
            self._rot_cryst=lattice.dot(self._rot_cart).dot(np.linalg.inv(lattice))
        if not (rot_cryst is None):
            self._rot_cryst=rot_cryst
            self._rot_cart=np.linalg.inv(lattice).dot(self._rot_cryst).dot(lattice)
        _check_eq_approx(abs(np.linalg.det(self._rot_cart  )),1)                  
        _check_eq_approx(abs(np.linalg.det(self._rot_cryst )),1)
        _check_eq_approx(np.linalg.norm(self._rot_cart,axis=1),1)
        _check_eq_approx(np.sum(self._rot_cart*np.roll(self._rot_cart,1,axis=0),axis=1),0)
        
        self._rot_cart_pseudo=


    def show(self):
        print ("rotation in cartesian axes : \n {0} \n".format(self._rot_cart ))
        print ("rotation in crystal   axes : \n {0} \n".format(self._rot_cryst))


lattice=np.array([[1,1,0],[1,0,1],[0,1,1]])
sym=np.array([[0,1,0],[1,0,0],[0,0,1]])
symmetry(lattice,sym).show()