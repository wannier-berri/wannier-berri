__debug = False

import inspect
import numpy as np

def print_my_name_start():
    if __debug: 
        print("DEBUG: Running {} ..".format(inspect.stack()[1][3]))


def print_my_name_end():
    if __debug: 
        print("DEBUG: Running {} - done ".format(inspect.stack()[1][3]))



def einsumk(*args):
#   self._HHUU_K=np.einsum("kmi,kmn,knj->kij",self.UUC_K,_HH_K,self.UU_K).real
    left,right=args[0].split("->")
    left=left.split(",")
    for s in left + [right]:
        if s[0]!='k':
            raise RuntimeError("the first index should be 'k', found '{1}".format(s[0]))
    string_new=",".join(s[1:]for s in left)+"->"+right[1:]
    print ("string_new"+"  ".join(str(a.shape) for a in args[1:]))
    nmat=len(args)-1
    assert(len(left)==nmat)
    if nmat==2:
        return np.array([np.einsum(string_new,a,b) for a,b in zip(args[1],args[2])])
    elif nmat==3:
#        return np.array([np.einsum(string_new,a,b,c) for a,b,c in zip(args[1],args[2],args[3])])
        return np.array([a.dot(b).dot(c) for a,b,c in zip(args[1],args[2],args[3])])
    elif nmat==4:
        return np.array([np.einsum(string_new,a,b,c,d) for a,b,c,d in zip(args[1],args[2],args[3],args[4])])
    else:
        raise RuntimeError("einsumk is not implemented for number of matrices {}".format(nmat))
    


from scipy.constants import Boltzmann,elementary_charge,hbar

class smoother():
    def __init__(self,E,T=10):  # T in K
        self.T=T*Boltzmann/elementary_charge  # now in eV
        self.E=np.copy(E)
        dE=E[1]-E[0]
        maxdE=5
        self.NE1=int(maxdE*T/dE)
        self.NE=E.shape[0]
        self.smt=self._broaden(np.arange(-self.NE1,self.NE1+1)*dE)*dE


    def _broaden(self,E):
        return 0.25/self.T/np.cosh(E/(2*self.T))**2

    def __call__(self,A):
        assert self.E.shape[0]==A.shape[0]
        res=np.zeros(A.shape)
        for i in range(self.NE):
            start=max(0,i-self.NE1)
            end=min(self.NE,i+self.NE1+1)
            start1=self.NE1-(i-start)
            end1=self.NE1+(end-i)
            res[i]=A[start:end].transpose(tuple(range(1,len(A.shape)))+(0,)).dot(self.smt[start1:end1])+A[0]*self.smt[:start1].sum()+A[-1]*self.smt[end1:].sum()
        return res


class voidsmoother(smoother):
    def __init__(self):
        pass
        
    def __call__(self,A):
        return A

