#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                     written by                             #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#


__debug = False

import inspect
import numpy as np
from lazy_property import LazyProperty as Lazy

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


    @Lazy
    def __str__(self):
        return ("<Smoother T={}, NE={}, NE1={} , E={}..{} step {}>".format(self.T,self.NE,self.NE1,self.Emin,self.Emax,self.dE) )
        
    @Lazy 
    def dE(self):
        return self.E[1]-self.E[0]

    @Lazy 
    def Emin(self):
        return self.E[0]

    @Lazy 
    def Emax(self):
        return self.E[-1]

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


    def __eq__(self,other):
        if isinstance(other,voidsmoother):
            return False
        elif not isinstance(other,smoother):
            return False
        else:
            for var in ['T','dE','NE','NE1','Emin','Emax']:
                if getattr(self,var)!=getattr(other,var):
                    return False
        return True
#            return self.T==other.T and self.dE=other.E and self.NE==other.NE and self.



class voidsmoother(smoother):
    def __init__(self):
        pass
    
    def __eq__(self,other):
        if isinstance(other,voidsmoother):
            return True
        else:
            return False
    
    def __call__(self,A):
        return A

    def __str__(self):
        return ("<Smoother - void " )



def str2bool(v):
    if v[0] in "fF" :
        return False
    elif v[0] in "tT":
        return True
    else :
        raise RuntimeError(" unrecognized value of bool parameter :{0}".format(v) )
