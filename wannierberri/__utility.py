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



__debug = False

import inspect
import numpy as np
from lazy_property import LazyProperty as Lazy



alpha_A=np.array([1,2,0])
beta_A =np.array([2,0,1])
TAU_UNIT=1E-9 # tau in nanoseconds
TAU_UNIT_TXT="ns"



def print_my_name_start():
    if __debug: 
        print("DEBUG: Running {} ..".format(inspect.stack()[1][3]))


def print_my_name_end():
    if __debug: 
        print("DEBUG: Running {} - done ".format(inspect.stack()[1][3]))



def einsumk(*args):
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
        self.NE1=int(maxdE*self.T/dE)
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
            res[i]=A[start:end].transpose(tuple(range(1,len(A.shape)))+(0,)).dot(self.smt[start1:end1])/self.smt[start1:end1].sum()
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


def fourier_R_to_k(AAA_R,iRvec,NKPT,hermitian=False,antihermitian=False):
    print_my_name_start()
    #  AAA_R is an array of dimension ( num_wann x num_wann x nRpts X ... ) (any further dimensions allowed)
    if  hermitian and antihermitian :
        raise ValueError("A matrix cannot be bothe Haermitian and antihermitian, unless it is zero")
    if hermitian:
        return fourier_R_to_k_hermitian(AAA_R,iRvec,NKPT)
    if antihermitian:
        return fourier_R_to_k_hermitian(AAA_R,iRvec,NKPT,anti=True)

    #now the generic case
    NK=tuple(NKPT)
    nRvec=iRvec.shape[0]
    shapeA=AAA_R.shape
    assert(nRvec==shapeA[2])
    AAA_R=AAA_R.transpose( (2,0,1)+tuple(range(3,len(shapeA)))  )    
    assert(nRvec==AAA_R.shape[0])
    AAA_R=AAA_R.reshape(nRvec,-1)
    AAA_K=np.zeros( NK+(AAA_R.shape[1],), dtype=complex )

    for ir,irvec in enumerate(iRvec):
#            print ("ir {0} of {1}".format(ir,len(iRvec)))
            AAA_K[tuple(irvec)]=AAA_R[ir]
    for m in range(AAA_K.shape[3]):
#            print ("Fourier {0} of {1}".format(m,AAA_K.shape[3]))
            AAA_K[:,:,:,m]=np.fft.fftn(AAA_K[:,:,:,m])
    AAA_K=AAA_K.reshape( (np.prod(NK),)+shapeA[0:2]+shapeA[3:])
#    print ("finished fourier")
    print_my_name_end()
    return AAA_K


def fourier_R_to_k_hermitian(AAA_R,iRvec,NKPT,anti=False):
###  in practice (at least for the test example)  use of hermiticity does not speed the calculation. 
### probably, because FFT is faster then reshaping matrices
#    return fourier_R_to_k(AAA_R,iRvec,NKPT)
    #  AAA_R is an array of dimension ( num_wann x num_wann x nRpts X ... ) (any further dimensions allowed)
    #  AAA_k is assumed Hermitian (in n,m) , so only half of it is calculated
    print_my_name_start()
    NK=tuple(NKPT)
    nRvec=iRvec.shape[0]
    shapeA=AAA_R.shape
    num_wann=shapeA[0]
    assert(nRvec==shapeA[2])
    M,N=np.triu_indices(num_wann)
    ntriu=len(M)
    AAA_R=AAA_R[M,N].transpose( (1,0)+tuple(range(2,len(shapeA)-1))  ).reshape(nRvec,-1)
    AAA_K=np.zeros( NK+(AAA_R.shape[1],), dtype=complex )
    for ir,irvec in enumerate(iRvec):
#            print ("ir {0} of {1}".format(ir,len(iRvec)))
            AAA_K[tuple(irvec)]=AAA_R[ir]
    for m in range(AAA_K.shape[3]):
#            print ("Fourier {0} of {1}".format(m,AAA_K.shape[3]))
            AAA_K[:,:,:,m]=np.fft.fftn(AAA_K[:,:,:,m])
    AAA_K=AAA_K.reshape( (np.prod(NK),ntriu)+shapeA[3:])
    result=np.zeros( (np.prod(NK),num_wann,num_wann)+shapeA[3:],dtype=complex)
    result[:,M,N]=AAA_K
    diag=np.arange(num_wann)
    if anti:
        result[:,N,M]=-AAA_K.conjugate()
        result[:,diag,diag]=result[:,diag,diag].imag
    else:
        result[:,N,M]=AAA_K.conjugate()
        result[:,diag,diag]=result[:,diag,diag].real
#    print ("finished fourier")
    print_my_name_end()
    return result




