#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# part of this file is an adapted translation of             #
# Fortran90 code from  Wannier 90 project                    #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
# The webpage of the Wannier90 code is www.wannier.org       #
# The Wannier90 code is hosted on GitHub:                    #
# https://github.com/wannier-developers/wannier90            #
#------------------------------------------------------------#
#                                                            #
#  Translated to python and adapted for wannier19 project by #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#


import numpy as np


alpha=np.array([1,2,0])
beta =np.array([2,0,1])


def fourier_R_to_k(AAA_R,iRvec,NKPT,hermitian=False,antihermitian=False):
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
    return AAA_K


def fourier_R_to_k_hermitian(AAA_R,iRvec,NKPT,anti=False):
###  in practice (at least for the test example)  use of hermiticity does not speed the calculation. 
### probably, because FFT is faster then reshaping matrices
#    return fourier_R_to_k(AAA_R,iRvec,NKPT)
    #  AAA_R is an array of dimension ( num_wann x num_wann x nRpts X ... ) (any further dimensions allowed)
    #  AAA_k is assumed Hermitian (in n,m) , so only half of it is calculated
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
    return result




