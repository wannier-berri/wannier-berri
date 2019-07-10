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
            AAA_K[tuple(irvec)]=AAA_R[ir]
    for m in range(AAA_K.shape[3]):
            AAA_K[:,:,:,m]=np.fft.fftn(AAA_K[:,:,:,m])
    AAA_K=AAA_K.reshape( (np.prod(NK),)+shapeA[0:2]+shapeA[3:])
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
            AAA_K[tuple(irvec)]=AAA_R[ir]
    for m in range(AAA_K.shape[3]):
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
    return result


def get_eig_HH_UU(NK,HH_R,iRvec):
    res=get_eig_deleig(NK,HH_R,iRvec)
    return res[0],res[2],res[3]  # E_K,UU_K,HH_


def get_eig(NK,HH_R,iRvec):
    num_wann=HH_R.shape[0]
    HH_K=fourier_R_to_k(HH_R,iRvec,NK,hermitian=True)
    return np.array([np.linalg.eigvalsh(Hk) for Hk in HH_K])


def get_eig_deleig(NK,HH_R,iRvec,cRvec=None,calcdE=False):
    ## For all  k point on a NK grid this function returns eigenvalues E and
    ## derivatives of the eigenvalues dE/dk_a, using wham_get_deleig_a
    num_wann=HH_R.shape[0]
    HH_K=fourier_R_to_k(HH_R,iRvec,NK,hermitian=True)
    EUU=[np.linalg.eigh(Hk) for Hk in HH_K]
    E_K=np.array([euu[0] for euu in EUU])
    UU_K =np.array([euu[1] for euu in EUU])
    
    if cRvec is None: return E_K, None, UU_K, HH_K, None 
    
    delHH_R=1j*HH_R[:,:,:,None]*cRvec[None,None,:,:]
    delHH_K=fourier_R_to_k(delHH_R,iRvec,NK,hermitian=True)
    
    if calcdE:
        delE_K=np.einsum("kml,kmna,knl->kla",UU_K.conj(),delHH_K,UU_K)    
        check=np.abs(delE_K).imag.max()
        if check>1e-10: raiseruntimeError ("The band derivatives have considerable imaginary part: {0}".format(check))
        delE_K=delE_K.real
    else:
        delE_K=None
    
    return E_K, delE_K, UU_K, HH_K, delHH_K 




not_used="""

    check=np.max( [np.abs(H-H.T.conj()).max() for H in HH_K] )
    if check>1e-10 : raise RuntimeError ("Hermiticity of interpolated Hamiltonian is not good : {0}".format(check))


def get_eig_slow(NK,HH_R,iRvec):
    num_wann=HH_R.shape[0]
    dk=1./NK
    kpt_list=[dk*np.array([x,y,z]) for x in range(NK[0]) for y in range(NK[1]) for z in range(NK[2]) ]
    HH_K=[HH_R.dot(np.exp(2j*np.pi*iRvec.dot(kpt))) for kpt in kpt_list]
#    check=np.max( [np.abs(H-H.T.conj()).max() for H in HH_K] )
#    if check>1e-10 : raise RuntimeError ("Hermiticity of interpolated Hamiltonian is not good : {0}".format(check))
    return np.array([np.linalg.eigvalsh(Hk) for Hk in HH_K])


def get_occ(eig_K,efermi):
    occ=np.zeros(eig_K.shape,dtype=float)
    occ[eig_K<efermi]=1
    return occ

def get_occ_mat_list(UUU_k, occ_K=None,efermi=None,eig_K=None):
    if occ_K is None:
        occ_K = get_occ(eig_K, efermi ) 
    f_list=np.array([uuu.dot(occ) for uuu,occ in zip(UUU_k,occ_K)] ) #,UU_k.conj())
    g_list=-f_list.copy()
    for ik in range(g_list.shape[0]):
        g_list[ik]+=np.eye(g_list.shape[1])
    return f_list,g_list


def get_J2_term(delHH_dE_K, eig_K, efermi):
    selm=np.sum(eig_K< efermi,axis=1)
    seln=np.sum(eig_K<=efermi,axis=1)
    return np.array([sum( (delhh[n:,:m,b]*delhh[:m,n:,a].T).imag.sum() 
           for delhh,n,m in zip(delHH_dE_K,seln,selm) ) for a,b in zip(alpha,beta)] )


def get_J1_term(delHH_dE_K, eig_K, AAUU_K,efermi):
    selm=np.sum(eig_K< efermi,axis=1)
    seln=np.sum(eig_K<=efermi,axis=1)
    return np.array([sum( (delhh[n:,:m,b]*aa[:m,n:,a].T).imag.sum() +
                (delhh[:m,n:,a]*aa[n:,:m,b].T).imag.sum()
           for delhh,aa,n,m in zip(delHH_dE_K,AAUU_K,seln,selm)  ) for a,b in zip(alpha,beta)])

def get_JJp_JJm_list(delHH_dE_K, UU_K, UUC_K, eig_K, occ_K=None,efermi=None):
    #===============================================#
    #                                               #
    # Compute JJ^+_a and JJ^-_a (a=Cartesian index) #
    # for a list of Fermi energies                  #
    #                                               #
    #===============================================#

    
    if occ_K is None and efermi is None:
        raise RuntimeError("either occ or efermi should be specified")
    if not( (occ_K is None) or (efermi is None)):
        raise RuntimeError("either occ or efermi should be specified, NOT BOTH!")
    
    if efermi is None:
        selm=(occ_K<0.5)
        seln=(occ_K>0.5)
        JJp_list=np.array( [np.einsum("lm,mna,pn->lpa",
            UU_K[ik][:,seln[ik]],delHH_dE_K[ik][seln[ik],:,:][:,selm[ik],:]  , UU_K[ik][:,selm[ik]].conj() )
                for ik in range(UU_K.shape[0]) ] )

        JJm_list=np.array( [np.einsum("lm,mna,pn->lpa",
            UU_K[ik][:,selm[ik]],delHH_dE_K[ik][selm[ik],:,:][:,seln[ik],:]  , UU_K[ik][:,seln[ik]].conj() )
                for ik in range(UU_K.shape[0]) ] )

        return JJp_list, JJm_list


    else:
        selm=np.sum(eig_K< efermi,axis=1)
        seln=np.sum(eig_K<=efermi,axis=1)

        return __get_JJp_JJm_list(UU_K,UUC_K,delHH_dE_K,seln,selm)


def __get_JJp_JJm_list(UU_K,UUC_K,delHH_dE_K,seln,selm):


#        JJp_list=np.array( [np.einsum("lm,mna,pn->lpa",
#            uu[:,n:],delhh[n:,:m,:]  , uu[:,:m].conj() )
#                for uu,delhh,n,m in zip(UU_K,delHH_dE_K,seln,selm) ] )


        JJp_list=np.array( [
            uu[:,n:].dot(  uuc[:,:m].dot( delhh[n:,:m,:])) 
                for uu,uuc,delhh,n,m in zip(UU_K,UUC_K,delHH_dE_K,seln,selm) ] )

        JJm_list=np.array( [
            uu[:,:m].dot( uuc[:,n:].dot(delhh[:m,n:,:])  )
                for uu,uuc,delhh,n,m in zip(UU_K,UUC_K,delHH_dE_K,seln,selm) ] )


        return JJp_list, JJm_list




def __get_JJp_JJm_list(UU_K,UUC_K,delHH_dE_K,seln,selm):


#        JJp_list=np.array( [np.einsum("lm,mna,pn->lpa",
#            uu[:,n:],delhh[n:,:m,:]  , uu[:,:m].conj() )
#                for uu,delhh,n,m in zip(UU_K,delHH_dE_K,seln,selm) ] )


        JJp_list=np.array( [
            uu[:,n:].dot(  uuc[:,:m].dot( delhh[n:,:m,:])) 
                for uu,uuc,delhh,n,m in zip(UU_K,UUC_K,delHH_dE_K,seln,selm) ] )

        JJm_list=np.array( [
            uu[:,:m].dot( uuc[:,n:].dot(delhh[:m,n:,:])  )
                for uu,uuc,delhh,n,m in zip(UU_K,UUC_K,delHH_dE_K,seln,selm) ] )


        return JJp_list, JJm_list




"""