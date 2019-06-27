#-*- mode: python -*-#
#------------------------------------------------------------#
# This file WAS distributed as part of the Wannier90 code and #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier90      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The webpage of the Wannier90 code is www.wannier.org       #
#                                                            #
# The Wannier90 code is hosted on GitHub:                    #
#                                                            #
# https://github.com/wannier-developers/wannier90            #
#------------------------------------------------------------#
#                                                               #
#  Translated to python and adapted for wannier19 project by Stepan Tsirkin #
#                                                               #
#------------------------------------------------------------#

import numpy as np

    #================================#
    ## Occupation matrix f, and g=1-f
    ## for a list of Fermi energies
    #================================#


def fourier_R_to_k(AAA_R,iRvec,NKPT):
    #  AAA_R is an array of dimension ( num_wann x num_wann x nRpts X ... ) (any further dimensions allowed)
    NK=tuple(NKPT)
    nRvec=iRvec.shape[0]
    shapeA=AAA_R.shape
    assert(nRvec==shapeA[2])
    print (AAA_R.shape)
    AAA_R=AAA_R.transpose( (2,0,1)+tuple(range(3,len(shapeA)))  )    
    print (AAA_R.shape)
    AAA_R=AAA_R.reshape(nRvec,-1)
    print (AAA_R.shape)
    AAA_K=np.zeros( NK+(AAA_R.shape[1],), dtype=complex )
    print (AAA_K.shape)

    for ir,irvec in enumerate(iRvec):
#        if np.all(np.abs(iRvec[iR])<=np.array(NK)/2):
            AAA_K[irvec]=AAA_R[ir]
    print (shapeA,AAA_K.shape)
    for m in range(AAA_K.shape[3]):
            AAA_K[:,:,:,m]=np.fft.fftn(AAA_K[:,:,:,m])
    print(AAA_K.shape)
    AAA_K=AAA_K.reshape( (np.prod(NK),)+shapeA[0:2]+shapeA[3:])
    print(AAA_K.shape)
    return AAA_K


def get_occ_mat_list_fermi(UU,num_wann,fermi_energy_list, eig ):
    occ_list = np.array([get_occ(eig, ef )  for ef in efermi])
    f_list=np.einsum("ni,fi,mi->nmf",UU,occ_list,UU.cong())
    g_list=-f_list.copy
    for n in range(UU.shape[0]):
        g_list[n,n,:]+=1    
    return f_list,g_list


def get_occ_mat_list_occ(UU, occ):
    f_list=np.einsum("ni,i,mi->nm",UU,occ_list,UU.cong())
    g_list=np.eye(f_list.shape)-f_list    
    return f_list,g_list


def  get_eig_deleig(NK,HH_R,iRvec,cRvec=None):
    ## For all  k point on a NK grid this function returns eigenvalues E and
    ## derivatives of the eigenvalues dE/dk_a, using wham_get_deleig_a
    
    num_wann=HH_R.shape[0]
    HH_K=fourier_R_to_k(HH_R,iRvec,NK)
 
    check=np.max( [np.abs(H-H.T.conj()).max() for H in HH_K] )
    if check>1e-10 : raise RuntimeError ("Hermiticity of interpolated Hamiltonian is not good : {0}".format(check))


    EUU=[np.linalg.eigh(Hk) for Hk in HH_K]
    E_K=np.array([euu[0] for euu in EUU])
    UU_K =np.array([euu[1] for euu in EUU])
    print ("Energies calculated")
    
    if cRvec is None: return E_K,  UU_K, HH_K 
    
    delHH_R=HH_R[:,:,:,None]*cRvec[None,None,:,:]
    delHH_K=fourier_R_to_k(delHH_R,iRvec,NK)
    
    delE_K=np.einsum("kml,kmna,knl->kla",UU.conj(),delHH,UU)
    
    return E_K, delE_K, UU_K, HH_K, delHH_K 

older="""

  subroutine wham_get_eig_UU_HH_JJlist(kpt, eig, UU, HH, JJp_list, JJm_list, occ)
    #========================================================#
    #                                                        #
    ## Wrapper routine used to reduce number of Fourier calls
    #    Added the optional occ parameter                    #
    #========================================================#

    use w90_parameters, only: num_wann
    use w90_get_oper, only: HH_R, get_HH_R
    use w90_postw90_common, only: pw90common_fourier_R_to_k_new
    use w90_utility, only: utility_diagonalize

    real(kind=dp), dimension(3), intent(in)           :: kpt
    real(kind=dp), intent(out)                        :: eig(num_wann)
    complex(kind=dp), dimension(:, :), intent(out)     :: UU
    complex(kind=dp), dimension(:, :), intent(out)     :: HH
    complex(kind=dp), dimension(:, :, :, :), intent(out) :: JJp_list
    complex(kind=dp), dimension(:, :, :, :), intent(out) :: JJm_list
    real(kind=dp), intent(in), optional, dimension(:) :: occ

    integer                       :: i
    complex(kind=dp), allocatable :: delHH(:, :, :)

    call get_HH_R

    allocate (delHH(num_wann, num_wann, 3))
    call pw90common_fourier_R_to_k_new(kpt, HH_R, OO=HH, &
                                       OO_dx=delHH(:, :, 1), &
                                       OO_dy=delHH(:, :, 2), &
                                       OO_dz=delHH(:, :, 3))
    call utility_diagonalize(HH, num_wann, eig, UU)
    do i = 1, 3
      if (present(occ)) then
        call wham_get_JJp_JJm_list(delHH(:, :, i), UU, eig, JJp_list(:, :, :, i), JJm_list(:, :, :, i), occ=occ)
      else
        call wham_get_JJp_JJm_list(delHH(:, :, i), UU, eig, JJp_list(:, :, :, i), JJm_list(:, :, :, i))
      endif
    enddo

  end subroutine wham_get_eig_UU_HH_JJlist



end module w90_wan_ham
"""