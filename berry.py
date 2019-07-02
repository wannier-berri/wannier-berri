import wan_ham as wham
import numpy as np
from scipy import constants as constants

def  calcAHC(NK,Data,Efermi=None,occ=None, evalJ0=True,evalJ1=True,evalJ2=True,printJ=False,dk=None):
    if dk is not None:
        expdk=np.exp(2j*np.pi*Data.iRvec.dot(dk))
        AA_R_dk=Data.AA_R[:,:,:,:]*expdk[None,None,:,None]
        HH_R_dk=Data.HH_R[:,:,:]*expdk[None,None,:]
    else:
        AA_R_dk=Data.AA_R
        HH_R_dk=Data.HH_R


    if evalJ0 or evalJ1 or evalJ2:
        E_K, delE_K, UU_K, HH_K, delHH_K =wham.get_eig_deleig(NK,HH_R_dk,Data.iRvec,Data.cRvec)
    
    if evalJ1 or evalJ2:
        JJp_list,JJm_list=wham.get_JJp_JJm_list(delHH_K, UU_K, E_K, efermi=Efermi,occ_K=occ)
    
    if evalJ0:
        f_list,g_list=wham.get_occ_mat_list(UU_K, efermi=Efermi, eig_K=E_K, occ_K=occ)

    alpha=np.array([1,2,0])
    beta =np.array([2,0,1])
    
    
    if evalJ1:
        AA_K=wham.fourier_R_to_k( AA_R_dk,Data.iRvec, NK )
    
    if evalJ0:
        OOmega=-1j* wham.fourier_R_to_k( 
             AA_R_dk[:,:,:,alpha]*Data.cRvec[None,None,:,beta ] - 
             AA_R_dk[:,:,:,beta ]*Data.cRvec[None,None,:,alpha]   , Data.iRvec, NK )

    AHC0=np.zeros(3)
    AHC1=np.zeros(3)
    AHC2=np.zeros(3)
#    print (f_list.shape, OOmega.shape)
    fac = -1.0e8*constants.elementary_charge**2/(constants.hbar*Data.cell_volume)/np.prod(NK)


    if evalJ0:
        AHC0= fac*np.einsum("knm,kmna->a",f_list,OOmega).real
        if printJ: print ("J0 term:",AHC0)
    if evalJ1:
        AHC1=-2*fac*( np.einsum("knma,kmna ->a" , AA_K[:,:,:,alpha],JJp_list[:,:,:,beta]).imag +
               np.einsum("knma,kmna ->a" , AA_K[:,:,:,beta],JJm_list[:,:,:,alpha]).imag )
        if printJ: print ("J1 term:",AHC1)
    if evalJ2:
        AHC2=-2*fac*np.einsum("knma,kmna ->a" , JJm_list[:,:,:,alpha],JJp_list[:,:,:,beta]).imag 
        if printJ: print ("J2 term:",AHC2)
    AHC=(AHC0+AHC1+AHC2)
    
    if printJ: print ("Anomalous Hall conductivity: (in S/cm ) \n",AHC)
    return AHC
    
    
    
Fortran="""
  subroutine berry_get_imfgh_klist(kpt, imf_k_list, img_k_list, imh_k_list, occ)
    !=========================================================!
    !
    !! Calculates the three quantities needed for the orbital
    !! magnetization:
    !!
    !! * -2Im[f(k)] [Eq.33 CTVR06, Eq.6 LVTS12]
    !! * -2Im[g(k)] [Eq.34 CTVR06, Eq.7 LVTS12]
    !! * -2Im[h(k)] [Eq.35 CTVR06, Eq.8 LVTS12]
    !! They are calculated together (to reduce the number of
    !! Fourier calls) for a list of Fermi energies, and stored
    !! in axial-vector form.
    !
    ! The two optional output parameters 'imh_k_list' and
    ! 'img_k_list' are only calculated if both of them are
    ! present.
    !
    !=========================================================!

    use w90_constants, only: dp, cmplx_0, cmplx_i
    use w90_utility, only: utility_re_tr_prod, utility_im_tr_prod
    use w90_parameters, only: num_wann, nfermi
    use w90_postw90_common, only: pw90common_fourier_R_to_k_vec, pw90common_fourier_R_to_k
    use w90_wan_ham, only: wham_get_eig_UU_HH_JJlist, wham_get_occ_mat_list
    use w90_get_oper, only: AA_R, BB_R, CC_R
    use w90_utility, only: utility_zgemm_new

    ! Arguments
    !
    real(kind=dp), intent(in)     :: kpt(3)
    real(kind=dp), intent(out), dimension(:, :, :), optional &
      :: imf_k_list, img_k_list, imh_k_list
    real(kind=dp), intent(in), optional, dimension(:) :: occ

    complex(kind=dp), allocatable :: HH(:, :)
    complex(kind=dp), allocatable :: UU(:, :)
    complex(kind=dp), allocatable :: f_list(:, :, :)
    complex(kind=dp), allocatable :: g_list(:, :, :)
    complex(kind=dp), allocatable :: AA(:, :, :)
    complex(kind=dp), allocatable :: BB(:, :, :)
    complex(kind=dp), allocatable :: CC(:, :, :, :)
    complex(kind=dp), allocatable :: OOmega(:, :, :)
    complex(kind=dp), allocatable :: JJp_list(:, :, :, :)
    complex(kind=dp), allocatable :: JJm_list(:, :, :, :)
    real(kind=dp)                 :: eig(num_wann)
    integer                       :: i, j, ife, nfermi_loc
    real(kind=dp)                 :: s

    ! Temporary space for matrix products
    complex(kind=dp), allocatable, dimension(:, :, :) :: tmp

    if (present(occ)) then
      nfermi_loc = 1
    else
      nfermi_loc = nfermi
    endif

    allocate (HH(num_wann, num_wann))
    allocate (UU(num_wann, num_wann))
    allocate (f_list(num_wann, num_wann, nfermi_loc))
    allocate (g_list(num_wann, num_wann, nfermi_loc))
    allocate (JJp_list(num_wann, num_wann, nfermi_loc, 3))
    allocate (JJm_list(num_wann, num_wann, nfermi_loc, 3))
    allocate (AA(num_wann, num_wann, 3))
    allocate (OOmega(num_wann, num_wann, 3))

    ! Gather W-gauge matrix objects
    !

    if (present(occ)) then
      call wham_get_eig_UU_HH_JJlist(kpt, eig, UU, HH, JJp_list, JJm_list, occ=occ)
      call wham_get_occ_mat_list(UU, f_list, g_list, occ=occ)
    else
      call wham_get_eig_UU_HH_JJlist(kpt, eig, UU, HH, JJp_list, JJm_list)
      call wham_get_occ_mat_list(UU, f_list, g_list, eig=eig)
    endif

    call pw90common_fourier_R_to_k_vec(kpt, AA_R, OO_true=AA, OO_pseudo=OOmega)

    if (present(imf_k_list)) then
      ! Trace formula for -2Im[f], Eq.(51) LVTS12
      !
      do ife = 1, nfermi_loc
        do i = 1, 3
          !
          ! J0 term (Omega_bar term of WYSV06)
          imf_k_list(1, i, ife) = &
            utility_re_tr_prod(f_list(:, :, ife), OOmega(:, :, i))
          !
          ! J1 term (DA term of WYSV06)
          imf_k_list(2, i, ife) = -2.0_dp* &
                                  ( &
                                  utility_im_tr_prod(AA(:, :, alpha_A(i)), JJp_list(:, :, ife, beta_A(i))) &
                                  + utility_im_tr_prod(JJm_list(:, :, ife, alpha_A(i)), AA(:, :, beta_A(i))) &
                                  )
          !
          ! J2 term (DD of WYSV06)
          imf_k_list(3, i, ife) = -2.0_dp* &
                                  utility_im_tr_prod(JJm_list(:, :, ife, alpha_A(i)), JJp_list(:, :, ife, beta_A(i)))
        end do
      end do
    end if

    if (present(img_k_list)) img_k_list = 0.0_dp
    if (present(imh_k_list)) imh_k_list = 0.0_dp

    if (present(img_k_list) .and. present(imh_k_list)) then
      allocate (BB(num_wann, num_wann, 3))
      allocate (CC(num_wann, num_wann, 3, 3))

      allocate (tmp(num_wann, num_wann, 5))
      ! tmp(:,:,1:3) ... not dependent on inner loop variables
      ! tmp(:,:,1) ..... HH . AA(:,:,alpha_A(i))
      ! tmp(:,:,2) ..... LLambda_ij [Eq. (37) LVTS12] expressed as a pseudovector
      ! tmp(:,:,3) ..... HH . OOmega(:,:,i)
      ! tmp(:,:,4:5) ... working matrices for matrix products of inner loop

      call pw90common_fourier_R_to_k_vec(kpt, BB_R, OO_true=BB)
      do j = 1, 3
        do i = 1, j
          call pw90common_fourier_R_to_k(kpt, CC_R(:, :, :, i, j), CC(:, :, i, j), 0)
          CC(:, :, j, i) = conjg(transpose(CC(:, :, i, j)))
        end do
      end do

      ! Trace formula for -2Im[g], Eq.(66) LVTS12
      ! Trace formula for -2Im[h], Eq.(56) LVTS12
      !
      do i = 1, 3
        call utility_zgemm_new(HH, AA(:, :, alpha_A(i)), tmp(:, :, 1))
        call utility_zgemm_new(HH, OOmega(:, :, i), tmp(:, :, 3))
        !
        ! LLambda_ij [Eq. (37) LVTS12] expressed as a pseudovector
        tmp(:, :, 2) = cmplx_i*(CC(:, :, alpha_A(i), beta_A(i)) &
                                - conjg(transpose(CC(:, :, alpha_A(i), beta_A(i)))))

        do ife = 1, nfermi_loc
          !
          ! J0 terms for -2Im[g] and -2Im[h]
          !
          ! tmp(:,:,5) = HH . AA(:,:,alpha_A(i)) . f_list(:,:,ife) . AA(:,:,beta_A(i))
          call utility_zgemm_new(tmp(:, :, 1), f_list(:, :, ife), tmp(:, :, 4))
          call utility_zgemm_new(tmp(:, :, 4), AA(:, :, beta_A(i)), tmp(:, :, 5))

          s = 2.0_dp*utility_im_tr_prod(f_list(:, :, ife), tmp(:, :, 5)); 
          img_k_list(1, i, ife) = utility_re_tr_prod(f_list(:, :, ife), tmp(:, :, 2)) - s
          imh_k_list(1, i, ife) = utility_re_tr_prod(f_list(:, :, ife), tmp(:, :, 3)) + s

          !
          ! J1 terms for -2Im[g] and -2Im[h]
          !
          ! tmp(:,:,1) = HH . AA(:,:,alpha_A(i))
          ! tmp(:,:,4) = HH . JJm_list(:,:,ife,alpha_A(i))
          call utility_zgemm_new(HH, JJm_list(:, :, ife, alpha_A(i)), tmp(:, :, 4))

          img_k_list(2, i, ife) = -2.0_dp* &
                                  ( &
                                  utility_im_tr_prod(JJm_list(:, :, ife, alpha_A(i)), BB(:, :, beta_A(i))) &
                                  - utility_im_tr_prod(JJm_list(:, :, ife, beta_A(i)), BB(:, :, alpha_A(i))) &
                                  )
          imh_k_list(2, i, ife) = -2.0_dp* &
                                  ( &
                                  utility_im_tr_prod(tmp(:, :, 1), JJp_list(:, :, ife, beta_A(i))) &
                                  + utility_im_tr_prod(tmp(:, :, 4), AA(:, :, beta_A(i))) &
                                  )

          !
          ! J2 terms for -2Im[g] and -2Im[h]
          !
          ! tmp(:,:,4) = JJm_list(:,:,ife,alpha_A(i)) . HH
          ! tmp(:,:,5) = HH . JJm_list(:,:,ife,alpha_A(i))
          call utility_zgemm_new(JJm_list(:, :, ife, alpha_A(i)), HH, tmp(:, :, 4))
          call utility_zgemm_new(HH, JJm_list(:, :, ife, alpha_A(i)), tmp(:, :, 5))

          img_k_list(3, i, ife) = -2.0_dp* &
                                  utility_im_tr_prod(tmp(:, :, 4), JJp_list(:, :, ife, beta_A(i)))
          imh_k_list(3, i, ife) = -2.0_dp* &
                                  utility_im_tr_prod(tmp(:, :, 5), JJp_list(:, :, ife, beta_A(i)))
        end do
      end do
      deallocate (tmp)
    end if

  end subroutine berry_get_imfgh_klist
"""