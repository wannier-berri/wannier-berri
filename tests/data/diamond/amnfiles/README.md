these files were obtained with a patched version of pw2wannier90.x

in the subroutine radialpart a line should be added:

  mesh_r = mesh_r * rvalue

otherwise, the resulta do not match severelly for radial function with nodes.

See https://github.com/wannier-developers/wannier90/issues/590

also, some parameters increased for better accuracy:

```fortran 

SUBROUTINE radialpart(ng, q, alfa, rvalue, lmax, radial)
  !-----------------------------------------------------------------------
  !
  ! This routine computes a table with the radial Fourier transform
  ! of the radial functions.
  !
  USE kinds,      ONLY : dp
  USE constants,  ONLY : fpi
  USE cell_base,  ONLY : omega
  !
  IMPLICIT NONE
  ! I/O
  INTEGER :: ng, rvalue, lmax
  real(DP) :: q(ng), alfa, radial(ng,0:lmax)
  ! local variables
  real(DP), PARAMETER :: xmin=-6.d0, dx=0.01d0, rmax=20.d0

  real(DP) :: rad_int, pref, x
  INTEGER :: l, lp1, ir, ig, mesh_r, ierr
  real(DP), ALLOCATABLE :: bes(:), func_r(:), r(:), rij(:), aux(:)

  
  mesh_r = nint ( ( log ( rmax/ min(alfa, 1d0) ) - xmin ) / dx + 1 )
  mesh_r = mesh_r * rvalue
  ALLOCATE ( bes(mesh_r), func_r(mesh_r), r(mesh_r), rij(mesh_r), stat=ierr)
  IF (ierr /= 0) CALL errore('pw2wannier90', 'Error allocating bes/func_r/r/rij', 1)
  ALLOCATE ( aux(mesh_r), stat=ierr)
  IF (ierr /= 0) CALL errore('pw2wannier90', 'Error allocating aux', 1)
  !
  !    compute the radial mesh
  !
  DO ir = 1, mesh_r
     x = xmin  + dble (ir - 1) * dx
     r (ir) = exp (x) / alfa
     rij (ir) = dx  * r (ir)
  ENDDO
  !
  IF (rvalue==1) func_r(:) = 2.d0 * alfa**(3.d0/2.d0) * exp(-alfa*r(:))
  IF (rvalue==2) func_r(:) = 1.d0/sqrt(8.d0) * alfa**(3.d0/2.d0) * &
                     (2.0d0 - alfa*r(:)) * exp(-alfa*r(:)*0.5d0)
  IF (rvalue==3) func_r(:) = sqrt(4.d0/27.d0) * alfa**(3.0d0/2.0d0) * &
                     (1.d0 - 2.0d0/3.0d0*alfa*r(:) + 2.d0*(alfa*r(:))**2/27.d0) * &
                                           exp(-alfa*r(:)/3.0d0)
  pref = fpi/sqrt(omega)
  !
  DO l = 0, lmax
     DO ig=1,ng
       CALL sph_bes (mesh_r, r(1), q(ig), l, bes)
       aux(:) = bes(:) * func_r(:) * r(:) * r(:)
       ! second r factor added upo suggestion by YY Liang
       CALL simpson (mesh_r, aux, rij, rad_int)
       radial(ig,l) = rad_int * pref
     ENDDO
  ENDDO

  DEALLOCATE (bes, func_r, r, rij, aux )
  RETURN
END SUBROUTINE radialpart  
```
