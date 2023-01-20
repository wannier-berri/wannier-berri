from ..__utility import FortranFileR


def readelph(filename):

    iuelph = FortranFileR(filename)
    readint = lambda: iuelph.read_record('i4')
    readfloat = lambda: iuelph.read_record('f8')
    readstr = lambda : (c.decode('ascii') for c in iuelph.read_record('c'))
    readbool = lambda: iuelph.read_record('b')

# many of the variables are not needed for us at all, but we assign them some names to keed track of the structure of the file
# after end of the routine they will be garbage-collected anyway


#        WRITE(iuelph) stitle, sdate, stime
    stitle, sdate, stime = readstr()
#        WRITE(iuelph) ibrav, nat, nsp, nrot, nsym, nsym_ns, nsym_na, &
#             ngm_g, nspin, nbnd, nmodes, nqs
    ibrav, nat, nsp, nrot, nsym, nsym_ns, nsym_na, ngm_g, nspin, nbnd, nmodes, nqs = readint()
#        WRITE(iuelph) nq1, nq2, nq3, nk1, nk2, nk3, k1, k2, k3
    nq1, nq2, nq3, nk1, nk2, nk3, k1, k2, k3 = readint()
#        WRITE(iuelph) time_reversal, invsym, nofrac, allfrac, nosym, &
#             nosym_evc, no_t_rev
    time_reversal, invsym, nofrac, allfrac, nosym, nosym_evc, no_t_rev = readbool()
#        WRITE(iuelph) alat, omega, tpiba, nelec, ecutrho, ecutwfc
    alat, omega, tpiba, nelec, ecutrho, ecutwfc = readfloat()
#        WRITE(iuelph) dfftp%nr1, dfftp%nr2, dfftp%nr3
    dfftp_nr = readint() 
#        WRITE(iuelph) dffts%nr1, dffts%nr2, dffts%nr3
    dffts_nr = readint()
#        WRITE(iuelph) dfftb%nr1, dfftb%nr2, dfftb%nr3
    dfftb_nr = readint()
#        WRITE(iuelph) ((at(ii, jj), ii = 1, 3), jj = 1, 3)
    at = readfloat().reshape((3,3),order='F')
#        WRITE(iuelph) ((bg(ii, jj), ii = 1, 3), jj = 1, 3)
    bg = readfloat().reshape((3,3),order='F')
#        WRITE(iuelph) (atomic_number(atm(ii)), ii = 1, nsp)
    atomic_numbers = readint()
#        WRITE(iuelph) (ityp(ii), ii = 1, nat)
    ityp = readint()
#        WRITE(iuelph) ((tau(ii, jj), ii = 1, 3), jj = 1, nat)
    tau = readfloat().reshape( (nat,3) , order='C')
#        WRITE(iuelph) ((x_q(ii, jj), ii = 1, 3), jj = 1, nqs)
    x_q = readfloat.reshape( (nqs,3), order='C')
#        WRITE(iuelph) (wq(ii), ii = 1, nqs)
    wq = readfloat()
#        WRITE(iuelph) (lgamma_iq(ii), ii = 1, nqs)
    lgamma_ig = readbool()

    for iq in range(nqs):

#     WRITE(iuelph) nsymq, irotmq, nirr, npertx, nkstot, nksqtot
        nsymq, irotmq, nirr, npertx, nkstot, nksqtot = readint()
#     WRITE(iuelph) minus_q, invsymq
        minus_q, invsymq = readint()
#     WRITE(iuelph) (irgq(ii), ii = 1, 48)
        irgq = readint()
#     WRITE(iuelph) (npert(ii), ii = 1, nmodes)
        npert = readint()
#     WRITE(iuelph) (((rtau(ii, jj, kk), ii = 1, 3), jj = 1, 48), &
#          kk = 1, nat)
        rtau = radfloat()
#     WRITE(iuelph) ((gi(ii, jj), ii = 1, 3), jj = 1, 48)
        gi = readfloat()
#     WRITE(iuelph) (gimq(ii), ii = 1, 3)
        readfloat() 
#     WRITE(iuelph) ((u(ii, jj), ii = 1, nmodes), jj = 1, nmodes)
        readfloat()
#     WRITE(iuelph) ((((t(ii, jj, kk, ll), ii = 1, npertx), &
#          jj = 1, npertx), kk = 1, 48), ll = 1, nmodes)
        readfloat()
#     WRITE(iuelph) (((tmq(ii, jj, kk), ii = 1, npertx), &
#          jj = 1, npertx), kk = 1, nmodes)
#     WRITE(iuelph) (name_rap_mode(ii), ii = 1, nmodes)
#     WRITE(iuelph) (num_rap_mode(ii), ii = 1, nmodes)
#     WRITE(iuelph) (((s(ii, jj, kk), ii = 1, 3), jj = 1, 3), kk = 1, 48)
#     WRITE(iuelph) (invs(ii), ii = 1, 48)
#     ! FIXME: should disappear
#     WRITE(iuelph) ((ftau(ii, jj), ii = 1, 3), jj = 1, 48)
#     ! end FIXME
#     WRITE(iuelph) ((ft(ii, jj), ii = 1, 3), jj = 1, 48)
#     WRITE(iuelph) (((sr(ii, jj, kk), ii = 1, 3), jj = 1, 3), kk = 1, 48)
#     WRITE(iuelph) (sname(ii), ii = 1, 48)
#     WRITE(iuelph) (t_rev(ii), ii = 1, 48)
#     WRITE(iuelph) ((irt(ii, jj), ii = 1, 48), jj = 1, nat)
#     WRITE(iuelph) ((xk_collect(ii, jj), ii = 1, 3), jj = 1, nkstot)
#     WRITE(iuelph) (wk_collect(ii), ii = 1, nkstot)
#     WRITE(iuelph) ((et_collect(ii, jj), ii = 1, nbnd), jj = 1, nkstot)
#     WRITE(iuelph) ((wg_collect(ii, jj), ii = 1, nbnd), jj = 1, nkstot)
#     WRITE(iuelph) (isk(ii), ii = 1, nkstot)
#     WRITE(iuelph) (ngk_collect(ii), ii = 1, nkstot)
#     WRITE(iuelph) (ikks_collect(ii), ii = 1, nksqtot)
#     WRITE(iuelph) (ikqs_collect(ii), ii = 1, nksqtot)
#     WRITE(iuelph) (eigqts(ii), ii = 1, nat)
#     WRITE(iuelph) (w2(ii), ii = 1, nmodes)
#     WRITE(iuelph) ((dyn(ii, jj), ii = 1, nmodes), jj = 1, nmodes)
#     WRITE(iuelph) ((((el_ph_mat_collect(ii, jj, kk, ll), ii = 1, nbnd), &
#          jj = 1, nbnd), kk = 1, nksqtot), ll = 1, nmodes)


 