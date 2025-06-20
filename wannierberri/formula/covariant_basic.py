import numpy as np
from ..utility import alpha_A, beta_A, cached_einsum
from .formula import Formula_ln
from .covariant import DerDcov, Eavln
from ..symmetry.point_symmetry import transform_ident, transform_odd

""" The following  Formulue are fundamental. They can be used to construct all
quantities relatred to Berry curvature and orbital magnetic moment. They are written
in the most explicit form, although probably not the most efecient.
Foe practical reasons more eficient formulae may be constructed (e.g. by  excluding
some terms that cancel out). However, the following may be used as benchmark.
"""

########################
#   Berry curvature    #
########################


class tildeFab(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov

        #        print (f"tildeFab evaluating: internal({self.internal_terms}) and external({self.external_terms})")
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.V = data_K.covariant('Ham', gender=1)
            self.F = data_K.covariant('FF')

        self.ndim = 2
#        self.Iodd=False
#        self.TRodd=True

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        Dnl = self.D.nl(ik, inn, out)
        Dln = self.D.ln(ik, inn, out)
        if self.internal_terms:
            summ += -cached_einsum("mla,lnb->mnab", Dnl, Dln)

        if self.external_terms:
            summ += self.F.nn(ik, inn, out)
            summ += 2j * cached_einsum("mla,lnb->mnab", Dnl, self.A.ln(ik, inn, out))
            #            summ += 1j * cached_einsum("mla,lnb->mnab", self.A.nl (ik,inn,out),Dln   )
            summ += -1 * cached_einsum("mla,lnb->mnab", self.A.nn(ik, inn, out), self.A.nn(ik, inn, out))

        summ = 0.5 * (summ + summ.transpose((1, 0, 3, 2)).conj())
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   derivative of      #
#   Berry curvature    #
########################


class tildeFab_d(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = data_K.Dcov

        #        print (f"derOmega evaluating: internal({self.internal_terms}) and external({self.external_terms})")

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.dF = data_K.covariant('FF', gender=1)
        self.ndim = 3
#        self.Iodd=True
#        self.TRodd=False

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        Dnl = self.D.nl(ik, inn, out)
        dDln = self.dD.ln(ik, inn, out)

        if self.internal_terms:
            summ += -2 * cached_einsum("mla,lnbd->mnabd", Dnl, dDln)

        if self.external_terms:
            summ += self.dF.nn(ik, inn, out)
            summ += 2j * cached_einsum("mla,lnbd->mnabd", Dnl, self.dA.ln(ik, inn, out))
            summ += 2j * cached_einsum("mla,lnbd->mnabd", self.A.nl(ik, inn, out), dDln)
            summ += -2 * cached_einsum("mla,lnbd->mnabd", self.A.nn(ik, inn, out), self.dA.nn(ik, inn, out))

#  Terms (a<->b, m<-n> )*   are accounted above by factor 2
        summ = 0.5 * (summ + summ.transpose((1, 0, 3, 2, 4)).conj())
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


###############################################################
#  tildeHab = <\tilde\partial_a u | H |\tilde\partial_b u>    #
###############################################################


class tildeHab(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.B = data_K.covariant('BB')
            self.H = data_K.covariant('CCab')
        self.D = data_K.Dcov
        self.E = data_K.E_K
        self.ndim = 2
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)

        if self.internal_terms:
            summ += -cached_einsum(
                "mla,lnb->mnab",
                self.D.nl(ik, inn, out) * self.E[ik][out][None, :, None], self.D.ln(ik, inn, out))

        if self.external_terms:
            summ += self.H.nn(ik, inn, out)
            summ += 2j * cached_einsum("mla,lnb->mnab", self.D.nl(ik, inn, out), self.B.ln(ik, inn, out))
            summ += -cached_einsum(
                "mla,lnb->mnab",
                self.A.nn(ik, inn, out) * self.E[ik][inn][None, :, None], self.A.nn(ik, inn, out))
        summ = 0.5 * (summ + summ.transpose((1, 0, 3, 2)).conj())
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class tildeHGab(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        self.F = tildeFab(data_K, **parameters)
        self.H = tildeHab(data_K, **parameters)
        self.E = Eavln(data_K)
        self.sign = sign
        self.ndim = 2

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        return self.H.nn(ik, inn, out) + self.sign * self.E.nn(ik, inn, out)[:, :, None, None] * self.F.nn(ik, inn, out)

    def ln(self, ik, inn, out):
        raise NotImplementedError()


##################################################################################
#  tildeHab:d = \tilde\partial_d <\tilde\partial_a u | H |\tilde\partial_b u>    #
##################################################################################


class tildeHab_d(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = data_K.Dcov
        self.V = data_K.covariant('Ham', gender=1)
        self.E = data_K.E_K
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB', gender=1)
            self.dH = data_K.covariant('CCab', gender=1)
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -1 * cached_einsum(
                "mpa,pld,lnb->mnabd", self.D.nl(ik, inn, out), self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
            summ += -2 * cached_einsum(
                "mla,lnbd->mnabd",
                self.D.nl(ik, inn, out) * self.E[ik][out][None, :, None], self.dD.ln(ik, inn, out))

        if self.external_terms:
            summ += self.dH.nn(ik, inn, out)
            summ += -cached_einsum(
                "mpa,pld,lnb->mnabd", self.A.nn(ik, inn, out), self.V.nn(ik, inn, out), self.A.nn(ik, inn, out))
            summ += -2 * cached_einsum(
                "mla,lnbd->mnabd",
                self.A.nn(ik, inn, out) * self.E[ik][inn][None, :, None], self.dA.nn(ik, inn, out))
            summ += 2j * cached_einsum("mla,lnbd->mnabd", self.D.nl(ik, inn, out), self.dB.ln(ik, inn, out))
            summ += 2j * cached_einsum("lma,lnbd->mnabd", (self.B.ln(ik, inn, out)).conj(), self.dD.ln(ik, inn, out))

        summ = 0.5 * (summ + summ.transpose((1, 0, 3, 2, 4)).conj())
        return summ

    @property
    def additive(self):
        return False

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class tildeHGab_d(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        self.F = tildeFab(data_K, **parameters)
        self.dF = tildeFab_d(data_K, **parameters)
        self.dH = tildeHab_d(data_K, **parameters)
        self.E = Eavln(data_K)
        self.V = data_K.covariant('Ham', gender=1)
        self.sign = sign
        self.ndim = 3

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        res = self.dH.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * self.E.nn(ik, inn, out)[:, :, None, None, None] * self.dF.nn(ik, inn, out)
            res += 0.5 * self.sign * cached_einsum("mpab,pnd->mnabd", self.F.nn(ik, inn, out), self.V.nn(ik, inn, out))
            res += 0.5 * self.sign * cached_einsum("mpd,pnab->mnabd", self.V.nn(ik, inn, out), self.F.nn(ik, inn, out))

        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()


###################################################
#   Now define their anitsymmetric combinations   #
###################################################


class AntiSymmetric(Formula_ln):

    def __init__(self, full, data_K, **parameters):
        self.full = full(data_K, **parameters)
        self.ndim = self.full.ndim - 1

    def nn(self, ik, inn, out):
        fab = self.full.nn(ik, inn, out)
        return 1j * (fab[:, :, alpha_A, beta_A] - fab[:, :, beta_A, alpha_A])

    def ln(self, ik, inn, out):
        fab = self.full.ln(ik, inn, out)
        return 1j * (fab[:, :, alpha_A, beta_A] - fab[:, :, beta_A, alpha_A])


class Symmetric(Formula_ln):

    def __init__(self, full, data_K, axes=[0, 1], **parameters):
        self.full = full(data_K, **parameters)
        self.ndim = self.full.ndim
        self.axes = axes

    def nn(self, ik, inn, out):
        fab = self.full.nn(ik, inn, out)
        return fab + fab.swapaxes(self.axes[0] + 2, self.axes[1] + 2)

    def ln(self, ik, inn, out):
        fab = self.full.nn(ik, inn, out)
        return fab + fab.swapaxes(self.axes[0] + 2, self.axes[1] + 2)


class tildeFc(AntiSymmetric):

    def __init__(self, data_K, **parameters):
        super().__init__(tildeFab, data_K, **parameters)
        self.transformTR = transform_odd
        self.transformInv = transform_ident


class tildeHGc(AntiSymmetric):

    def __init__(self, data_K, **parameters):
        super().__init__(tildeHGab, data_K, **parameters)
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    @property
    def additive(self):
        return False


class tildeFc_d(AntiSymmetric):

    def __init__(self, data_K, **parameters):
        super().__init__(tildeFab_d, data_K, **parameters)
        self.transformTR = transform_ident
        self.transformInv = transform_odd


class tildeHGc_d(AntiSymmetric):

    def __init__(self, data_K, **parameters):
        super().__init__(tildeHGab_d, data_K, **parameters)
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    @property
    def additive(self):
        return False


# derivative of band orbital moment
class Der_morb(tildeHGc_d):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)
