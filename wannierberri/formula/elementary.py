from .formula import Formula_ln, Matrix_ln, Matrix_GenDer_ln
from ..utility import cached_einsum
from ..symmetry.point_symmetry import transform_ident, transform_odd

# Fundamental formulae


class Eavln(Matrix_ln):
    """ be careful : this is not a covariant matrix"""

    def __init__(self, data_K):
        super().__init__(0.5 * (data_K.E_K[:, :, None] + data_K.E_K[:, None, :]))
        self.ndim = 0
        self.transformTR = transform_ident
        self.transformInv = transform_ident


class DEinv_ln(Matrix_ln):
    """DEinv_ln.matrix[ik, m, n] = 1 / (E_mk - E_nk)"""

    def __init__(self, data_K):
        super().__init__(data_K.dEig_inv)

    def nn(self, ik, inn, out):
        raise NotImplementedError("dEinv_ln should not be called within inner states")


class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=2), data_K.Dcov)
        self.transformTR = transform_ident
        self.transformInv = transform_ident


class DerWln(Matrix_GenDer_ln):
    r""" :math:`\overline{W}^{bc:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', 2), data_K.covariant('Ham', 3), data_K.Dcov)
        self.transformTR = transform_odd
        self.transformInv = transform_odd


class Dcov(Matrix_ln):

    def __init__(self, data_K):
        super().__init__(data_K.D_H)

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")


class DerDcov(Dcov):

    def __init__(self, data_K):
        self.W = data_K.covariant('Ham', commader=2)
        self.V = data_K.covariant('Ham', gender=1)
        self.D = data_K.Dcov
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        Dln = self.D.ln(ik, inn, out)
        summ = self.W.ln(ik, inn, out)
        tmp = cached_einsum("lpb,pnd->lnbd", self.V.ll(ik, inn, out), Dln)
        tmp -= cached_einsum("lmb,mnd->lnbd", Dln, self.V.nn(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None]
        return summ


class Der2Dcov(Formula_ln):

    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.WV = DerWln(data_K)
        self.dV = InvMass(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.D = Dcov(data_K)
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        summ = self.WV.ln(ik, inn, out)
        Vll = self.V.ll(ik, inn, out)
        Vnn = self.V.nn(ik, inn, out)
        dVll = self.dV.ll(ik, inn, out)
        dVnn = self.dV.nn(ik, inn, out)
        Dln = self.D.ln(ik, inn, out)
        dDln = self.dD.ln(ik, inn, out)
        summ += cached_einsum("lpbe,pnd->lnbde", dVll, Dln)
        summ += cached_einsum("lpde,pnb->lnbde", dVll, Dln)
        summ += cached_einsum("lpe,pnbd->lnbde", Vll, dDln)
        summ += cached_einsum("lpd,pnbe->lnbde", Vll, dDln)
        summ += cached_einsum("lpb,pnde->lnbde", Vll, dDln)
        summ += -cached_einsum("lmde,mnb->lnbde", dDln, Vnn)
        summ += -cached_einsum("lmbd,mne->lnbde", dDln, Vnn)
        summ += -cached_einsum("lmbe,mnd->lnbde", dDln, Vnn)
        summ += -cached_einsum("lmb,mnde->lnbde", Dln, dVnn)
        summ += -cached_einsum("lmd,mnbe->lnbde", Dln, dVnn )
        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None, None]
        return summ

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")
