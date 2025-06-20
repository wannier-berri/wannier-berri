import numpy as np
from ..utility import alpha_A, beta_A, cached_einsum, delta_f
from .formula import Formula_ln, Matrix_ln, Matrix_GenDer_ln, FormulaProduct, FormulaSum, DeltaProduct
from ..symmetry.point_symmetry import transform_ident, transform_odd


class Identity(Formula_ln):

    def __init__(self, data_K=None, **parameters):
        super().__init__(data_K, **parameters)
        self.ndim = 0
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        return np.eye(len(inn))

    def ln(self, ik, inn, out):
        return np.zeros((len(out), len(inn)))


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
        summ = self.W.ln(ik, inn, out)
        tmp = cached_einsum("lpb,pnd->lnbd", self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        tmp = -cached_einsum("lmb,mnd->lnbd", self.D.ln(ik, inn, out), self.V.nn(ik, inn, out))
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
        summ += cached_einsum("lpbe,pnd->lnbde", self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lpde,pnb->lnbde", self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += cached_einsum("lpe,pnbd->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("lpd,pnbe->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("lpb,pnde->lnbde", self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += -cached_einsum("lmde,mnb->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmbd,mne->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmbe,mnd->lnbde", self.dD.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += -cached_einsum("lmb,mnde->lnbde", self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))
        summ += -cached_einsum("lmd,mnbe->lnbde", self.D.ln(ik, inn, out), self.dV.nn(ik, inn, out))

        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None, None]
        return summ

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")

# TODO Der2A,B,O,H,S can be merged to one class.


class Der2A(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.A = data_K.covariant('AA')
        self.dA = data_K.covariant('AA', gender=1)
        self.Abar_de = Matrix_GenDer_ln(data_K.covariant('AA', commader=1), data_K.covariant('AA', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Abar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.A.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dA.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.A.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dA.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        summ = self.Abar_de.ln(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.ln(ik, inn, out), self.A.nn(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.ln(ik, inn, out), self.dA.nn(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.A.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dA.ll(ik, inn, out), self.D.ln(ik, inn, out))
        return summ


class Der2B(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.B = data_K.covariant('BB')
        self.dB = data_K.covariant('BB', gender=1)
        self.Bbar_de = Matrix_GenDer_ln(data_K.covariant('BB', commader=1), data_K.covariant('BB', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Bbar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.B.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dB.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.B.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dB.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        summ = self.Bbar_de.ln(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.ln(ik, inn, out), self.B.nn(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.ln(ik, inn, out), self.dB.nn(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.B.ll(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dB.ll(ik, inn, out), self.D.ln(ik, inn, out))
        return summ


class Der2O(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.O = data_K.covariant('OO')
        self.dO = data_K.covariant('OO', gender=1)
        self.Obar_de = Matrix_GenDer_ln(data_K.covariant('OO', commader=1), data_K.covariant('OO', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Obar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.O.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dO.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.O.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dO.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Der2H(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.H = data_K.covariant('CC')
        self.dH = data_K.covariant('CC', gender=1)
        self.Hbar_de = Matrix_GenDer_ln(data_K.covariant('CC', commader=1), data_K.covariant('CC', commader=2),
                    Dcov(data_K))

    def nn(self, ik, inn, out):
        summ = self.Hbar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.H.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dH.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.H.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dH.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()



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


#############################
#   Third derivative of  E  #
#############################


class Der3E(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.V = data_K.covariant('Ham', commader=1)
        self.D = data_K.Dcov
        self.dV = InvMass(data_K)
        self.dD = DerDcov(data_K)
        self.dW = DerWln(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        summ += 1 * self.dW.nn(ik, inn, out)
        summ += 1 * cached_einsum("mlac,lnb->mnabc", self.dV.nl(ik, inn, out), self.D.ln(ik, inn, out))
        summ += 1 * cached_einsum("mla,lnbc->mnabc", self.V.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += -1 * cached_einsum("mlbc,lna->mnabc", self.dD.nl(ik, inn, out), self.V.ln(ik, inn, out))
        summ += -1 * cached_einsum("mlb,lnac->mnabc", self.D.nl(ik, inn, out), self.dV.ln(ik, inn, out))

        # TODO: alternatively: add factor 0.5 to first term, remove 4th and 5th, and ad line below.
        # Should give the same, I think, but needs to be tested
        # summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   Berry curvature    #
########################


class Omega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.O = data_K.covariant('OO')

        self.ndim = 1
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.O.nn(ik, inn, out)
            summ += -1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.A.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.A.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.A.nn(ik, inn, out)[:, :, alpha_A],
                self.A.nn(ik, inn, out)[:, :, beta_A])

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   derivative of      #
#   Berry curvature    #
########################


class DerOmega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.dO = data_K.covariant('OO', gender=1)
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.external_terms:
            summ += 0.5 * self.dO.nn(ik, inn, out)

        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            if self.internal_terms:
                summ += -1j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dD.ln(ik, inn, out)[:, :, b])
                pass

            if self.external_terms:
                summ += -1 * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum(
                    "mlcd,lnc->mncd",
                    self.dD.nl(ik, inn, out)[:, :, a, :],
                    self.A.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.A.nn(ik, inn, out)[:, :, a],
                    self.dA.nn(ik, inn, out)[:, :, b, :])
                pass

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

###############################
###  second derivative of  ####
###  Berry curvature       ####
###############################


class Der2Omega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddD = Der2Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.ddA = Der2A(data_K)
            self.ddO = Der2O(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.external_terms:
            summ += 0.5 * self.ddO.nn(ik, inn, out)

        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            if self.internal_terms:
                summ += -1j * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a], self.dD.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a], self.ddD.ln(ik, inn, out)[:, :, b])
                pass

            if self.external_terms:
                summ += -1 * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a], self.dA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a], self.ddA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * cached_einsum("mlcde,lnc->mncde", self.ddD.nl(ik, inn, out)[:, :, a, :],
                        self.A.ln(ik, inn, out)[:, :, b])
                summ += -1 * s * cached_einsum("mlcd,lnce->mncde", self.dD.nl(ik, inn, out)[:, :, a, :],
                        self.dA.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mlce,lncd->mncde", self.dA.nn(ik, inn, out)[:, :, a], self.dA.nn(ik, inn, out)[:, :, b, :])
                summ += -1j * s * cached_einsum("mlc,lncde->mncde", self.A.nn(ik, inn, out)[:, :, a], self.ddA.nn(ik, inn, out)[:, :, b, :])
                pass

        summ += summ.swapaxes(0, 1).conj()
        return summ


    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Hamiltonian(Matrix_ln):

    def __init__(self, data_K):
        v = data_K.covariant('Ham', gender=0)
        self.__dict__.update(v.__dict__)


class Velocity(Matrix_ln):

    def __init__(self, data_K, external_terms=False):
        v = data_K.covariant('Ham', gender=1)
        self.__dict__.update(v.__dict__)
        if external_terms:
            self.matrix = self.matrix + 1j * data_K.Xbar('AA') * (
                data_K.E_K[:, :, None, None] - data_K.E_K[:, None, :, None])


class Spin(Matrix_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS')
        self.__dict__.update(s.__dict__)


class DerSpin(Matrix_GenDer_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS', gender=1)
        self.__dict__.update(s.__dict__)


class Der2Spin(Formula_ln):
    def __init__(self, data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.S = data_K.covariant('SS')
        self.dS = data_K.covariant('SS', gender=1)
        self.Sbar_de = Matrix_GenDer_ln(data_K.covariant('SS', commader=1), data_K.covariant('SS', commader=2),
                    Dcov(data_K))
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = self.Sbar_de.nn(ik, inn, out)
        summ -= cached_einsum("mlde,lnb...->mnb...de", self.dD.nl(ik, inn, out), self.S.ln(ik, inn, out))
        summ -= cached_einsum("mld,lnb...e->mnb...de", self.D.nl(ik, inn, out), self.dS.ln(ik, inn, out))
        summ += cached_einsum("mlb...,lnde->mnb...de", self.S.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += cached_einsum("mlb...e,lnd->mnb...de", self.dS.nl(ik, inn, out), self.D.ln(ik, inn, out))

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   orbital moment     #
########################


class Morb_H(Formula_ln):

    def __init__(self, data_K, **parameters):
        r"""  :math:`\varepcilon_{abc} \langle \partial_a u | H | \partial_b \rangle` """
        super().__init__(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.B = data_K.covariant('BB')
            self.C = data_K.covariant('CC')
        self.D = data_K.Dcov
        self.E = data_K.E_K
        self.ndim = 1
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A] * self.E[ik][out][None, :, None],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.C.nn(ik, inn, out)
            summ += -1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.B.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * cached_einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.B.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * cached_einsum(
                "mlc,lnc->mnc",
                self.A.nn(ik, inn, out)[:, :, alpha_A] * self.E[ik][inn][None, :, None],
                self.A.nn(ik, inn, out)[:, :, beta_A])
        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Morb_Hpm(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        r""" Morb_H  +- (En+Em)/2 * Omega """
        super().__init__(data_K, **parameters)
        self.H = Morb_H(data_K, **parameters)
        self.sign = sign
        if self.sign != 0:
            self.O = Omega(data_K, **parameters)
            self.Eav = Eavln(data_K)
        self.ndim = 1
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        res = self.H.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * self.Eav.nn(ik, inn, out)[:, :, None] * self.O.nn(ik, inn, out)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class morb(Morb_Hpm):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)


########################
#   derivative of      #
#   orbital moment     #
########################

class DerMorb_H(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.E = data_K.E_K
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB', gender=1)
            self.dH = data_K.covariant('CC', gender=1)
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -2j * cached_einsum(
                "mpc,pld,lnc->mncd",
                self.D.nl(ik, inn, out)[:, :, alpha_A], self.V.ll(ik, inn, out),
                self.D.ln(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.E[ik][out][:, None, None, None] * self.dD.ln(ik, inn, out)[:, :, b])
        if self.external_terms:
            summ += 1 * self.dH.nn(ik, inn, out)
            summ += -2j * cached_einsum(
                "mpc,pld,lnc->mncd",
                self.A.nn(ik, inn, out)[:, :, alpha_A], self.V.nn(ik, inn, out),
                self.A.nn(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.A.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None],
                    self.dA.nn(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum(
                    "mlc,lncd->mncd", (self.B.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2).conj(),
                    self.dD.ln(ik, inn, out)[:, :, b, :])

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class DerMorb(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        super().__init__(data_K, **parameters)
        self.dermorb_H = DerMorb_H(data_K, **parameters)
        self.sign = sign
        if self.sign != 0:
            self.V = data_K.covariant('Ham', commader=1)
            self.dO = DerOmega(data_K, **parameters)
            self.O = Omega(data_K, **parameters)
            self.E = data_K.E_K
        self.ndim = 2
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def nn(self, ik, inn, out):
        res = self.dermorb_H.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * cached_einsum("mlc,lnd->mncd", self.O.nn(ik, inn, out), self.V.nn(ik, inn, out))
            res += self.sign * self.E[ik][inn][:, None, None, None] * self.dO.nn(ik, inn, out)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Dermorb(DerMorb):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)


##############################
###  second derivative of ####
###   orbital moment      ####
##############################

class Der2Morb_H(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.ddD = Der2Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.dV = InvMass(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.E = data_K.E_K
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.ddA = Der2A(data_K)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB', gender=1)
            self.ddB = Der2B(data_K)
            self.dH = data_K.covariant('CC', gender=1)
            self.ddH = Der2H(data_K)
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident
    # TODO merge term if possible.

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -2j * cached_einsum("mpc,plde,lnc->mncde", self.D.nl(ik, inn, out)[:, :, alpha_A],
                    self.dV.ll(ik, inn, out), self.D.ln(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -1j * s * cached_einsum("mpce,pld,lnc->mncde", self.dD.nl(ik, inn, out)[:, :, a, :],
                        self.V.ll(ik, inn, out), self.D.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * cached_einsum("mpc,pld,lnce->mncde", self.D.nl(ik, inn, out)[:, :, a],
                        self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out)[:, :, b, :])
                summ += -2j * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a, :],
                    self.E[ik][out][:, None, None, None] * self.dD.ln(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a],
                    self.E[ik][out][:, None, None, None, None] * self.ddD.ln(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mpc,ple,lncd->mncde", self.D.nl(ik, inn, out)[:, :, a],
                    self.V.ll(ik, inn, out), self.dD.ln(ik, inn, out)[:, :, b])
        if self.external_terms:
            summ += 1 * self.ddH.nn(ik, inn, out)
            summ += -2j * cached_einsum("mpc,plde,lnc->mncde", self.A.nn(ik, inn, out)[:, :, alpha_A],
                    self.dV.nn(ik, inn, out), self.A.nn(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -1j * cached_einsum("mpce,pld,lnc->mncde", self.dA.nn(ik, inn, out)[:, :, a],
                        self.V.nn(ik, inn, out), self.A.nn(ik, inn, out)[:, :, b])
                summ += -1j * cached_einsum("mpc,pld,lnce->mncde", self.A.nn(ik, inn, out)[:, :, a],
                        self.V.nn(ik, inn, out), self.dA.nn(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlce,lncd->mncde",
                        self.dA.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None, None], self.dA.nn(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlc,lncde->mncde",
                        self.A.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None], self.ddA.nn(ik, inn, out)[:, :, b])
                summ += -2j * s * cached_einsum("mlc,ple,lncd->mncde", self.A.nn(ik, inn, out)[:, :, a],
                        self.V.nn(ik, inn, out), self.dA.nn(ik, inn, out)[:, :, b])
                summ += -2 * s * cached_einsum("mlce,lncd->mncde", self.dD.nl(ik, inn, out)[:, :, a],
                        self.dB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum("mlc,lncde->mncde", self.D.nl(ik, inn, out)[:, :, a],
                        self.ddB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * cached_einsum("mlce,lncd->mncde", (self.dB.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2, 3).conj(),
                        self.dD.ln(ik, inn, out)[:, :, b])
                summ += -2 * s * cached_einsum("mlc,lncde->mncde", (self.B.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2).conj(),
                        self.ddD.ln(ik, inn, out)[:, :, b])

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Der2Morb(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        super().__init__(data_K, **parameters)
        self.der2morb_H = Der2Morb_H(data_K, **parameters)
        self.sign = sign
        if self.sign != 0:
            self.V = data_K.covariant('Ham', commader=1)
            self.dV = InvMass(data_K)
            self.dO = DerOmega(data_K, **parameters)
            self.ddO = Der2Omega(data_K, **parameters)
            self.O = Omega(data_K, **parameters)
            self.E = data_K.E_K
        self.ndim = 3
        self.transformTR = transform_odd
        self.transformInv = transform_ident

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        res = self.der2morb_H.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * cached_einsum("mlce,lnd->mncde", self.dO.nn(ik, inn, out), self.V.nn(ik, inn, out))
            res += self.sign * cached_einsum("mlc,lnde->mncde", self.O.nn(ik, inn, out), self.dV.nn(ik, inn, out))
            res += self.sign * cached_einsum("mle,lncd->mncde", self.V.nn(ik, inn, out), self.dO.nn(ik, inn, out))
            res += self.sign * self.E[ik][inn][:, None, None, None, None] * self.ddO.nn(ik, inn, out)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class Der2morb(Der2Morb):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)


########################
#   spin transport     #
########################

def _spin_velocity_einsum_opt(C, A, B):
    # Optimized version of C += cached_einsum('knls,klma->knmas', A, B). Used in shc_B_H.
    nk = C.shape[0]
    nw = C.shape[1]
    for ik in range(nk):
        # Performing C[ik] += cached_einsum('nls,lma->nmas', A[ik], B[ik])
        tmp_a = np.swapaxes(A[ik], 1, 2)  # nls -> nsl
        tmp_a = np.reshape(tmp_a, (nw * 3, nw))  # nsl -> (ns)l
        tmp_b = np.reshape(B[ik], (nw, nw * 3))  # lma -> l(ma)
        tmp_c = tmp_a @ tmp_b  # (ns)l, l(ma) -> (ns)(ma)
        tmp_c = np.reshape(tmp_c, (nw, 3, nw, 3))  # (ns)(ma) -> nsma
        C[ik] += np.transpose(tmp_c, (0, 2, 3, 1))  # nsma -> nmas


class SpinVelocity(Matrix_ln):
    """spin current matrix elements. SpinVelocity.matrix[ik, m, n, a, s] = <u_mk|{v^a S^s}|u_nk> / 2"""

    def __init__(self, data_K, spin_current_type, external_terms=True):
        if spin_current_type == "simple":
            # tight-binding case
            super().__init__(self._J_H_simple(data_K, external_terms=external_terms))
        elif spin_current_type == "qiao":
            # J. Qiao et al PRB (2018)
            super().__init__(self._J_H_qiao(data_K, external_terms=external_terms))
        elif spin_current_type == "ryoo":
            # J. H. Ryoo et al PRB (2019)
            super().__init__(self._J_H_ryoo(data_K, external_terms=external_terms))
        else:
            raise ValueError(f"spin_current_type must be `qiao` or `ryoo` or `simple`, not {spin_current_type}")
        self.transformTR = transform_ident
        self.transformInv = transform_odd

    def _J_H_simple(self, data_K, external_terms=True):
        # Spin current operator, J. Qiao et al PRB (2019)
        # J_H[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        S = data_K.Xbar('SS')
        V = Velocity(data_K, external_terms=external_terms).matrix
        J = cached_einsum("klms,kmna->klnas", S, V)
        return (J + J.swapaxes(1, 2).conj()) / 2

    def _J_H_qiao(self, data_K, external_terms=True):
        if not external_terms:
            raise NotImplementedError(
                "spin Hall qiao without external terms is not implemented yet. Use `SHC_type='simple'`")
        # Spin current operator, J. Qiao et al PRB (2019)
        # J_H_qiao[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        SS_H = data_K.Xbar('SS')
        SH_H = data_K.Xbar("SH")
        shc_K_H = -1j * data_K.Xbar("SR")
        _spin_velocity_einsum_opt(shc_K_H, SS_H, data_K.D_H)
        shc_L_H = -1j * data_K._R_to_k_H(data_K.get_R_mat('SHR'), hermitian=False)
        _spin_velocity_einsum_opt(shc_L_H, SH_H, data_K.D_H)
        J = (
            data_K.delE_K[:, None, :, :, None] * SS_H[:, :, :, None, :] +
            data_K.E_K[:, None, :, None, None] * shc_K_H[:, :, :, :, :] - shc_L_H)
        return (J + J.swapaxes(1, 2).conj()) / 2

    def _J_H_ryoo(self, data_K, external_terms=True):
        if not external_terms:
            raise NotImplementedError(
                "spin Hall ryoo without external terms is not implemented yet. Use `SHC_type='simple'`")
        # Spin current operator, J. H. Ryoo et al PRB (2019)
        # J_H_ryoo[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        SA_H = data_K.Xbar("SA")
        SHA_H = data_K.Xbar("SHA")
        J = -1j * (data_K.E_K[:, None, :, None, None] * SA_H - SHA_H)
        _spin_velocity_einsum_opt(J, data_K.Xbar('SS'), data_K.Xbar('Ham', 1))
        return (J + J.swapaxes(1, 2).conj()) / 2


class SpinOmega(Formula_ln):
    """spin Berry curvature"""

    def __init__(self, data_K, spin_current_type="ryoo", **parameters):
        super().__init__(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
        self.D = data_K.Dcov
        self.J = SpinVelocity(data_K, spin_current_type, external_terms=self.external_terms)
        self.dEinv = DEinv_ln(data_K)
        self.ndim = 3
        self.transformTR = transform_ident
        self.transformInv = transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)

        # v_over_de[l,n,b] = v[l,n,b] / (e[n] - e[l]) = D[l,n,b] - 1j * A[l,n,b]
        v_over_de = self.D.ln(ik, inn, out)
        if self.external_terms:
            v_over_de += - 1j * self.A.ln(ik, inn, out)

        # j_over_de[m,l,a,s] = j[m,l,a,s] / (e[m] - e[l])
        j_over_de = self.J.nl(ik, inn, out) * self.dEinv.nl(ik, inn, out)[:, :, None, None]

        summ += -2 * cached_einsum("mlas,lnb->mnabs", j_over_de, v_over_de).imag

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


####################################
#                                  #
#    Some Prooducts                #
#                                  #
####################################

class VelOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Omega(data_K, **kwargs_formula)], name='VelOmega')


class VelHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Morb_Hpm(data_K, sign=+1, **kwargs_formula)],
                         name='VelHplus')


class VelSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Spin(data_K)], name='VelSpin')


class VelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1)], name='VelVel')


class VelVelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1),
                          data_K.covariant('Ham', commader=1)], name='VelVelVel')


class MassVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), data_K.covariant('Ham', commader=1)], name='MassVel')


class MassMass(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), InvMass(data_K)], name='MassMass')


class VelMassVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), InvMass(data_K),
                          data_K.covariant('Ham', commader=1)], name='VelMassVel')


class OmegaS(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Spin(data_K)], name='SpinOmega')


class OmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='OmegaOmega')


class OmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)],
                         name='OmegaHplus')


class emcha_surf(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        velocity = data_K.covariant('Ham', commader=1)
        formula1 = FormulaProduct([InvMass(data_K), Omega(data_K, **kwargs_formula), velocity],
            name='mass-berry-vel (apus)(psua) ([au]bpbs) ([us]abbp)')
        formula2 = FormulaProduct([velocity, DerOmega(data_K, **kwargs_formula), velocity],
            name='v-derberry-vel (aups) ([au]bbps) ([us]abpb)')
        tmp = FormulaSum([formula2, formula1], [1, 1], ['aups', 'apus'])
        super().__init__([
            tmp,
            DeltaProduct(delta_f, formula1, 'us,MLabbp->MLaups'),
            DeltaProduct(delta_f, tmp, 'au,MLbbps->MLaups'),
            DeltaProduct(delta_f, formula2, 'us,MLabpb->MLaups'),
            formula2,
            # romove antisymmatric part
            # DeltaProduct(Levi_Civita
            #    DeltaProduct(Levi_Civita,tmp,'pta,MLxtbs->MLaxpbs'),
            #    'xub,MLaxpbs->MLaups')
        ],
            # [2,-2,-1,1,-1,-1],['aups','aups','aups','aups','aups','aups'])
            [2, -2, -1, 1, -1], ['aups', 'aups', 'aups', 'aups', 'aups'])


class NLDrude_Z_spin(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Spin(data_K)], name='Der3ESpin')
        term2 = FormulaProduct([Der2Spin(data_K), data_K.covariant('Ham', commader=1)], name='Der2SpinVel')
        super().__init__([term1, term2], [-1, 1], ['apsu', 'uaps'])


class NLDrude_Z_orb_Hplus(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='Der3EHplus')
        term2 = FormulaProduct([Der2Morb(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='Der2HplusVel')
        super().__init__([term1, term2], [-1, 1], ['apsu', 'uaps'])


class NLDrude_Z_orb_Omega(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Omega(data_K, **kwargs_formula)], name='Der3EOmega')
        term2 = FormulaProduct([Der2Omega(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='Der2OmegaVel')
        super().__init__([term1, term2], [-1, 1], ['apsu', 'uaps'])
