import numpy as np
from wannierberri.__utility import alpha_A, beta_A, delta_f, Levi_Civita
from . import Formula_ln, Matrix_ln, Matrix_GenDer_ln, FormulaProduct, FormulaSum, DeltaProduct


class Identity(Formula_ln):

    def __init__(self,data_K=None):
        self.ndim = 0
        self.TRodd = False
        self.Iodd = False

    def nn(self, ik, inn, out):
        return np.eye(len(inn))

    def ln(self, ik, inn, out):
        return np.zeros((len(out), len(inn)))


class Eavln(Matrix_ln):
    """ be careful : this is not a covariant matrix"""

    def __init__(self, data_K):
        super().__init__(0.5 * (data_K.E_K[:, :, None] + data_K.E_K[:, None, :]))
        self.ndim = 0
        self.TRodd = False
        self.Iodd = False


class DEinv_ln(Matrix_ln):
    "DEinv_ln.matrix[ik, m, n] = 1 / (E_mk - E_nk)"

    def __init__(self, data_K):
        super().__init__(data_K.dEig_inv)

    def nn(self, ik, inn, out):
        raise NotImplementedError("dEinv_ln should not be called within inner states")


class Dcov(Matrix_ln):

    def __init__(self, data_K):
        super().__init__(data_K.D_H)

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")


class DerDcov(Formula_ln):

    def __init__(self, data_K):
        self.W = data_K.covariant('Ham', commader=2)
        self.V = data_K.covariant('Ham', gender=1)
        self.D = Dcov(data_K)
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        summ = self.W.ln(ik, inn, out)
        tmp = np.einsum("lpb,pnd->lnbd", self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        tmp = -np.einsum("lmb,mnd->lnbd", self.D.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None]
        return summ

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")


class Der2Dcov(Formula_ln):

    def __init__(self,data_K):
        self.dD = DerDcov(data_K)
        self.WV = DerWln(data_K)
        self.dV = InvMass(data_K)
        self.V = data_K.covariant('Ham',gender=1)
        self.D = Dcov(data_K)
        self.dEinv = DEinv_ln(data_K)

    def ln(self,ik,inn,out):
        summ = self.WV.ln(ik,inn,out)
        summ += np.einsum( "lpbe,pnd->lnbde" , self.dV.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        summ += np.einsum( "lpde,pnb->lnbde" , self.dV.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        summ += np.einsum( "lpe,pnbd->lnbde" , self.V.ll(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "lpd,pnbe->lnbde" , self.V.ll(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "lpb,pnde->lnbde" , self.V.ll(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += -np.einsum( "lmde,mnb->lnbde" , self.dD.ln(ik,inn,out) , self.V.nn(ik,inn,out) )
        summ += -np.einsum( "lmbd,mne->lnbde" , self.dD.ln(ik,inn,out) , self.V.nn(ik,inn,out) )
        summ += -np.einsum( "lmbe,mnd->lnbde" , self.dD.ln(ik,inn,out) , self.V.nn(ik,inn,out) )
        summ += -np.einsum( "lmb,mnde->lnbde" , self.D.ln(ik,inn,out) , self.dV.nn(ik,inn,out) )
        summ += -np.einsum( "lmd,mnbe->lnbde" , self.D.ln(ik,inn,out) , self.dV.nn(ik,inn,out) )
        summ *= -self.dEinv.ln(ik,inn,out)[:,:,None,None,None]
        return summ

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")


#TODO Der2A,B,O can be merged to one class.
class Der2A(Formula_ln):
    def __init__(self,data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.A  = data_K.covariant('AA')
        self.dA = data_K.covariant('AA',gender=1)
        self.Abar_de  = Matrix_GenDer_ln(data_K.covariant('AA',commader=1),data_K.covariant('AA',commader=2),
                    Dcov(data_K) ,Iodd=None,TRodd=None)
    def nn(self,ik,inn,out):
        summ = self.Abar_de.nn(ik,inn,out)
        summ -= np.einsum( "mlde,lnb...->mnb...de" , self.dD.nl(ik,inn,out) , self.A.ln(ik,inn,out) )
        summ -= np.einsum( "mld,lnb...e->mnb...de" , self.D.nl(ik,inn,out) , self.dA.ln(ik,inn,out) )
        summ += np.einsum( "mlb...,lnde->mnb...de" , self.A.nl(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "mlb...e,lnd->mnb...de" , self.dA.nl(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ

    def ln(self,ik,inn,out):
        summ = self.Abar_de.ln(ik,inn,out)
        summ -= np.einsum( "mlde,lnb...->mnb...de" , self.dD.ln(ik,inn,out) , self.A.nn(ik,inn,out) )
        summ -= np.einsum( "mld,lnb...e->mnb...de" , self.D.ln(ik,inn,out) , self.dA.nn(ik,inn,out) )
        summ += np.einsum( "mlb...,lnde->mnb...de" , self.A.ll(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "mlb...e,lnd->mnb...de" , self.dA.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ


class Der2B(Formula_ln):
    def __init__(self,data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.B  = data_K.covariant('BB')
        self.dB = data_K.covariant('BB',gender=1)
        self.Bbar_de = Matrix_GenDer_ln(data_K.covariant('BB',commader=1),data_K.covariant('BB',commader=2),
                    Dcov(data_K) ,Iodd=None,TRodd=None)
    def nn(self,ik,inn,out):
        summ = self.Bbar_de.nn(ik,inn,out)
        summ -= np.einsum( "mlde,lnb...->mnb...de" , self.dD.nl(ik,inn,out) , self.B.ln(ik,inn,out) )
        summ -= np.einsum( "mld,lnb...e->mnb...de" , self.D.nl(ik,inn,out) , self.dB.ln(ik,inn,out) )
        summ += np.einsum( "mlb...,lnde->mnb...de" , self.B.nl(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "mlb...e,lnd->mnb...de" , self.dB.nl(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ

    def ln(self,ik,inn,out):
        summ = self.Bbar_de.ln(ik,inn,out)
        summ -= np.einsum( "mlde,lnb...->mnb...de" , self.dD.ln(ik,inn,out) , self.B.nn(ik,inn,out) )
        summ -= np.einsum( "mld,lnb...e->mnb...de" , self.D.ln(ik,inn,out) , self.dB.nn(ik,inn,out) )
        summ += np.einsum( "mlb...,lnde->mnb...de" , self.B.ll(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "mlb...e,lnd->mnb...de" , self.dB.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ


class Der2O(Formula_ln):
    def __init__(self,data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.O  = data_K.covariant('OO')
        self.dO = data_K.covariant('OO',gender=1)
        self.Obar_de  = Matrix_GenDer_ln(data_K.covariant('OO',commader=1),data_K.covariant('OO',commader=2),
                    Dcov(data_K) ,Iodd=False ,TRodd=True)
    def nn(self,ik,inn,out):
        summ = self.Obar_de.nn(ik,inn,out)
        summ -= np.einsum( "mlde,lnb...->mnb...de" , self.dD.nl(ik,inn,out) , self.O.ln(ik,inn,out) )
        summ -= np.einsum( "mld,lnb...e->mnb...de" , self.D.nl(ik,inn,out) , self.dO.ln(ik,inn,out) )
        summ += np.einsum( "mlb...,lnde->mnb...de" , self.O.nl(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "mlb...e,lnd->mnb...de" , self.dO.nl(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()


class Der2H(Formula_ln):
    def __init__(self,data_K):
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.H  = data_K.covariant('CC')
        self.dH = data_K.covariant('CC',gender=1)
        self.Hbar_de  = Matrix_GenDer_ln(data_K.covariant('CC',commader=1),data_K.covariant('CC',commader=2),
                    Dcov(data_K) ,Iodd=False ,TRodd=True)
    def nn(self,ik,inn,out):
        summ = self.Hbar_de.nn(ik,inn,out)
        summ -= np.einsum( "mlde,lnb...->mnb...de" , self.dD.nl(ik,inn,out) , self.H.ln(ik,inn,out) )
        summ -= np.einsum( "mld,lnb...e->mnb...de" , self.D.nl(ik,inn,out) , self.dH.ln(ik,inn,out) )
        summ += np.einsum( "mlb...,lnde->mnb...de" , self.H.nl(ik,inn,out) , self.dD.ln(ik,inn,out) )
        summ += np.einsum( "mlb...e,lnd->mnb...de" , self.dH.nl(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()


class Velocity(Matrix_ln):

    def __init__(self, data_K):
        v = data_K.covariant('Ham', gender=1)
        self.__dict__.update(v.__dict__)

class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=2), Dcov(data_K))
        self.TRodd = False
        self.Iodd = False


class DerWln(Matrix_GenDer_ln):
    r""" :math:`\overline{W}^{bc:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', 2), data_K.covariant('Ham', 3), Dcov(data_K))
        self.TRodd = False
        self.Iodd = False


#############################
#   Third derivative of  E  #
#############################


class Der3E(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.V = data_K.covariant('Ham', commader=1)
        self.D = Dcov(data_K)
        self.dV = InvMass(data_K)
        self.dD = DerDcov(data_K)
        self.dW = DerWln(data_K)
        self.ndim = 3
        self.Iodd = True
        self.TRodd = True

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        summ += 1 * self.dW.nn(ik, inn, out)
        summ += 1 * np.einsum("mlac,lnb->mnabc", self.dV.nl(ik, inn, out), self.D.ln(ik, inn, out))
        summ += 1 * np.einsum("mla,lnbc->mnabc", self.V.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += -1 * np.einsum("mlbc,lna->mnabc", self.dD.nl(ik, inn, out), self.V.ln(ik, inn, out))
        summ += -1 * np.einsum("mlb,lnac->mnabc", self.D.nl(ik, inn, out), self.dV.ln(ik, inn, out))

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
        self.D = Dcov(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.O = data_K.covariant('OO')

        self.ndim = 1
        self.Iodd = False
        self.TRodd = True

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.O.nn(ik, inn, out)
            summ += -1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.A.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.A.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * np.einsum(
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
        self.D = Dcov(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.dO = data_K.covariant('OO', gender=1)
        self.ndim = 2
        self.Iodd = True
        self.TRodd = False

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.external_terms:
            summ += 0.5 * self.dO.nn(ik, inn, out)

        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            if self.internal_terms:
                summ += -1j * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dD.ln(ik, inn, out)[:, :, b])
                pass

            if self.external_terms:
                summ += -1 * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * np.einsum(
                    "mlcd,lnc->mncd",
                    self.dD.nl(ik, inn, out)[:, :, a, :],
                    self.A.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * np.einsum(
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

    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.ddD = Der2Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.D  = Dcov(data_K)

        if self.external_terms:
            self.A  = data_K.covariant('AA')
            self.dA = data_K.covariant('AA',gender=1)
            self.ddA = Der2A(data_K)
            self.ddO  = Der2O(data_K)
        self.ndim=3
        self.Iodd=False
        self.TRodd=True

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3,3),dtype=complex )
        if self.external_terms:
            summ += 0.5 * self.ddO.nn(ik,inn,out)

        for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
            if self.internal_terms:
                summ+= -1j*s*np.einsum("mlce,lncd->mncde",self.dD.nl(ik,inn,out)[:,:,a],self.dD.ln(ik,inn,out)[:,:,b])
                summ+= -1j*s*np.einsum("mlc,lncde->mncde",self.D.nl(ik,inn,out)[:,:,a],self.ddD.ln(ik,inn,out)[:,:,b])
                pass

            if self.external_terms:
                summ +=  -1 *s* np.einsum("mlce,lncd->mncde",self.dD.nl(ik,inn,out)[:,:,a], self.dA.ln(ik,inn,out)[:,:,b,:])
                summ +=  -1 *s* np.einsum("mlc,lncde->mncde",self.D.nl(ik,inn,out)[:,:,a], self.ddA.ln(ik,inn,out)[:,:,b,:])
                summ +=  -1 *s* np.einsum("mlcde,lnc->mncde",self.ddD.nl(ik,inn,out)[:,:,a,:],
                        self.A.ln (ik,inn,out)[:,:,b])
                summ +=  -1 *s* np.einsum("mlcd,lnce->mncde",self.dD.nl(ik,inn,out)[:,:,a,:],
                        self.dA.ln (ik,inn,out)[:,:,b])
                summ+=  -1j *s* np.einsum("mlce,lncd->mncde",self.dA.nn(ik,inn,out)[:,:,a], self.dA.nn(ik,inn,out)[:,:,b,:])
                summ+=  -1j *s* np.einsum("mlc,lncde->mncde",self.A.nn(ik,inn,out)[:,:,a], self.ddA.nn(ik,inn,out)[:,:,b,:])
                pass

        summ+=summ.swapaxes(0,1).conj()
        return summ


    def ln(self,ik,inn,out):
        raise NotImplementedError()


########################
#     spin moment      #
########################


class Spin(Matrix_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS')
        self.__dict__.update(s.__dict__)


class DerSpin(Matrix_GenDer_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS', gender=1)
        self.__dict__.update(s.__dict__)

class Der2Spin(Matrix_GenDer_ln):

    def __init__(self, data_K):
        super().__init__(data_K.covariant('SS', commader=1), data_K.covariant('SS', commader=2), Dcov(data_K))
        self.Iodd = False
        self.TRodd = True


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
        self.D = Dcov(data_K)
        self.E = data_K.E_K
        self.ndim = 1
        self.Iodd = False
        self.TRodd = True

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A] * self.E[ik][out][None, :, None],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.C.nn(ik, inn, out)
            summ += -1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.B.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.B.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * np.einsum(
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
        self.Iodd = False
        self.TRodd = True

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


class DerMorb(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = Dcov(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.E = data_K.E_K
        self.dO = DerOmega(data_K, **parameters)
        self.Omega = Omega(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB',gender=1)
            self.dH  = data_K.covariant('CC',gender=1)
        self.ndim=2
        self.Iodd=True
        self.TRodd=False
    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3),dtype=complex )
        if self.internal_terms:
            summ += -1j * np.einsum("mpc,pld,lnc->mncd",self.D.nl(ik,inn,out)[:,:,alpha_A],
                    self.V.ll(ik,inn,out),self.D.ln(ik,inn,out)[:,:,beta_A] )
            for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
                summ +=  -1j *s* np.einsum("mlc,lncd->mncd",self.D.nl(ik,inn,out)[:,:,a],
                        self.E[ik][out][:,None,None,None]*self.dD.ln(ik,inn,out)[:,:,b])

        if self.external_terms:
            summ += 0.5 * self.dH.nn(ik,inn,out)
            summ += -1j * np.einsum("mpc,pld,lnc->mncd",self.A.nn(ik,inn,out)[:,:,alpha_A],
                    self.V.nn(ik,inn,out),self.A.nn(ik,inn,out)[:,:,beta_A] )
            for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
                summ+=  -1j *s* np.einsum("mlc,lncd->mncd",self.A.nn(ik,inn,out)[:,:,a]*self.E[ik][inn][None,:,None],
                        self.dA.nn(ik,inn,out)[:,:,b,:])
                summ +=  -1 *s* np.einsum("mlc,lncd->mncd",self.D.nl (ik,inn,out)[:,:,a], self.dB.ln(ik,inn,out)[:,:,b,:])
                summ +=  -1 *s* np.einsum("mlc,lncd->mncd",(self.B.ln(ik,inn,out)[:,:,a]).transpose(1,0,2).conj(),
                        self.dD.ln (ik,inn,out)[:,:,b,:])

        summ += 0.5 * np.einsum("mlc,lnd->mncd",self.Omega.nn(ik,inn,out),self.V.nn(ik,inn,out) )
        summ += 0.5 * self.E[ik][inn][:,None,None,None]*self.dO.nn(ik,inn,out)

        summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False

##############################
###  second derivative of ####
###   orbital moment      ####
##############################

class Der2Morb(Formula_ln):
    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.ddD = Der2Dcov(data_K)
        self.dD = DerDcov(data_K)
        self.D  = Dcov(data_K)
        self.dV= InvMass(data_K)
        self.V = data_K.covariant('Ham',commader=1)
        self.E = data_K.E_K
        self.dO  = DerOmega(data_K,**parameters)
        self.ddO  = Der2Omega(data_K)
        self.Omega = Omega(data_K,**parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA',gender=1)
            self.ddA = Der2A(data_K)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB',gender=1)
            self.ddB = Der2B(data_K)
            self.dH  = data_K.covariant('CC',gender=1)
            self.ddH  = Der2H(data_K)
        self.ndim=3
        self.Iodd=False
        self.TRodd=True
    #TODO merge term if possible.
    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3,3),dtype=complex )
        if self.internal_terms:
            summ += -1j * np.einsum("mpc,plde,lnc->mncde",self.D.nl(ik,inn,out)[:,:,alpha_A],
                    self.dV.ll(ik,inn,out),self.D.ln(ik,inn,out)[:,:,beta_A] )
            for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
                summ += -0.5j *s*  np.einsum("mpce,pld,lnc->mncde",self.dD.nl(ik,inn,out)[:,:,a,:],
                        self.V.ll(ik,inn,out),self.D.ln(ik,inn,out)[:,:,b] )
                summ += -0.5j *s*  np.einsum("mpc,pld,lnce->mncde",self.D.nl(ik,inn,out)[:,:,a],
                        self.V.ll(ik,inn,out),self.dD.ln(ik,inn,out)[:,:,b,:] )
                summ+=  -1j *s* np.einsum("mlce,lncd->mncde",self.dD.nl(ik,inn,out)[:,:,a,:],
                    self.E[ik][out][:,None,None,None]*self.dD.ln(ik,inn,out)[:,:,b])
                summ+=  -1j *s* np.einsum("mlc,lncde->mncde",self.D.nl(ik,inn,out)[:,:,a],
                    self.E[ik][out][:,None,None,None,None]*self.ddD.ln(ik,inn,out)[:,:,b])
                summ+=  -1j *s* np.einsum("mpc,ple,lncd->mncde",self.D.nl(ik,inn,out)[:,:,a],
                    self.V.ll(ik,inn,out),self.dD.ln(ik,inn,out)[:,:,b])
        if self.external_terms:
            summ += 0.5 * self.ddH.nn(ik,inn,out)
            summ += -1j * np.einsum("mpc,plde,lnc->mncde",self.A.nn(ik,inn,out)[:,:,alpha_A],
                    self.dV.nn(ik,inn,out),self.A.nn(ik,inn,out)[:,:,beta_A] )
            for s,a,b in (+1,alpha_A,beta_A),(-1,beta_A,alpha_A):
                summ += -0.5j * np.einsum("mpce,pld,lnc->mncde",self.dA.nn(ik,inn,out)[:,:,a],
                        self.V.nn(ik,inn,out),self.A.nn(ik,inn,out)[:,:,b] )
                summ += -0.5j * np.einsum("mpc,pld,lnce->mncde",self.A.nn(ik,inn,out)[:,:,a],
                        self.V.nn(ik,inn,out),self.dA.nn(ik,inn,out)[:,:,b] )
                summ+=  -1j *s* np.einsum("mlce,lncd->mncde",
                        self.dA.nn(ik,inn,out)[:,:,a]*self.E[ik][inn][None,:,None,None],self.dA.nn(ik,inn,out)[:,:,b])
                summ+=  -1j *s* np.einsum("mlc,lncde->mncde",
                        self.A.nn(ik,inn,out)[:,:,a]*self.E[ik][inn][None,:,None],self.ddA.nn(ik,inn,out)[:,:,b])
                summ+=  -1j *s* np.einsum("mlc,ple,lncd->mncde",self.A.nn(ik,inn,out)[:,:,a],
                        self.V.nn(ik,inn,out),self.dA.nn(ik,inn,out)[:,:,b])
                summ +=  -1 *s* np.einsum("mlce,lncd->mncde",self.dD.nl (ik,inn,out)[:,:,a],
                        self.dB.ln(ik,inn,out)[:,:,b,:])
                summ +=  -1 *s* np.einsum("mlc,lncde->mncde",self.D.nl (ik,inn,out)[:,:,a],
                        self.ddB.ln(ik,inn,out)[:,:,b,:])
                summ +=  -1 *s* np.einsum("mlce,lncd->mncde",(self.dB.ln(ik,inn,out)[:,:,a]).transpose(1,0,2,3).conj(),
                        self.dD.ln (ik,inn,out)[:,:,b])
                summ +=  -1 *s* np.einsum("mlc,lncde->mncde",(self.B.ln(ik,inn,out)[:,:,a]).transpose(1,0,2).conj(),
                        self.ddD.ln (ik,inn,out)[:,:,b])

        summ += 0.5 * np.einsum("mlce,lnd->mncde",self.dO.nn(ik,inn,out),self.V.nn(ik,inn,out) )
        summ += 0.5 * np.einsum("mlc,lnde->mncde",self.Omega.nn(ik,inn,out),self.dV.nn(ik,inn,out) )
        summ += 0.5 * np.einsum("mle,lncd->mncde",self.V.nn(ik,inn,out),self.dO.nn(ik,inn,out) )
        summ += 0.5 * self.E[ik][inn][:,None,None,None,None]*self.ddO.nn(ik,inn,out)

        summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False

########################
#   spin transport     #
########################


def _spin_velocity_einsum_opt(C, A, B):
    # Optimized version of C += np.einsum('knls,klma->knmas', A, B). Used in shc_B_H.
    nk = C.shape[0]
    nw = C.shape[1]
    for ik in range(nk):
        # Performing C[ik] += np.einsum('nls,lma->nmas', A[ik], B[ik])
        tmp_a = np.swapaxes(A[ik], 1, 2)  # nls -> nsl
        tmp_a = np.reshape(tmp_a, (nw * 3, nw))  # nsl -> (ns)l
        tmp_b = np.reshape(B[ik], (nw, nw * 3))  # lma -> l(ma)
        tmp_c = tmp_a @ tmp_b  # (ns)l, l(ma) -> (ns)(ma)
        tmp_c = np.reshape(tmp_c, (nw, 3, nw, 3))  # (ns)(ma) -> nsma
        C[ik] += np.transpose(tmp_c, (0, 2, 3, 1))  # nsma -> nmas


def _J_H_qiao(data_K):
    # Spin current operator, J. Qiao et al PRB (2019)
    # J_H_qiao[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
    SS_H = data_K.Xbar('SS')
    SH_H = data_K.Xbar("SH")
    shc_K_H = -1j * data_K.Xbar("SR")
    _spin_velocity_einsum_opt(shc_K_H, SS_H, data_K.D_H)
    shc_L_H = -1j * data_K._R_to_k_H(data_K.SHR_R, hermitean=False)
    _spin_velocity_einsum_opt(shc_L_H, SH_H, data_K.D_H)
    J = (
        data_K.delE_K[:, None, :, :, None] * SS_H[:, :, :, None, :]
        + data_K.E_K[:, None, :, None, None] * shc_K_H[:, :, :, :, :] - shc_L_H)
    return (J + J.swapaxes(1, 2).conj()) / 2


def _J_H_ryoo(data_K):
    # Spin current operator, J. H. Ryoo et al PRB (2019)
    # J_H_ryoo[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
    SA_H = data_K.Xbar("SA")
    SHA_H = data_K.Xbar("SHA")
    J = -1j * (data_K.E_K[:, None, :, None, None] * SA_H - SHA_H)
    _spin_velocity_einsum_opt(J, data_K.Xbar('SS'), data_K.Xbar('Ham', 1))
    return (J + J.swapaxes(1, 2).conj()) / 2


class SpinVelocity(Matrix_ln):
    "spin current matrix elements. SpinVelocity.matrix[ik, m, n, a, s] = <u_mk|{v^a S^s}|u_nk> / 2"

    def __init__(self, data_K, spin_current_type):
        if spin_current_type == "qiao":
            # J. Qiao et al PRB (2018)
            super().__init__(_J_H_qiao(data_K))
        elif spin_current_type == "ryoo":
            # J. H. Ryoo et al PRB (2019)
            super().__init__(_J_H_ryoo(data_K))
        else:
            raise ValueError(f"spin_current_type must be qiao or ryoo, not {spin_current_type}")
        self.TRodd = False
        self.Iodd = True


class SpinOmega(Formula_ln):
    "spin Berry curvature"

    def __init__(self, data_K, spin_current_type="ryoo", **parameters):
        super().__init__(data_K, **parameters)
        self.A = data_K.covariant('AA')
        self.D = Dcov(data_K)
        self.J = SpinVelocity(data_K, spin_current_type)
        self.dEinv = DEinv_ln(data_K)
        self.ndim = 3
        self.TRodd = False
        self.Iodd = False

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)

        # v_over_de[l,n,b] = v[l,n,b] / (e[n] - e[l]) = D[l,n,b] - 1j * A[l,n,b]
        v_over_de = self.D.ln(ik, inn, out) - 1j * self.A.ln(ik, inn, out)

        # j_over_de[m,l,a,s] = j[m,l,a,s] / (e[m] - e[l])
        j_over_de = self.J.nl(ik, inn, out) * self.dEinv.nl(ik, inn, out)[:, :, None, None]

        summ += -2 * np.einsum("mlas,lnb->mnabs", j_over_de, v_over_de).imag

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


####################################
#                                  #
#    Some Products                 #
#                                  #
####################################

class VelOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Omega(data_K, **kwargs_formula)], name='VelOmega')

class VelOmegaVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Omega(data_K, **kwargs_formula),data_K.covariant('Ham', commader=1)], name='VelOmegaVel')


class VelHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='VelHplus')


class VelSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Spin(data_K)], name='VelSpin')


class VelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1)], name='VelVel')


class VelVelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1)], name='VelVelVel')


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


class MassSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), Spin(data_K)], name='MassSpin')

class OmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='OmegaOmega', additive=True)


class OmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='OmegaHplus')


class lmr_surf(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        formula = FormulaProduct ( [VelOmega(data_K,**kwargs_formula),data_K.covariant('Ham', commader=1)],
                name='VelOmegaVel')
        super().__init__([formula,
            DeltaProduct(delta_f,formula,'pu,MLabb->MLaup'),
            DeltaProduct(delta_f,formula,'au,MLpbb->MLaup')],
            [-1,1,1],['aup','aup','aup'], name='lmr_surf')


class lmr_sea(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        formula = FormulaProduct ( [InvMass(data_K),Omega(data_K,**kwargs_formula)], name='MassOmega')
        super().__init__([FormulaProduct(
            [data_K.covariant('Ham', commader=1), DerOmega(data_K,**kwargs_formula)], name='VelDerOmega'),
            formula,
            DeltaProduct(delta_f,formula,'pu,MLabb->MLaup'),
            DeltaProduct(delta_f,formula,'au,MLpbb->MLaup')],
            [-1,-1,1,1],['aup','apu','aup','aup'], name='lmr_sea')


class qmr_surf(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        formula = FormulaProduct ( [data_K.covariant('Ham', commader=1),
                data_K.covariant('Ham', commader=1),
                OmegaOmega(data_K,**kwargs_formula)],
                name='VelVelOmegaOmega')
        super().__init__([
            formula,
            DeltaProduct(delta_f,formula,'pv,MLabub->MLapuv'),
            DeltaProduct(delta_f,
                DeltaProduct(delta_f,formula,'pv,MLbcbc->MLpv'),
                'au,MLpv->MLapuv'),
            DeltaProduct(delta_f,formula,'au,MLpbvb->MLapuv')],
            [1,-1,1,-1],['apuv','apuv','apuv','apuv'], name='qmr_surf')


class MassOmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K),
                OmegaOmega(data_K,**kwargs_formula)],
                name='MassOmegaOmega')


class VelDerOmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Velocity(data_K),
                DerOmega(data_K,**kwargs_formula),
                Omega(data_K,**kwargs_formula)],
                name='VelDerOmegaOmega')

class ddo(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([
                DerOmega(data_K,**kwargs_formula),
                Velocity(data_K)],
                name='VelDerOmegaOmega')

class qmr_sea(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        formula1  = FormulaProduct ( [InvMass(data_K),OmegaOmega(data_K,**kwargs_formula)],
            name='mass-berryberry (apuv) (ab[pv]ub) (pb[au]vb) ([au][pv]bcbc)',additive = True)
        formula2  = FormulaProduct(
                [data_K.covariant('Ham', commader=1), DerOmega(data_K,**kwargs_formula),Omega(data_K,**kwargs_formula)],
                name='vel-derberry-berry (aupv) (auvp) (a[pv]ubb) (p[au]vbb) ([au][pv]bbcc)', additive=True)
        super().__init__([
            formula1, formula2, formula2,
            DeltaProduct(delta_f,formula1,'pv,MLabub->MLapuv'),
            DeltaProduct(delta_f,formula2,'pv,MLaubb->MLapuv'),
            DeltaProduct(delta_f,
                DeltaProduct(delta_f,formula1,'pv,MLbcbc->MLpv'),
                'au,MLpv->MLapuv'),
            DeltaProduct(delta_f,
                DeltaProduct(delta_f,formula2,'pv,MLbbcc->MLpv'),
                'au,MLpv->MLapuv'),
            DeltaProduct(delta_f,formula1,'au,MLpbvb->MLapuv'),
            DeltaProduct(delta_f,formula2,'au,MLpvbb->MLapuv')],
            [1,1,1,-1,-1,1,1,-1,-1],
            ['apuv','aupv','avpu','apuv','apuv','apuv','apuv','apuv','apuv'],
            name='qmr_sea', additive = False)


class nlhall_surf(DeltaProduct):

    def __init__(self, data_K, **kwargs_formula):
        formula  = FormulaProduct ( [OmegaOmega(data_K,**kwargs_formula),data_K.covariant('Ham', commader=1)],
        name='OmegaOmegaVel')
        formula_sum = FormulaSum([DeltaProduct(delta_f,formula,'pu,MLdbb->MLdpu'),formula],
            [1,-1],['dpu','dup'],)
        super().__init__(Levi_Civita, formula_sum, 'sda,MLdpu->MLapsu')


class nlhall_sea(DeltaProduct):

    def __init__(self, data_K, **kwargs_formula):
        formula = FormulaProduct ( [Omega(data_K,**kwargs_formula),DerOmega(data_K,**kwargs_formula)],
            name='OmegaDerOmega')
        formula_sum = FormulaSum([DeltaProduct(delta_f,formula,'pu,MLbdb->MLdpu'),formula,formula],
            [1,-1,-1],['dpu','pud','dup'],)
        super().__init__(Levi_Civita, formula_sum, 'sda,MLdpu->MLapsu')


class emcha_surf(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        velocity =  data_K.covariant('Ham',commader=1)
        formula1  = FormulaProduct ( [InvMass(data_K),Omega(data_K,**kwargs_formula),velocity],
            name='mass-berry-vel (apus)(psua) ([au]bpbs) ([us]abbp)')
        formula2  = FormulaProduct ( [velocity,DerOmega(data_K,**kwargs_formula),velocity],
            name='v-derberry-vel (aups) ([au]bbps) ([us]abpb)')
        tmp = FormulaSum([formula2,formula1],[1,1],['aups','apus'])
        super().__init__([tmp,
            DeltaProduct(delta_f,formula1,'us,MLabbp->MLaups'),
            DeltaProduct(delta_f,tmp,'au,MLbbps->MLaups'),
            DeltaProduct(delta_f,formula2,'us,MLabpb->MLaups'),
            formula2,
            DeltaProduct(Levi_Civita,
                DeltaProduct(Levi_Civita,tmp,'pta,MLxtbs->MLaxpbs'),
                'xub,MLaxpbs->MLaups')],
            [2,-2,-1,-1,1,-1],['aups','aups','aups','aups','aups','aups'])


class emcha_sea(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        velocity =  data_K.covariant('Ham',commader=1)
        formula1  = FormulaProduct ( [Der3E(data_K),Omega(data_K,**kwargs_formula)],
            name='Der3E-berry (apsu) ([au]bpsb)')
        formula2  = FormulaProduct ( [InvMass(data_K),DerOmega(data_K,**kwargs_formula)],
            name='mass-derberry (apus)(asup) ([au]bpbs)')
        formula3  = FormulaProduct ( [velocity,Der2Omega(data_K,**kwargs_formula)],
            name='vel-der2berry-vel (aups) ([au]bbps)')
        tmp = FormulaSum([formula3,formula1,formula2,formula2],[1,1,1,1],['aups','apsu','apus','asup'])
        super().__init__([tmp,
            DeltaProduct(delta_f,FormulaSum([formula2,formula1],[1,1],['aups','ausp']),'us,MLabbp->MLaups'),
            DeltaProduct(delta_f,tmp,'au,MLbbps->MLaups'),
            DeltaProduct(delta_f,formula2,'us,MLabbp->MLaups'),
            FormulaSum([formula3,formula2],[1,1],['aups','asup']),
            DeltaProduct(Levi_Civita,
                DeltaProduct(Levi_Civita,tmp,'pta,MLxtbs->MLaxpbs'),
                'xub,MLaxpbs->MLaups')],
            [2,-2,-1,-1,1,-1],['aups','aups','aups','aups','aups','aups'])


class MassOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), Omega(data_K, **kwargs_formula)], name='MassOmega')


class MassHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='MassHplus')


class MassHplusHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), Morb_Hpm(data_K, sign=+1, **kwargs_formula),
            Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='MassHplusHplus')


class MassHplusOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), Morb_Hpm(data_K, sign=+1, **kwargs_formula),
            Omega(data_K, **kwargs_formula)], name='MassHplusOmega')


class MassOmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='MassOmegaOmega')


class Der2OmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Der2Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='Der2OmegaOmega')


class Der2OmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Der2Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='Der2OmegaHplus')


class Der2HplusHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Der2Morb(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='Der2HplusHplus')


class Der2HplusOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Der2Morb(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='Der2HplusOmega')


class Der2SpinSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Der2Spin(data_K), Spin(data_K)], name='Der2SpinSpin')


class MassSpinSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), Spin(data_K), Spin(data_K)], name='MassSpinSpin')


class OmegaSpinSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Spin(data_K), Spin(data_K)], name='OmegaSpinSpin')


class OmegaOmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='OmegaOmegaOmega')


class OmegaOmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='OmegaOmegaHplus')


class OmegaHplusHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='OmegaHplusHplus')


class NLAHC_Z_spin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([DerOmega(data_K, **kwargs_formula), Spin(data_K)], name='DerOmegaSpin')


class DerOmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([DerOmega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='DerOmegaOmega')


class DerOmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([DerOmega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='DerOmegaHplus')


class NLDrude_Z_spin(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Spin(data_K)], name='Der3ESpin')
        term2 = FormulaProduct([Der2Spin(data_K), data_K.covariant('Ham', commader=1)], name='Der2SpinVel')
        super().__init__([term1, term2], [-1,1], ['apsu', 'uaps'])


class NLDrude_Z_orb_Hplus(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='Der3EHplus')
        term2 = FormulaProduct([Der2Morb(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='Der2HplusVel')
        super().__init__([term1, term2], [-1,1], ['apsu', 'uaps'])


class NLDrude_Z_orb_Omega(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([Der3E(data_K), Omega(data_K, **kwargs_formula)], name='Der3EOmega')
        term2 = FormulaProduct([Der2Omega(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='Der2OmegaVel')
        super().__init__([term1, term2], [-1,1], ['apsu', 'uaps'])


class lmr_spin_Z(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([lmr_sea(data_K, **kwargs_formula), Spin(data_K)], name='lmrSpin')
        term2 = FormulaProduct([DerSpin(data_K), Omega(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='DerSpinOmegaVel')
        term3 = DeltaProduct(delta_f, term2, 'pu,MLvabb->MLaupv'),
        term4 = DeltaProduct(delta_f, term2, 'au,MLvpbb->MLaupv'),
        super().__init__([term1, term2, term3, term4], [1,1,-1,-1], ['aupv', 'vaup', 'aupv', 'aupv'])


class lmr_orb_Z_Hplus(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([lmr_sea(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='lmrHplus')
        term2 = FormulaProduct([DerMorb(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='DerHplusOmegaVel')
        term3 = DeltaProduct(delta_f, term2, 'pu,MLvabb->MLaupv'),
        term4 = DeltaProduct(delta_f, term2, 'au,MLvpbb->MLaupv'),
        super().__init__([term1, term2, term3, term4], [1,1,-1,-1], ['aupv', 'vaup', 'aupv', 'aupv'])


class lmr_orb_Z_Omega(FormulaSum):

    def __init__(self, data_K, **kwargs_formula):
        term1 = FormulaProduct([lmr_sea(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='lmrOmega')
        term2 = FormulaProduct([DerOmega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula), data_K.covariant('Ham', commader=1)], name='DerOmegaOmegaVel')
        term3 = DeltaProduct(delta_f, term2, 'pu,MLvabb->MLaupv'),
        term4 = DeltaProduct(delta_f, term2, 'au,MLvpbb->MLaupv'),
        super().__init__([term1, term2, term3, term4], [1,1,-1,-1], ['aupv', 'vaup', 'aupv', 'aupv'])


class HC_Z_term1_spin(DeltaProduct):

    def __init__(self, data_K, **kwargs_formula):
        term = FormulaProduct([Der2Spin(data_K), InvMass(data_K)], name='Der2SpinMass')
        term_sum = FormulaSum([term, term], [1,1], ['vabps','vpsab'])
        super().__init__(Levi_Civita, term_sum, 'bup,MLvabps->MLasuv')


class HC_Z_term2_spin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        term = FormulaProduct([InvMass(data_K), InvMass(data_K), Spin(data_K)], name='MassMassSpin')
        super().__init__(Levi_Civita, term, 'bup,MLabpsv->MLasuv')


class HC_Z_term1_orb_Omega(DeltaProduct):

    def __init__(self, data_K, **kwargs_formula):
        term = FormulaProduct([Der2Omega(data_K, **kwargs_formula), InvMass(data_K)], name='Der2OmegaMass')
        term_sum = FormulaSum([term, term], [1,1], ['vabps','vpsab'])
        super().__init__(Levi_Civita, term_sum, 'bup,MLvabps->MLasuv')


class HC_Z_term1_orb_Hplus(DeltaProduct):

    def __init__(self, data_K, **kwargs_formula):
        term = FormulaProduct([Der2Morb(data_K, **kwargs_formula), InvMass(data_K)], name='Der2HplusMass')
        term_sum = FormulaSum([term, term], [1,1], ['vabps','vpsab'])
        super().__init__(Levi_Civita, term_sum, 'bup,MLvabps->MLasuv')


class HC_Z_term2_orb_Omega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        term = FormulaProduct([InvMass(data_K), InvMass(data_K), Omega(data_K, **kwargs_formula)], name='MassMassOmega')
        super().__init__(Levi_Civita, term, 'bup,MLabpsv->MLasuv')


class HC_Z_term2_orb_Hplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        term = FormulaProduct([InvMass(data_K), InvMass(data_K), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='MassMassHplus')
        super().__init__(Levi_Civita, term, 'bup,MLabpsv->MLasuv')
