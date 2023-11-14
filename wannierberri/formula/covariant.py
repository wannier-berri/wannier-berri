import numpy as np
from ..__utility import alpha_A, beta_A
from . import Formula_ln, Matrix_ln, Matrix_GenDer_ln, FormulaProduct
from ..symmetry import transform_ident, transform_odd


class Identity(Formula_ln):

    def __init__(self,data_K=None):
        self.ndim = 0
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        return np.eye(len(inn))

    def ln(self, ik, inn, out):
        return np.zeros((len(out), len(inn)))


class Eavln(Matrix_ln):
    """ be careful : this is not a covariant matrix"""

    def __init__(self, data_K):
        super().__init__(0.5 * (data_K.E_K[:, :, None] + data_K.E_K[:, None, :]))
        self.ndim = 0
        self.transformTR=transform_ident
        self.transformInv=transform_ident


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


class DerDcov(Dcov):

    def __init__(self, data_K):
        self.W = data_K.covariant('Ham', commader=2)
        self.V = data_K.covariant('Ham', gender=1)
        self.D = data_K.Dcov
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        summ = self.W.ln(ik, inn, out)
        tmp = np.einsum("lpb,pnd->lnbd", self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        tmp = -np.einsum("lmb,mnd->lnbd", self.D.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None]
        return summ


class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=2), data_K.Dcov)
        self.transformTR=transform_ident
        self.transformInv=transform_ident


class DerWln(Matrix_GenDer_ln):
    r""" :math:`\overline{W}^{bc:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', 2), data_K.covariant('Ham', 3), data_K.Dcov)
        self.transformTR=transform_odd
        self.transformInv=transform_odd


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
        self.transformTR=transform_odd
        self.transformInv=transform_odd

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
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.O = data_K.covariant('OO')

        self.ndim = 1
        self.transformTR=transform_odd
        self.transformInv=transform_ident

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
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.dO = data_K.covariant('OO', gender=1)
        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_odd

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

class Hamiltonian(Matrix_ln):

    def __init__(self, data_K):
        v = data_K.covariant('Ham', gender=0)
        self.__dict__.update(v.__dict__)


class Velocity(Matrix_ln):

    def __init__(self, data_K, external_terms=False):
        v = data_K.covariant('Ham', gender=1)
        self.__dict__.update(v.__dict__)
        if external_terms:
            self.matrix = self.matrix+1j*data_K.Xbar('AA')*(data_K.E_K[:, :, None,None] - data_K.E_K[:, None, :,None])

class Spin(Matrix_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS')
        self.__dict__.update(s.__dict__)


class DerSpin(Matrix_GenDer_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS', gender=1)
        self.__dict__.update(s.__dict__)

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
        self.transformTR=transform_odd
        self.transformInv=transform_ident

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
        self.transformTR=transform_odd
        self.transformInv=transform_ident

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
        self.D = data_K.Dcov
        self.V = data_K.covariant('Ham', commader=1)
        self.E = data_K.E_K
        self.dO = DerOmega(data_K, **parameters)
        self.Omega = Omega(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB', gender=1)
            self.dH = data_K.covariant('CC', gender=1)
        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -2j * np.einsum(
                "mpc,pld,lnc->mncd",
                self.D.nl(ik, inn, out)[:, :, alpha_A], self.V.ll(ik, inn, out),
                self.D.ln(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.E[ik][out][:, None, None, None] * self.dD.ln(ik, inn, out)[:, :, b])
        if self.external_terms:
            summ += 1 * self.dH.nn(ik, inn, out)
            summ += -2j * np.einsum(
                "mpc,pld,lnc->mncd",
                self.A.nn(ik, inn, out)[:, :, alpha_A], self.V.nn(ik, inn, out),
                self.A.nn(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.A.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None],
                    self.dA.nn(ik, inn, out)[:, :, b, :])
                summ += -2 * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * np.einsum(
                    "mlc,lncd->mncd", (self.B.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2).conj(),
                    self.dD.ln(ik, inn, out)[:, :, b, :])

        summ += 1 * np.einsum("mlc,lnd->mncd", self.Omega.nn(ik, inn, out), self.V.nn(ik, inn, out))
        summ += 1 * self.E[ik][inn][:, None, None, None] * self.dO.nn(ik, inn, out)

        # Stepan: Shopuldn't we use the line below?
        # TODO: check this formula
        #summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self, ik, inn, out):
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




class SpinVelocity(Matrix_ln):
    "spin current matrix elements. SpinVelocity.matrix[ik, m, n, a, s] = <u_mk|{v^a S^s}|u_nk> / 2"

    def __init__(self, data_K, spin_current_type,external_terms=True):
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
        self.transformTR=transform_ident
        self.transformInv=transform_odd

    def _J_H_simple(self, data_K, external_terms=True):
        # Spin current operator, J. Qiao et al PRB (2019)
        # J_H[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        S = data_K.Xbar('SS')
        V = Velocity(data_K, external_terms=external_terms).matrix
        J = np.einsum("klms,kmna->klnas", S, V)
        return (J + J.swapaxes(1, 2).conj()) / 2


    def _J_H_qiao(self, data_K, external_terms=True):
        if not external_terms :
            raise NotImplementedError("spin Hall qiao without external terms is not implemented yet. Use `SHC_type='simple'`")
        # Spin current operator, J. Qiao et al PRB (2019)
        # J_H_qiao[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        SS_H = data_K.Xbar('SS')
        SH_H = data_K.Xbar("SH")
        shc_K_H = -1j * data_K.Xbar("SR")
        _spin_velocity_einsum_opt(shc_K_H, SS_H, data_K.D_H)
        shc_L_H = -1j * data_K._R_to_k_H(data_K.get_R_mat('SHR'), hermitean=False)
        _spin_velocity_einsum_opt(shc_L_H, SH_H, data_K.D_H)
        J = (
            data_K.delE_K[:, None, :, :, None] * SS_H[:, :, :, None, :]
            + data_K.E_K[:, None, :, None, None] * shc_K_H[:, :, :, :, :] - shc_L_H)
        return (J + J.swapaxes(1, 2).conj()) / 2


    def _J_H_ryoo(self,data_K,external_terms=True):
        if not external_terms :
            raise NotImplementedError("spin Hall ryoo without external terms is not implemented yet. Use `SHC_type='simple'`")
        # Spin current operator, J. H. Ryoo et al PRB (2019)
        # J_H_ryoo[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
        SA_H = data_K.Xbar("SA")
        SHA_H = data_K.Xbar("SHA")
        J = -1j * (data_K.E_K[:, None, :, None, None] * SA_H - SHA_H)
        _spin_velocity_einsum_opt(J, data_K.Xbar('SS'), data_K.Xbar('Ham', 1))
        return (J + J.swapaxes(1, 2).conj()) / 2



class SpinOmega(Formula_ln):
    "spin Berry curvature"

    def __init__(self, data_K, spin_current_type="ryoo", **parameters):
        super().__init__(data_K, **parameters)
        self.A = data_K.covariant('AA')
        self.D = data_K.Dcov
        self.J = SpinVelocity(data_K, spin_current_type)
        self.dEinv = DEinv_ln(data_K)
        self.ndim = 3
        self.transformTR=transform_ident
        self.transformInv=transform_ident

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




class PertOmega(Formula_ln):
    " evaluates the change of Berry curvature in response to small perturbation"

    def __init__(self, data_K, pertS, pertScomma):
        # pertScomma - the comma derivative of perturbation, with derivative being the first cartesian index
        self.ndimPert  = pertS.ndim - 3 
        print ( "perturbation is {} dimensiona; ".format(self.ndimPert))
        A    = data_K.Xbar('AA')
        D    = data_K.D_H
        V    = data_K.covariant('Ham',commader=1).matrix
        O    = data_K.covariant('OO').matrix
        S    = pertS
        Sc   = pertScomma
        dEi  = data_K.dEig_inv
        T    = -S*( dEi.reshape(dEi.shape+(1,)*(self.ndimPert) ))
        Acal= (-(A+1j*D)*data_K.dEig_inv[:,:,:,None])
        del dEi

        # Now add the dimentsions of the perturbation
#        newshape=A.shape+(1,)*self.ndimPert
#        for var in 'A','D','Acal','V','O':
#            locals()[var].shape  = newshape

        # now define the "alpha" and "beta" components
        A_,D_,V_,Acal_,Sc_={},{},{},{},{}
        for var in 'A','D','V','Acal','Sc':
            for c in 'alpha','beta':
#                print (var,c,locals()[var].shape)
                locals()[var+"_"][c]=locals()[var][:,:,:,globals()[c+'_A']]

        self.A_=A_
        self.D_=D_
        self.V_=V_
        self.Acal_=Acal_
        self.Sc_=Sc_
        self.S=S
        self.O=O
        self.T=T
        self.ndim=self.ndimPert+1


    def nn(self, ik, inn, out):
        # This is the formula to be implemented:
        # orange terms

        indpert="defghi"[:self.ndimPert]
#        formula =  TraceFormula ( [ ],ndim=1+ndimPert,TRodd=False,Iodd=True)
        summ = np.zeros((len(inn), len(inn)) +(3,)*self.ndim, dtype=complex)
#        formula.add_term  ( ('nl,ln', 2*O , T ) )
        summ +=2*np.einsum(f"mla,ln{indpert}->mna{indpert}", self.O[ik,inn][:,out],self.T[ik,out][:,inn])
        for s,a,b in ( +1.,'alpha','beta'),(-1.,'beta','alpha'):
            #  blue terms
#            formula.add_term( ('nl,ln',    Acal_ [a] ,  2*Sc_[b]*s        ) )
            summ += 2*s*np.einsum(f"mla,lna{indpert}->mna{indpert}", self.Acal_ [a][ik,inn][:,out], self.Sc_[b][ik,out][:,inn] )
#            formula.add_term( ('nl,lp,pn', Acal_ [a] ,  2*V_[b] *s, T     ) )
            summ += 2*s*np.einsum(f"mla,lpa,pn{indpert}->mna{indpert}", self.Acal_ [a][ik,inn][:,out], self.V_[b][ik,out][:,out], self.T[ik,out][:,inn] )
#            formula.add_term( ('nl,lp,pn', Acal_ [a] ,  2*S     *s, D_[b] ) )
            summ += 2*s*np.einsum(f"mla,lp{indpert},pna->mna{indpert}", self.Acal_ [a][ik,inn][:,out], self.S[ik,out][:,out], self.D_[b][ik,out][:,inn] )
#            formula.add_term( ('nl,lm,mn', Acal_ [a] , -2*D_[b] *s, S     ) )
            summ -= 2*s*np.einsum(f"mla,loa,on{indpert}->mna{indpert}", self.Acal_ [a][ik,inn][:,out], self.D_[b][ik,out][:,inn], self.S[ik,inn][:,inn] )
#            formula.add_term( ('nl,lm,mn', Acal_ [a] , -2*T     *s, V_[b] ) )
            summ -= 2*s*np.einsum(f"mla,lo{indpert},ona->mna{indpert}", self.Acal_ [a][ik,inn][:,out], self.T[ik,out][:,inn], self.V_[b][ik,inn][:,inn] )
            #  green terms
#            formula.add_term( ('nl,lp,pn',    D_ [a] , -2*A_[b] *s, T     ) )
            summ -= 2*s*np.einsum(f"mla,lpa,pn{indpert}->mna{indpert}", self.D_[a][ik,inn][:,out], self.A_[b][ik,out][:,out], self.T[ik,out][:,inn] )
#            formula.add_term( ('nl,lm,mn',    D_ [a] ,  2*T     *s, A_[b] ) )
            summ += 2*s*np.einsum(f"mla,lo{indpert},ona->mna{indpert}", self.D_[a][ik,inn][:,out], self.T[ik,out][:,inn], self.A_[b][ik,inn][:,inn] )
        return summ

#    TODO:
#    TODO: I think it is correct only for the trace
#    TODO: more generally, the facxtor of 2 should be changed to adding a hermitean conjugate, right?


    def ln(self, ik, inn, out):
        raise NotImplementedError()




class PertOmega2(Formula_ln):
    " evaluates the change of Berry curvature in response to small perturbation"

    def __init__(self, data_K, pertS, pertScomma):
        # pertScomma - the comma derivative of perturbation, with derivative being the first cartesian index
        self.ndimPert  = pertS.ndim - 3 
        print ( "perturbation is {} dimensiona; ".format(self.ndimPert))
        self.A    = data_K.covariant('AA')
        self.D    = data_K.Dcov
        self.V    = data_K.covariant('Ham',commader=1)
        self.O    = data_K.covariant('OO')
        self.S    = Matrix_ln(pertS)
        self.Sc   = Matrix_ln(pertScomma)
        dEi  = data_K.dEig_inv
        self.T    = Matrix_ln(-self.S.matrix*( dEi.reshape(dEi.shape+(1,)*(self.ndimPert) )) )
        self.Acal= Matrix_ln(-(data_K.Xbar('AA')+1j*data_K.D_H)*data_K.dEig_inv[:,:,:,None])
        del dEi

        self.ndim=self.ndimPert+1


    def nn(self, ik, inn, out):
        # This is the formula to be implemented:
        # orange terms
        iio = (ik,inn,out)
        indpert="defghi"[:self.ndimPert]
            #  blue terms
        summ_loc_1 = self.Sc.ln(*iio)
        summ_loc_1 += np.einsum(f"lpa,pn{indpert}->lna{indpert}", self.V.ll(*iio), self.T.ln(*iio) )
        summ_loc_1 += np.einsum(f"lp{indpert},pna->lna{indpert}", self.S.ll(*iio), self.D.ln(*iio) )
        summ_loc_1 -= np.einsum(f"lma,mn{indpert}->lna{indpert}", self.D.ln(*iio), self.S.nn(*iio) )
        summ_loc_1 -= np.einsum(f"lm{indpert},mna->lna{indpert}", self.T.ln(*iio), self.V.nn(*iio) )
        #  green terms
        summ_loc_2 = -np.einsum(f"lpa,pn{indpert}->lna{indpert}", self.A.ll(*iio), self.T.ln(*iio) )
        summ_loc_2 += np.einsum(f"lm{indpert},mna->lna{indpert}", self.T.ln(*iio), self.A.nn(*iio) )

        summ = 2*np.einsum(f"mla,ln{indpert}->mna{indpert}", self.O.nl(*iio),self.T.ln(*iio))
        Acal=self.Acal.nl(*iio)
        D=self.D.nl(*iio)
        for s,a,b in ( +1.,alpha_A,beta_A),(-1.,beta_A,alpha_A):
            #  blue terms
            summ += 2*s*np.einsum(f"mla,lna{indpert}->mna{indpert}", Acal[:,:,a], summ_loc_1[:,:,b] )
            #  green terms
            summ += 2*s*np.einsum(f"mla,lna{indpert}->mna{indpert}", D[:,:,a], summ_loc_2[:,:,b] )
        return summ


    def ln(self, ik, inn, out):
        raise NotImplementedError()


class PertOmegaZeemanSpin(PertOmega):
    " evaluates the change of Berry curvature in response to small Zeeman B-field correction"

    def __init__(self,data_K,Bdirection=None):
        pertS     = data_K.Xbar('SS')
        pertScomma = data_K.covariant('SS',commader=1).matrix.transpose(0,1,2,4,3)  # transpose, because the derivative is the last index, but should be the first
        if Bdirection is not None:
            print ("Bdirection is {}".format(Bdirection))
            Bdirection=np.array(Bdirection,dtype=float)
            Bdirection/=np.linalg.norm(Bdirection)
            pertS =  pertS @ Bdirection
            pertScomma =  pertScomma @ Bdirection
        super().__init__(data_K,pertS,pertScomma)
        self.transformTR=transform_ident
        self.transformInv=transform_ident


class PertOmegaZeemanSpin2(PertOmega2):
    " evaluates the change of Berry curvature in response to small Zeeman B-field correction"

    def __init__(self,data_K,Bdirection=None):
        pertS     = data_K.Xbar('SS')
        pertScomma = data_K.covariant('SS',commader=1).matrix.transpose(0,1,2,4,3)  # transpose, because the derivative is the last index, but should be the first
        if Bdirection is not None:
            print ("Bdirection is {}".format(Bdirection))
            Bdirection=np.array(Bdirection,dtype=float)
            Bdirection/=np.linalg.norm(Bdirection)
            pertS =  pertS @ Bdirection
            pertScomma =  pertScomma @ Bdirection
        super().__init__(data_K,pertS,pertScomma)
        self.transformTR=transform_ident
        self.transformInv=transform_ident


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


class OmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='OmegaOmega')


class OmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='OmegaHplus')




