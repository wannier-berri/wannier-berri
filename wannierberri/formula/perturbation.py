import numpy as np
from .formula import Formula_ln, Matrix_ln
from ..utility import alpha_A, beta_A
from ..symmetry.point_symmetry import transform_ident


class PertOmega(Formula_ln):
    " evaluates the change of Berry curvature in response to small perturbation"

    def __init__(self, data_K, pertS, pertScomma, **parameters):
        # pertScomma - the comma derivative of perturbation, with derivative being the first cartesian index
        super().__init__(data_K, **parameters)
        self.ndimPert = pertS.ndim - 3
        self.D = data_K.Dcov
        self.V = data_K.covariant('Ham', commader=1)
        self.S = Matrix_ln(pertS)
        self.Sc = Matrix_ln(pertScomma)
        dEi = data_K.dEig_inv
        self.T = Matrix_ln(-self.S.matrix * (dEi.reshape(dEi.shape + (1,) * (self.ndimPert))))
        self.Acal_int = Matrix_ln(-1j * data_K.D_H * dEi[:, :, :, None])
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.O = data_K.covariant('rotAA')
            self.Acal_ext = Matrix_ln(-data_K.Xbar('AA') * dEi[:, :, :, None])
        del dEi
        self.ndim = self.ndimPert + 1


    def nn(self, ik, inn, out):
        # This is the formula to be implemented:
        # orange terms
        iio = (ik, inn, out)
        indpert = "defghi"[:self.ndimPert]
        #  blue terms
        summ = np.zeros((len(inn), len(inn)) + (3,) * (self.ndimPert + 1), dtype=complex)
        summ_loc_1 = self.Sc.ln(*iio)
        summ_loc_1 += np.einsum(f"lpa,pn{indpert}->lna{indpert}", self.V.ll(*iio), self.T.ln(*iio))
        summ_loc_1 += np.einsum(f"lp{indpert},pna->lna{indpert}", self.S.ll(*iio), self.D.ln(*iio))
        summ_loc_1 -= np.einsum(f"lma,mn{indpert}->lna{indpert}", self.D.ln(*iio), self.S.nn(*iio))
        summ_loc_1 -= np.einsum(f"lm{indpert},mna->lna{indpert}", self.T.ln(*iio), self.V.nn(*iio))
        #  green terms
        if self.external_terms:
            summ_loc_2 = -np.einsum(f"lpa,pn{indpert}->lna{indpert}", self.A.ll(*iio), self.T.ln(*iio))
            summ_loc_2 += np.einsum(f"lm{indpert},mna->lna{indpert}", self.T.ln(*iio), self.A.nn(*iio))
            summ += 2 * np.einsum(f"mla,ln{indpert}->mna{indpert}", self.O.nl(*iio), self.T.ln(*iio))
            Acal_ext = self.Acal_ext.nl(*iio)
        Acal_int = self.Acal_int.nl(*iio)
        D = self.D.nl(*iio)
        for s, a, b in (+1., alpha_A, beta_A), (-1., beta_A, alpha_A):
            #  blue terms
            if self.internal_terms:
                summ += 2 * s * np.einsum(f"mla,lna{indpert}->mna{indpert}", Acal_int[:, :, a], summ_loc_1[:, :, b])
            if self.external_terms:
                summ += 2 * s * np.einsum(f"mla,lna{indpert}->mna{indpert}", Acal_ext[:, :, a], summ_loc_1[:, :, b])
                #  green terms
                summ += 2 * s * np.einsum(f"mla,lna{indpert}->mna{indpert}", D[:, :, a], summ_loc_2[:, :, b])
        return summ
    ## TODO : this is probably correct only for the traces, check for the full matrices

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class PertOmegaZeemanSpin(PertOmega):
    " evaluates the change of Berry curvature in response to small Zeeman B-field correction"

    def __init__(self, data_K, Bdirection=None, **parameters):
        pertS = data_K.Xbar('SS')
        pertScomma = data_K.covariant('SS', commader=1).matrix.transpose(0, 1, 2, 4, 3)  # transpose, because the derivative is the last index, but should be the first
        if Bdirection is not None:
            print("Bdirection is {}".format(Bdirection))
            Bdirection = np.array(Bdirection, dtype=float)
            Bdirection /= np.linalg.norm(Bdirection)
            pertS = pertS @ Bdirection
            pertScomma = pertScomma @ Bdirection
        super().__init__(data_K, pertS, pertScomma, **parameters)
        self.transformTR = transform_ident
        self.transformInv = transform_ident
