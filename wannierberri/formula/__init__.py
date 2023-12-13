import numpy as np
import abc
"""some basic classes to construct formulae for evaluation"""
from ..symmetry import transform_ident, transform_odd, TransformProduct


class Formula_ln(abc.ABC):

    """
    A "Formula" is a ground concept of our calculation. Assume that we have divided
    all our Wannierised states into two subspaces, called "inn" and "out".
    We call a matrix "covariant" if it is covariant under gauge transformations
    inside the "inn" or "out" subspaces but do not intermix them.
    A `Formula_ln` object has methods that returns submatrices of the covariant matrix.
    """

    @abc.abstractmethod
    def __init__(self, data_K, internal_terms=True, external_terms=True, correction_wcc=False, dT_wcc=False):
        self.internal_terms = internal_terms
        self.external_terms = external_terms
        self.correction_wcc = correction_wcc
        if self.correction_wcc:
            if not (self.external_terms and self.internal_terms):
                raise ValueError(
                    f"correction_wcc makes sense only with all terms, but called with "
                    f"internal:{self.internal_terms}"
                    f"external:{self.external_terms}")
            self.T_wcc = data_K.covariant('T_wcc')
            if dT_wcc:
                self.dT_wcc = data_K.covariant('T_wcc', gender=1)

    @abc.abstractmethod
    def ln(self, ik, inn, out):
        r"""Returns the submatrix :math:`X_{ln}` at point `ik`, where
        :math:`l \in \mathrm{out}` and :math:`n \in \mathrm{inn}`
        """

    @abc.abstractmethod
    def nn(self, ik, inn, out):
        r"""Returns the submatrix :math:`X_{nn'}` at point `ik`, where
        :math:`n, n' \in \mathrm{inn}`
        """

    def nl(self, ik, inn, out):
        r"""Returns the submatrix :math:`X_{nl}` at point `ik`, where
        :math:`l \in \mathrm{out}` and :math:`n \in \mathrm{inn}`
        """
        return self.ln(ik, out, inn)

    def ll(self, ik, inn, out):
        r"""Returns the submatrix :math:`X_{ll'}` at point `ik`, whee
        :math:`l, l' \in \mathrm{out}`
        """
        return self.nn(ik, out, inn)

    @property
    def additive(self):
        """ if Trace_A+Trace_B = Trace_{A+B} holds.
        needs override for quantities that do not obey this rule (e.g. Orbital magnetization)
        """
        return True

    def trace(self, ik, inn, out):
        "Returns a trace over the `inn` states"
        return np.einsum("nn...->...", self.nn(ik, inn, out)).real


class Matrix_ln(Formula_ln):
    "anything that can be called just as elements of a matrix"

    def __init__(self, matrix, transformTR=None, transformInv=None):
        self.matrix = matrix
        self.ndim = len(matrix.shape) - 3
        if transformTR is not None:
            self.transformTR = transformTR
        if transformInv is not None:
            self.transformInv = transformInv

    def ln(self, ik, inn, out):
        return self.matrix[ik][out][:, inn]

    def nn(self, ik, inn, out):
        return self.matrix[ik][inn][:, inn]


class Matrix_GenDer_ln(Formula_ln):
    "generalized erivative of MAtrix_ln"

    def __init__(self, matrix, matrix_comader, D, transformTR=None, transformInv=None):
        self.A = matrix
        self.dA = matrix_comader
        self.D = D
        self.ndim = matrix.ndim + 1
        if transformTR is not None:
            self.transformTR = transformTR
        if transformInv is not None:
            self.transformInv = transformInv

    def nn(self, ik, inn, out):
        summ = self.dA.nn(ik, inn, out)
        summ -= np.einsum("mld,lnb...->mnb...d", self.D.nl(ik, inn, out), self.A.ln(ik, inn, out))
        summ += np.einsum("mlb...,lnd->mnb...d", self.A.nl(ik, inn, out), self.D.ln(ik, inn, out))
        return summ

    def ln(self, ik, inn, out):
        summ = self.dA.ln(ik, inn, out)
        summ -= np.einsum("mld,lnb...->mnb...d", self.D.ln(ik, inn, out), self.A.nn(ik, inn, out))
        summ += np.einsum("mlb...,lnd->mnb...d", self.A.ll(ik, inn, out), self.D.ln(ik, inn, out))
        return summ


class FormulaProduct(Formula_ln):
    """a class to store a product of several formulae"""

    def __init__(self, formula_list, name="unknown", hermitian=False):
        if type(formula_list) not in (list, tuple):
            formula_list = [formula_list]
        self.transformTR = TransformProduct(f.transformTR for f in formula_list)
        self.transformInv = TransformProduct(f.transformInv for f in formula_list)
        self.name = name
        self.formulae = formula_list
        self.hermitian = hermitian
        ndim_list = [f.ndim for f in formula_list]
        self.ndim = sum(ndim_list)
        self.einsumlines = []
        letters = "abcdefghijklmnopqrstuvw"
        dim = ndim_list[0]
        for d in ndim_list[1:]:
            self.einsumlines.append("LM" + letters[:dim] + ",MN" + letters[dim:dim + d] + "->LN" + letters[:dim + d])
            dim += d

    def nn(self, ik, inn, out):
        matrices = [frml.nn(ik, inn, out) for frml in self.formulae]
        res = matrices[0]
        for mat, line in zip(matrices[1:], self.einsumlines):
            res = np.einsum(line, res, mat)
        if self.hermitian:
            res = 0.5 * (res + res.swapaxes(0, 1).conj())
        return np.array(res, dtype=complex)

    def ln(self, ik, inn, out):
        raise NotImplementedError()

from . import covariant
from . import covariant_basic
