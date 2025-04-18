import numpy as np
import abc
"""some basic classes to construct formulae for evaluation"""
from ..symmetry.point_symmetry import TransformProduct


class Formula(abc.ABC):
    @abc.abstractmethod
    def __init__(self, data_K=None, internal_terms=True, cross_terms=True, external_terms=True,
                 transformTR=None, transformInv=None, ndim=0):
        self.internal_terms = internal_terms
        self.external_terms = external_terms
        self.cross_terms = cross_terms
        if data_K is not None:
            if data_K.force_internal_terms_only:
                self.external_terms = False
                self.cross_terms = False
        self.ndim = ndim
        self.transformTR = transformTR
        self.transformInv = transformInv


class Formula_ln(Formula):

    """
    A "Formula" is a ground concept of our calculation. Assume that we have divided
    all our Wannierised states into two subspaces, called "inn" and "out".
    We call a matrix "covariant" if it is covariant under gauge transformations
    inside the "inn" or "out" subspaces but do not intermix them.
    A `Formula_ln` object has methods that returns submatrices of the covariant matrix.
    """

    @abc.abstractmethod
    def __init__(self, data_K=None, **parameters):
        super().__init__(data_K, **parameters)

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

    def __init__(self, matrix, **kwargs):
        super().__init__(ndim=len(matrix.shape) - 3, **kwargs)
        self.matrix = matrix

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



class FormulaSum(Formula_ln):
    """a class to store a sum of several formulae
    Parameters
    ----------
    formula_list: list
        list of formulas
    index_list: list of string
        Index of formulas.
        All formulas will transpose to index of first formula in the list before sum together.

    return an array with same index with first formula.
    """

    def __init__(self, formula_list, sign, index_list, name="unknown", additive=True):
        if type(formula_list) not in (list, tuple):
            formula_list = [formula_list]
        assert len(formula_list) > 0, 'formula_list is empty'
        TRodd_list = [f.transformTR.factor for f in formula_list]
        Iodd_list = [f.transformInv.factor for f in formula_list]
        # assert only works for same transform_ident or transform_odd
        assert len(set(TRodd_list)) == 1, 'formula in formula_list have different TRodd'
        assert len(set(Iodd_list)) == 1, 'formula in formula_list have different Iodd'
        self.transformTR = formula_list[0].transformTR
        self.transformInv = formula_list[0].transformInv
        self.name = name
        self.formulae = formula_list
        self.index = index_list
        self.sign = sign
        self.ndim = formula_list[0].ndim
        self.additive = additive

    def nn(self, ik, inn, out):
        matrices = [frml.nn(ik, inn, out) for frml in self.formulae]
        res = self.sign[0] * matrices[0]
        for mat, sign, index in zip(matrices[1:], self.sign[1:], self.index[1:]):
            res += sign * np.einsum('MN' + index + '->MN' + self.index[0], mat)
        return np.array(res, dtype=complex)

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    def additive(self):
        return self.additive


class DeltaProduct(Formula_ln):
    """a class to store a product of formulae and delta function"""

    def __init__(self, delta_f, formula, einsumstr):
        self.formula = formula
        self.delta_f = delta_f
        self.transformTR = self.formula.transformTR
        self.transformInv = self.formula.transformInv
        self.ndim = len(einsumstr.split('->')[1]) - 2
        self.einsumstr = einsumstr

    def nn(self, ik, inn, out):
        matrix = self.formula.nn(ik, inn, out)
        res = np.einsum(self.einsumstr, self.delta_f, matrix)
        return np.array(res, dtype=complex)

    def ln(self, ik, inn, out):
        raise NotImplementedError()
