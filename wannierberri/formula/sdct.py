import numpy as np
from ..formula import Formula
from ..symmetry.point_symmetry import transform_odd, transform_odd_trans_102


class Formula_SDCT(Formula):

    """
    Parameters
    ----------
    sym : bool
        if True - calculates the symmetric part of the SDCT, if False - calculates the antisymmetric part of the SDCT  """

    has_terms = ["M1", "E2", "V", "S"]

    def __init__(self, data_K, sym, nbandind=2, **kwargs):
        super().__init__(data_K, **kwargs)
        self.ndim = 3
        self.transformInv = transform_odd
        self.transformTR = transform_odd_trans_102
        self.summ = np.zeros((data_K.nk,) + (data_K.num_wann,) * nbandind + (3, 3, 3), dtype=complex)
        self.sym = sym

    def symsumm(self):
        if self.sym:
            self.summ = np.real(self.summ + self.summ.swapaxes(3, 4))
        else:
            self.summ = -np.imag(self.summ - self.summ.swapaxes(3, 4))

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)

############################ Sea I terms ############################


class Formula_SDCT_sea_I(Formula_SDCT):

    has_terms = ["M1", "E2", "V", "S"]

    def __init__(self, data_K, sym,
                 M1_terms=True, E2_terms=True, V_terms=True, S_terms=False,
                 degen_thresh=1e-3, **parameters):
        super().__init__(sym=sym, data_K=data_K, **parameters)
        # Intrinsic multipole moments
        A = data_K.get_E1(external_terms=self.external_terms, degen_thresh=degen_thresh)

        # Other quantities
        Vn = data_K.delE_K
        assert np.max(np.abs(Vn.imag)) < 1e-8, f"Vn should be real but has imaginary part of {np.max(np.abs(Vn.imag))}"
        Vnm_plus = 0.5 * (Vn[:, :, None, :] + Vn[:, None, :, :])

        # --- Formula --- #
        if M1_terms or S_terms:
            B_M1 = data_K.get_Bln_m(external_terms=self.external_terms, orb=M1_terms, spin=S_terms, key_OO=self.key_OO, degen_thresh=degen_thresh)
            self.summ += A[:, :, :, :, None, None] * B_M1.swapaxes(1, 2)[:, :, :, None, :, :]

        if E2_terms:
            B_E2 = data_K.get_Bln_q(external_terms=self.external_terms, degen_thresh=degen_thresh)
            self.summ += A[:, :, :, :, None, None] * B_E2.swapaxes(1, 2)[:, :, :, None, :, :]

        if V_terms:
            # This is weird, why we have to put a minus sign here to get the correct term for the symmetric part of the SDCT?
            sign_V_term = -1 if sym else 1
            self.summ += sign_V_term * (-Vnm_plus[:, :, :, :, None, None] * A[:, :, :, None, :, None] * A.swapaxes(1, 2)[:, :, :, None, None, :])
        self.symsumm()


class Formula_SDCT_sea_II(Formula_SDCT):

    has_terms = ["V"]

    def __init__(self, data_K, sym, M1_terms=True, E2_terms=True, V_terms=True, S_terms=False, **parameters):
        super().__init__(sym=sym, data_K=data_K, **parameters)
        # --- Formula --- #
        if V_terms:
            # Intrinsic multipole moments
            A = data_K.get_E1(external_terms=self.external_terms, degen_thresh=1e-3)

            # Other quantities
            Vn = data_K.delE_K
            Vnm_plus = 0.5 * (Vn[:, :, None, :] + Vn[:, None, :, :])
            self.summ -= A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None] * Vnm_plus[:, :, :, None, None, :]
            self.symsumm()


############################## Surface  terms ############################


class Formula_SDCT_surf_I(Formula_SDCT):

    has_terms = ["V"]

    def __init__(self, data_K, sym, M1_terms=True, E2_terms=True, V_terms=True, S_terms=False, degen_thresh=1e-3, **parameters):
        super().__init__(data_K, sym=sym, **parameters)
        # --- Formula --- #
        if V_terms:
            # Intrinsic multipole moments
            A = data_K.get_E1(external_terms=self.external_terms, degen_thresh=degen_thresh)
            Vn = data_K.delE_K
            self.summ += (A[:, :, :, :, None, None] * A.swapaxes(1, 2)[:, :, :, None, :, None]) * Vn[:, :, None, None, None, :]
        if sym:
            self.summ = np.real(self.summ)
        else:
            self.summ = -np.imag(self.summ)

    def trace_ln(self, ik, inn1, inn2):
        return self.summ[ik, inn1].sum(axis=0)[inn2].sum(axis=0)


class Formula_SDCT_surf_II(Formula_SDCT):

    has_terms = ["M1", "S", "V"]

    def __init__(self, data_K, sym,
                 M1_terms=True, E2_terms=True, V_terms=True, S_terms=False, degen_thresh=1e-3, **parameters):
        super().__init__(data_K, sym=sym, nbandind=1, **parameters)
        Vn = data_K.delE_K
        if sym:
            if V_terms:
                self.summ += Vn[:, :, :, None, None] * Vn[:, :, None, :, None] * Vn[:, :, None, None, :]
        else:
            if M1_terms or S_terms:
                # Intrinsic multipole moments
                B_M1 = data_K.get_Bln_m(external_terms=self.external_terms, orb=M1_terms, spin=S_terms, key_OO=self.key_OO, degen_thresh=degen_thresh)
                Bn_M1 = np.diagonal(B_M1, axis1=1, axis2=2).transpose(0, 3, 1, 2)
                # --- Formula --- #
                self.summ += Vn[:, :, :, None, None] * Bn_M1[:, :, None, :, :]
                self.summ = self.summ - self.summ.swapaxes(3, 4)

    def trace_ln(self, ik, inn1, inn2):  # There is no sum over l
        return self.summ[ik, inn1].sum(axis=0)
