import numpy as np
from ..utility import orthogonalize
from .utility import get_inverse_block, rotate_block_matrix
from .sawf import SymmetrizerSAWF, VoidSymmetrizer


def get_symmetrizer_Uirr(symmetrizer, ikirr):
    if isinstance(symmetrizer, VoidSymmetrizer):
        return VoidSymmetrizer()
    else:
        return Symmetrizer_Uirr(symmetrizer, ikirr)


def get_symmetrizer_Zirr(symmetrizer, ikirr, free=None):
    if free is None:
        free = np.ones(symmetrizer.NB, dtype=bool)
    if isinstance(symmetrizer, VoidSymmetrizer):
        return VoidSymmetrizer()
    else:
        return Symmetrizer_Zirr(symmetrizer, ikirr, free=free)


class Symmetrizer_Uirr(SymmetrizerSAWF):

    def __init__(self, symmetrizer, ikirr):
        self.ikirr = ikirr
        self.isym_little = symmetrizer.isym_little[ikirr]
        self.nsym_little = len(self.isym_little)
        self.ikpt = symmetrizer.kptirr[ikirr]
        self.d_indices = symmetrizer.d_band_block_indices[ikirr]
        self.D_indices = symmetrizer.D_wann_block_indices
        self.d_band_blocks = symmetrizer.d_band_blocks[ikirr]
        self.D_wann_blocks_inverse = symmetrizer.D_wann_blocks_inverse[ikirr]
        self.nb = symmetrizer.NB
        self.num_wann = symmetrizer.num_wann
        self.time_reversals = symmetrizer.time_reversals



    def rotate_U(self, U, isym):
        # forward = not forward
        Uloc = U.copy()
        if self.time_reversals[isym]:
            Uloc = Uloc.conj()
        Uloc = rotate_block_matrix(Uloc,
                                   lblocks=self.d_band_blocks[isym],
                                   lindices=self.d_indices,
                                   rblocks=self.D_wann_blocks_inverse[isym],
                                   rindices=self.D_indices)
        return Uloc



    def __call__(self, U):
        Usym = sum(self.rotate_U(U, isym) for isym in self.isym_little) / self.nsym_little
        return orthogonalize(Usym)


class Symmetrizer_Zirr(SymmetrizerSAWF):

    def __init__(self, symmetrizer, ikirr, free):
        self.ikirr = ikirr
        self.isym_little = symmetrizer.isym_little[ikirr]
        self.nsym_little = len(self.isym_little)
        self.ikpt = symmetrizer.kptirr[ikirr]
        self.nb = symmetrizer.NB
        self.num_wann = symmetrizer.num_wann
        self.time_reversals = symmetrizer.time_reversals

        if free is not None:
            (
                d_band_block_indices_free,
                d_band_blocks_free
            ) = symmetrizer.select_bands_in_blocks(symmetrizer.d_band_blocks[ikirr], symmetrizer.d_band_block_indices[ikirr], free)
            d_band_blocks_free_inverse = get_inverse_block(d_band_blocks_free)

            self.lblocks = d_band_blocks_free_inverse
            self.rblocks = d_band_blocks_free
            self.indices = d_band_block_indices_free
        else:
            self.lblocks = symmetrizer.d_band_blocks_inverse,
            self.rblocks = symmetrizer.d_band_blocks
            self.indices = symmetrizer.d_band_block_indices


    def __call__(self, Z):
        # return Z # temporary for testing
        if Z.shape[0] == 0:
            return Z
        else:
            Z_rotated = [self.rotate_Z(Z, isym) for isym in self.isym_little]
            Z[:] = sum(Z_rotated) / self.nsym_little
            return Z

    def rotate_Z(self, Z, isym):
        """
        Rotates the zmat matrix at the irreducible kpoint
        Z = d_band^+ @ Z @ d_band
        """
        Zloc = Z.copy()
        # if self.time_reversals[isym]:
        #     Zloc = Zloc.conj()
        Zloc = rotate_block_matrix(Zloc, lblocks=self.lblocks[isym],
                                 lindices=self.indices,
                                 rblocks=self.rblocks[isym],
                                 rindices=self.indices,
                                )
        if self.time_reversals[isym]:
            Zloc = Zloc.conj()

        return Zloc
