import warnings
import numpy as np
from ..utility import orthogonalize
from .utility import get_inverse_block, rotate_block_matrix
from .sawf import SymmetrizerSAWF, VoidSymmetrizer
import logging
logger = logging.getLogger(__name__)


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

    """Symmetrizer for the wannierization matrix at an irreducible k-point

    Parameters
    ----------
    symmetrizer : SymmetrizerSAWF
        The symmetrizer for the full k-point mesh
    ikirr : int
        The index of the irreducible k-point
    accuracy_threshold : float, optional
        The accuracy threshold for excluding bands that do not form a group (and therefore are not symmetrizable)

    """

    def __init__(self, symmetrizer, ikirr, accuracy_threshold=1e-6):
        self.ikirr = ikirr
        self.isym_little = symmetrizer.isym_little[ikirr]
        self.nsym_little = len(self.isym_little)
        self.ikpt = symmetrizer.kptirr[ikirr]
        self.kpt_red = symmetrizer.kpoints_all[self.ikpt]
        self.d_indices = symmetrizer.d_band_block_indices[ikirr]
        self.D_indices = symmetrizer.D_wann_block_indices
        self.d_band_blocks = symmetrizer.d_band_blocks[ikirr]
        self.D_wann_blocks_inverse = symmetrizer.D_wann_blocks_inverse[ikirr]
        self.nb = symmetrizer.NB
        self.num_wann = symmetrizer.num_wann
        self.time_reversals = symmetrizer.time_reversals
        self.include_bands = np.ones(self.nb, dtype=bool)
        err = self.check(accuracy_threshold=accuracy_threshold)
        if err > 1e-6:
            logger.info(f"Symmetrizer_Uirr initialized for ikirr={ikirr}, kpt={self.ikpt}, {self.kpt_red} with {self.nsym_little} symmetries, max error in included blocks: {err} ; "
                f"excluded bands are {np.where(~self.include_bands)[0]} out of {self.nb} total bands (accuracy threshold {accuracy_threshold})")


    def check(self, U=None, verbose=False, accuracy_threshold=1e-6, no_exclude_bands=-4):
        """
        Checks that the symmetrization is correct by comparing eig at the
        irreducible kpoint and at the symmetrized kpoints. 
        Also checks which blocks of bands cannot be symmetrized, and excludes them. (normally that should be only the upper block of bands, otherwise a warning is printed)

        Parameters
        ----------
        warning_precision : float
            the precision above which a warning is printed
        verbose : bool
            if True, prints more information about the errors

        Returns
        -------
        float
            the maximum error found
        """
        if no_exclude_bands <= 0:
            no_exclude_bands = self.nb + no_exclude_bands
        if U is None:
            U = np.random.rand(self.nb, self.num_wann) + 1j * np.random.rand(self.nb, self.num_wann)
        U = self(U)  # symmetrize U
        max_error_in_blocks = np.zeros(len(self.d_indices), dtype=float)
        for isym in self.isym_little:
            U1 = self.rotate_U(U, isym)
            diff = (U1 - U)
            diff = np.max(abs(diff), axis=1)
            for i, (start, end) in enumerate(self.d_indices):
                max_error_in_blocks[i] = max(max_error_in_blocks[i], diff[start:end].max())
        maxerr = 0.0
        for i, (start, end) in enumerate(self.d_indices):
            if max_error_in_blocks[i] > accuracy_threshold:
                if end > no_exclude_bands:
                    self.include_bands[start:end] = False
                    if verbose:
                        logger.info(f"Excluding bands {start} to {end} (block {i}) from symmetrization, max error in block {max_error_in_blocks[i]} exceeds threshold {accuracy_threshold}")
                else:
                    warnings.warn(f"Warning: max error in block {i} [{start}:{end}] is {max_error_in_blocks[i]}, exceeding threshold {accuracy_threshold}, and this is not among the  upper"
                                f" bands({no_exclude_bands}:{self.nb}) bands, this may indicate inaccuracy in the input data")
                    maxerr = max(maxerr, max_error_in_blocks[i])
            else:
                maxerr = max(maxerr, max_error_in_blocks[i])
        return maxerr


    def rotate_U(self, U, isym):
        Uloc = U.copy()
        if self.time_reversals[isym]:
            Uloc = Uloc.conj()
        Uloc = rotate_block_matrix(Uloc,
                                   lblocks=self.d_band_blocks[isym],
                                   lindices=self.d_indices,
                                   rblocks=self.D_wann_blocks_inverse[isym],
                                   rindices=self.D_indices)
        return Uloc


    def __call__(self, U, maxiter=10, tol=1e-6):
        Uprev = U.copy()
        for _ in range(maxiter):
            Uloc = Uprev.copy()
            Usym = sum(self.rotate_U(Uloc, isym) for isym in self.isym_little) / self.nsym_little
            diff1 = abs(Usym - Uprev).max()
            Usym_ortho = np.zeros(Usym.shape, dtype=complex)
            Usym_ortho[self.include_bands] = orthogonalize(Usym[self.include_bands])
            diff2 = abs(Usym_ortho - Usym).max()
            if diff1 < tol and diff2 < tol:
                return Usym
            Uprev = Usym_ortho
        else:
            logger.info(f"Warning: symmetrization did not converge in {maxiter} iterations, final changes {diff1}, {diff2}"
                  "; probably the input data are not perfectly symmetrizable, or the provided projections are not"
                  "compatible with the irreps of the DFT bands."
            )
        return Usym


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
        if Z.shape[0] == 0:
            # return Z # temporary for testing
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
        Zloc = rotate_block_matrix(Zloc, lblocks=self.lblocks[isym],
                                 lindices=self.indices,
                                 rblocks=self.rblocks[isym],
                                 rindices=self.indices,
                                )
        if self.time_reversals[isym]:
            Zloc = Zloc.conj()

        return Zloc
