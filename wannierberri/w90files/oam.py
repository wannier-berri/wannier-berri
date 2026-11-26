from .w90file import W90_file, auto_kptirr, check_shape
from ..utility import cached_einsum
import logging
logger = logging.getLogger(__name__)


class OAM(W90_file):
    """
    SPN.data[ik, m, n, ipol] = <u_{m,k}|S_ipol|u_{n,k}>
    """

    extension = "spn"

    def __init__(self, data, NK=None):
        super().__init__(data=data, NK=NK)
        shape = check_shape(self.data)
        self.NB = shape[0]
        assert shape == (self.NB, self.NB, 3), f"SPN data must have shape (NB, NB, 3), got {shape}"



    @classmethod
    def from_bandstructure(cls, bandstructure,
                           l_list,
                           verbose=False,
                           selected_kpoints=None,
                           kptirr=None,
                           NK=None
                           ):
        """
        Create an SPN object from a BandStructure object
        So far only delta-localised s-orbitals are implemented

        Parameters
        ----------
        bandstructure : BandStructure
            the band structure object
        l_list : list of int
            the angular momentum quantum numbers
        normalize : bool
            if True, the wavefunctions are normalised
        """
        assert not bandstructure.spinor, "BandStructure must be scalar for OAM calculation"
        NK, selected_kpoints, kptirr = auto_kptirr(
            bandstructure, selected_kpoints=selected_kpoints, kptirr=kptirr, NK=NK)
        from ..symmetry.angular_momentum import get_L_shell_gpaw

        L_matrices, start, end = [], [], []
        for l in l_list:
            start.append(end[-1] if end else 0)  # each block starts where the previous one ends
            end.append(start[-1] + 2 * l + 1)
            L_matrices.append(get_L_shell_gpaw(l).transpose(1, 2, 0))  # shape (n_orb, n_orb, 3)

        if verbose:
            logger.info(f"Creating OAM from bandstructure with {bandstructure.num_bands} bands and {len(bandstructure.kpoints)} k-points")
        data = {}
        for ikirr in kptirr:
            P = bandstructure.kpoints_paw[selected_kpoints[ikirr]].proj
            data[ikirr] = sum([cached_einsum('mi,nj,ija->mna', P[:, s:e].conj(), P[:, s:e], L) for s, e, L in zip(start, end, L_matrices)])
        return OAM(data=data, NK=NK)

    def select_bands(self, selected_bands):
        return super().select_bands(selected_bands, dimensions=(0, 1))
