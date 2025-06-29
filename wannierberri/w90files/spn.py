import numpy as np
from .w90file import W90_file, auto_kptirr, check_shape
from ..io import FortranFileR
from ..utility import cached_einsum, pauli_xyz


class SPN(W90_file):
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
    def from_w90_file(cls, seedname='wannier90', formatted=False):
        print("----------\n SPN  \n---------\n")
        if formatted:
            f_spn_in = open(seedname + ".spn", 'r')
            SPNheader = f_spn_in.readline().strip()
            nbnd, NK = (int(x) for x in f_spn_in.readline().split())
        else:
            f_spn_in = FortranFileR(seedname + ".spn")
            SPNheader = (f_spn_in.read_record(dtype='c'))
            nbnd, NK = f_spn_in.read_record(dtype=np.int32)
            SPNheader = "".join(a.decode('ascii') for a in SPNheader)

        print(f"reading {seedname}.spn : {SPNheader}")

        indm, indn = np.tril_indices(nbnd)
        data = np.zeros((NK, nbnd, nbnd, 3), dtype=complex)

        for ik in range(NK):
            A = np.zeros((3, nbnd, nbnd), dtype=complex)
            if formatted:
                tmp = np.array([f_spn_in.readline().split() for i in range(3 * nbnd * (nbnd + 1) // 2)], dtype=float)
                tmp = tmp[:, 0] + 1.j * tmp[:, 1]
            else:
                tmp = f_spn_in.read_record(dtype=np.complex128)
            A[:, indn, indm] = tmp.reshape(3, nbnd * (nbnd + 1) // 2, order='F')
            check = cached_einsum('ijj->', np.abs(A.imag))
            A[:, indm, indn] = A[:, indn, indm].conj()
            if check > 1e-10:
                raise RuntimeError(f"REAL DIAG CHECK FAILED : {check}")
            data[ik] = A.transpose(1, 2, 0)
        print("----------\n SPN OK  \n---------\n")
        return SPN(data=data)

    @classmethod
    def from_bandstructure(cls, bandstructure,
                           normalize=True, verbose=False,
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
        normalize : bool
            if True, the wavefunctions are normalised
        """
        NK, selected_kpoints, kptirr = auto_kptirr(
            bandstructure, selected_kpoints=selected_kpoints, kptirr=kptirr, NK=NK)
        from ..import IRREP_IRREDUCIBLE_VERSION
        from packaging import version
        from irrep import __version__ as irrep__version__
        irrep_new_version = (version.parse(irrep__version__) >= IRREP_IRREDUCIBLE_VERSION)

        assert bandstructure.spinor, "SPN only works for spinor bandstructures"

        if verbose:
            print(f"Creating SPN from bandstructure with {bandstructure.num_bands} bands and {len(bandstructure.kpoints)} k-points")
        data = {}
        for ikirr in kptirr:
            kp = bandstructure.kpoints[selected_kpoints[ikirr]]
            # print(f"setting spn for k={kp.k}")
            ng = kp.ig.shape[1]
            wf = kp.WF if irrep_new_version else kp.WF.reshape((bandstructure.num_bands, ng, 2), order='F')
            if normalize:
                wf /= np.linalg.norm(wf, axis=(1, 2))[:, None, None]
            # wf = wf.reshape((bandstructure.num_bands, 2, ng), order='C')
            data_k = cached_einsum('mir,nis,rst->mnt', wf.conj(), wf, pauli_xyz)
            data[ikirr] = data_k

        return SPN(data=data, NK=NK)

    def select_bands(self, selected_bands):
        return super().select_bands(selected_bands, dimensions=(0, 1))
