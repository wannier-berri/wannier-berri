import numpy as np
from .w90file import W90_file
from ..io import FortranFileR
from ..utility import pauli_xyz


class SPN(W90_file):
    """
    SPN.data[ik, m, n, ipol] = <u_{m,k}|S_ipol|u_{n,k}>
    """

    def __init__(self, seedname="wannier90", **kwargs):
        self.npz_tags = ["data"]
        super().__init__(seedname=seedname, ext="spn", **kwargs)

    def from_w90_file(self, seedname='wannier90', formatted=False):
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
        self.data = np.zeros((NK, nbnd, nbnd, 3), dtype=complex)

        for ik in range(NK):
            A = np.zeros((3, nbnd, nbnd), dtype=complex)
            if formatted:
                tmp = np.array([f_spn_in.readline().split() for i in range(3 * nbnd * (nbnd + 1) // 2)], dtype=float)
                tmp = tmp[:, 0] + 1.j * tmp[:, 1]
            else:
                tmp = f_spn_in.read_record(dtype=np.complex128)
            A[:, indn, indm] = tmp.reshape(3, nbnd * (nbnd + 1) // 2, order='F')
            check = np.einsum('ijj->', np.abs(A.imag))
            A[:, indm, indn] = A[:, indn, indm].conj()
            if check > 1e-10:
                raise RuntimeError(f"REAL DIAG CHECK FAILED : {check}")
            self.data[ik] = A.transpose(1, 2, 0)
        print("----------\n SPN OK  \n---------\n")
        return self


    def from_bandstructure(self, bandstructure,
                           normalize=True, verbose=False):
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
        assert bandstructure.spinor, "SPN only works for spinor bandstructures"

        if verbose:
            print(f"Creating SPN from bandstructure with {bandstructure.num_bands} bands and {len(bandstructure.kpoints)} k-points")
        self.data = []
        for kp in bandstructure.kpoints:
            print(f"setting spn for k={kp.k}")
            ng = kp.ig.shape[1]
            wf = kp.WF
            if normalize:
                wf /= np.linalg.norm(wf, axis=1)[:, None]
            wf = wf.reshape((bandstructure.num_bands, 2, ng), order='C')
            data = np.einsum('mri,nsi,rst->mnt', wf.conj(), wf, pauli_xyz)
            self.data.append(data)

        print(f"length of data = {len(data)}")
        print("NK={self.NK}")
        return self

    def select_bands(self, selected_bands):
        if selected_bands is not None:
            self.data = self.data[:, selected_bands, :, :][:, :, selected_bands, :]
