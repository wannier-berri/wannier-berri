import warnings
import numpy as np
from .w90file import W90_file, check_shape
from ..utility import cached_einsum, pauli_xyz


class SOC(W90_file):
    """
    stores the SOC hamiltonian in the basis of Blochj states of the non-SOC calculation.
    the order of bands is up-down-up-down, i.e. the spin channels are interlaced

    Parameters
    ----------
    data : np.ndarray or dict
        The SOC Hamiltonian data.
    overlap : np.ndarray, optional
        The overlap matrix elements between the spin-up and spin-down states.  O_{ij} = < Psi^up_i | Psi^down_j >

    Attributes
    ----------
    data : {ik:  np.ndarray(2,2,3,NB,NB)}
        The SOC Hamiltonian data. 
        data[ik][s1,s2,t1,t2,i,j] = < Psi^s1_i | H_soc^{t1,t2} | Psi^s2_j > 
    overlap : {ik:  np.ndarray(NB,NB)}
        The overlap matrix elements between the spin-up and spin-down states.  O_{ij} = < Psi^up_i | Psi^down_j >
    nspin : int
        Number of spin channels (1 or 2).
    NB : int
        Number of bands (in each spin channel).

    """

    extension = "soc"
    npz_tags = ["NK"]
    npz_keys_dict_int = ["data", "overlap"]

    def __init__(self, data, NK=None, overlap=None):
        super().__init__(data=data, NK=NK)
        shape = check_shape(self.data)
        self.NB = shape[4]
        self.nspin = shape[0]
        assert shape == (self.nspin, self.nspin, 3, self.NB, self.NB), f"SOC data must have shape (nspin, nspin, 3, NB, NB), got {shape}"
        if isinstance(overlap, list) or isinstance(overlap, np.ndarray):
            NK = len(data)
            self.overlap = {i: d for i, d in enumerate(overlap) if d is not None}
        elif isinstance(overlap, dict):
            self.overlap = overlap
        elif overlap is None:
            warnings.warn("No overlap matrix provided, using identity matrices - this mightt be very inaccurate in some cases.")
            self.overlap = {i: np.eye(self.NB, dtype=complex) for i in self.data}
        else:
            raise ValueError(f"Invalid overlap input: {overlap} of type {type(overlap)}. Should be a list or array or None")

    @classmethod
    def from_w90_file(cls, seedname='wannier90', formatted=False):
        raise NotImplementedError("SOC.from_w90_file is not implemented - there is no such w90 file")

    @classmethod
    def from_bandstructure(cls, bandstructure_soc,
                           bandsctructure_up,
                           bandsctructure_down=None,
                           verbose=False,
                           selected_kpoints_soc=None,
                           selected_kpoints_scal=None,
                           kptirr=None,
                           NK=None
                           ):
        raise NotImplementedError("SOC.from_bandstructure is not implemented")


    @classmethod
    def get_C_ss(cls, theta=0, phi=0):
        """
        Get the spin rotation matrix C_ss for the given angles theta and phi.
        """
        ct2 = np.cos(theta / 2)
        st2 = np.sin(theta / 2)
        ep2 = np.exp(-1.0j * phi / 2)
        C_ss = np.array([[ct2 * ep2, -st2 * ep2],
                         [st2 / ep2, ct2 / ep2]], complex)
        # C_ss = np.array([[np.cos(theta / 2) * np.exp(-1.0j * phi / 2),
        #                   -np.sin(theta / 2) * np.exp(-1.0j * phi / 2)],
        #                 [np.sin(theta / 2) * np.exp(1.0j * phi / 2),
        #                     np.cos(theta / 2) * np.exp(1.0j * phi / 2)]])
        return C_ss

    @classmethod
    def get_pauli_rotated(cls, theta=0, phi=0):
        """
        Get the rotated Pauli matrices for the given angles theta and phi.
        """
        C_ss = cls.get_C_ss(theta, phi)
        return cached_einsum('ai,abc,bj->ijc', C_ss.conj(), pauli_xyz, C_ss)

    # original code by Yaroslav
    # def get_S_ssv(cls, theta=0, phi=0):
        # sx_ss = np.array([[0, 1], [1, 0]], complex)
        # sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
        # sz_ss = np.array([[1, 0], [0, -1]], complex)
        # s_vss = [
        #     C_ss.T.conj() @ sx_ss @ C_ss,
        #     C_ss.T.conj() @ sy_ss @ C_ss,
        #     C_ss.T.conj() @ sz_ss @ C_ss,
        # ]
        # return np.array(s_vss).transpose(1, 2, 0)


    @classmethod
    def from_gpaw(cls, calculator, calc_overlap=True):
        """
        Create SOC from a GPAW magnetic calculation.
        """
        from gpaw.spinorbit import soc
        from ase.units import Hartree
        from gpaw import GPAW
        if isinstance(calculator, str):
            calculator = GPAW(calculator, txt=None)
        nspin = calculator.get_number_of_spins()
        assert nspin in [1, 2], f"Only nspin=1 or 2 supported, got {nspin}"

        dVL_avii = {
            a: soc(calculator.wfs.setups[a], calculator.hamiltonian.xc, D_sp)
            for a, D_sp in calculator.density.D_asp.items()
        }

        m = calculator.get_number_of_bands()
        print(f"number of bands = {m}")
        nk = len(calculator.get_ibz_k_points())

        # H_a = {}
        # for a, dVL_vii in dVL_avii.items():
        #     ni = dVL_vii.shape[1]
        #     H_ssii = np.zeros((2, 2, ni, ni), complex)
        #     H_ssii[0, 0] = dVL_vii[2]
        #     H_ssii[0, 1] = dVL_vii[0] - 1j * dVL_vii[1]
        #     H_ssii[1, 0] = dVL_vii[0] + 1j * dVL_vii[1]
        #     H_ssii[1, 1] = -dVL_vii[2]
        #     H_a[a] = H_ssii


        dV_soc = np.zeros((nk, nspin, nspin, 3, m, m), complex)

        # TODO : use time-reversal symmetry in case of non-magnetic calculation to calculate only one spin channel and one off-diagonal block
        for q in range(nk):
            for a, H_ssii in dVL_avii.items():
                for s1 in range(nspin):
                    for s2 in range(nspin):
                        P1_mi = calculator.wfs.kpt_qs[q][s1].P_ani[a].conj()
                        P2_mi = calculator.wfs.kpt_qs[q][s2].P_ani[a].T
                        for t in range(3):
                            dV_soc[q, s1, s2, t] += P1_mi @ H_ssii[t] @ P2_mi
        dV_soc *= Hartree

        if nspin == 2 and calc_overlap:
            overlap = np.zeros((nk, m, m), complex)
            alpha = calculator.wfs.gd.dv / calculator.wfs.gd.N_c.prod()
            for q in range(nk):
                psi1 = calculator.wfs.kpt_qs[q][0].psit_nG[:]
                psi2 = calculator.wfs.kpt_qs[q][1].psit_nG[:]
                overlap[q] += alpha * psi1.conj() @ psi2.T
                for a, setup in enumerate(calculator.wfs.setups):
                    P1_mi = calculator.wfs.kpt_qs[q][0].P_ani[a]
                    P2_mi = calculator.wfs.kpt_qs[q][1].P_ani[a]
                    overlap_ii = setup.dO_ii
                    overlap[q] += P1_mi.conj() @ overlap_ii @ P2_mi.T
        else:
            overlap = None

        return cls(data=dV_soc, overlap=overlap, NK=nk)


    def select_bands(self, selected_bands_up, selected_bands_down=None):
        if selected_bands_down is None:
            selected_bands_down = selected_bands_up
        self.NB = len(selected_bands_up)
        assert len(selected_bands_up) == len(selected_bands_down), \
            "selected_bands_up and selected_bands_down must have the same length"
        if self.overlap is not None:
            for ik in self.overlap:
                self.overlap[ik] = self.overlap[ik][selected_bands_up][:, selected_bands_down]
        selected_bands = [selected_bands_up, selected_bands_down]
        for ik in self.data:
            data_new = np.zeros((self.nspin, self.nspin, 3, self.NB, self.NB), dtype=complex)
            for s in range(self.nspin):
                for t in range(self.nspin):
                    data_new[s, t] = self.data[ik][s, t][:, selected_bands[s]][:, :, selected_bands[t]]
            self.data[ik] = data_new
