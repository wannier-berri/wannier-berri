import numpy as np
from .w90file import W90_file, check_shape
from ..utility import cached_einsum


class SOC(W90_file):
    """
    stores the SOC hamiltonian in the basis of Blochj states of the non-SOC calculation.
    the order of bands is up-down-up-down, i.e. the spin channels are interlaced
    """

    extension = "soc"
    npz_ags = ["data", "theta", "phi", ]

    def __init__(self, data, NK=None):
        super().__init__(data=data, NK=NK)
        shape = check_shape(self.data)
        self.NB = shape[2]
        assert shape == (2, 2, self.NB, self.NB), f"SOC data must have shape (NB, NB), got {shape}"


    @classmethod
    def from_w90_file(cls, seedname='wannier90', formatted=False):
        raise NotImplementedError("SOC.from_w90_file is not implemented")

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
    def get_S_vss(cls, theta=0, phi=0):
        """
        Get the spin Pauli matrices in the spinor basis defined by C_ss.
        """
        C_ss = cls.get_C_ss(theta, phi)
        sx_ss = np.array([[0, 1], [1, 0]], complex)
        sy_ss = np.array([[0, -1.0j], [1.0j, 0]], complex)
        sz_ss = np.array([[1, 0], [0, -1]], complex)
        s_vss = [
            C_ss.T.conj() @ sx_ss @ C_ss,
            C_ss.T.conj() @ sy_ss @ C_ss,
            C_ss.T.conj() @ sz_ss @ C_ss,
        ]
        return np.array(s_vss)


    @classmethod
    def from_gpaw(cls, calc, magnetic=True, theta=0, phi=0):
        """
        Create SOC from a GPAW magnetic calculation.
        """
        from gpaw.spinorbit import soc
        from ase.units import Hartree
        from gpaw import GPAW
        if isinstance(calc, str):
            calc = GPAW(calc, txt=None)
        C_ss = cls.get_C_ss(theta, phi)

        dVL_avii = {
            a: soc(calc.wfs.setups[a], calc.hamiltonian.xc, D_sp)
            for a, D_sp in calc.density.D_asp.items()
        }

        m = calc.get_number_of_bands()
        print(f"number of bands = {m}")
        nk = len(calc.get_ibz_k_points())

        H_a = {}
        for a, dVL_vii in dVL_avii.items():
            ni = dVL_vii.shape[1]
            H_ssii = np.zeros((2, 2, ni, ni), complex)
            H_ssii[0, 0] = dVL_vii[2]
            H_ssii[0, 1] = dVL_vii[0] - 1j * dVL_vii[1]
            H_ssii[1, 0] = dVL_vii[0] + 1j * dVL_vii[1]
            H_ssii[1, 1] = -dVL_vii[2]
            H_a[a] = H_ssii

        h_soc = np.zeros((nk, 2, 2, m, m), complex)
        print(f"{h_soc.shape=}")

        for q in range(nk):
            for a, H_ssii in H_a.items():
                h_ssii = cached_einsum("ab,bcij,cd->adij", C_ss.T.conj(), H_ssii, C_ss, optimize=True)
                for s1 in range(2):
                    for s2 in range(2):
                        h_ii = h_ssii[s1, s2]
                        if magnetic:
                            P1_mi = calc.wfs.kpt_qs[q][s1].P_ani[a]
                            P2_mi = calc.wfs.kpt_qs[q][s2].P_ani[a]
                        else:
                            P1_mi = calc.wfs.kpt_qs[q].P_ani[a]
                            P2_mi = calc.wfs.kpt_qs[q].P_ani[a]
                        h_soc[q, s1, s2] += np.dot(np.dot(P1_mi.conj(), h_ii), P2_mi.T)
        h_soc *= Hartree
        return cls(data=h_soc, NK=nk)


    def select_bands(self, selected_bands):
        raise NotImplementedError()
