import numpy as np  

from .system_R import System_R

class SystemSOC(System_R):

    def __init__(self, system_up, system_down=None, axis=(0,0,1), ):
        assert isinstance(system_up, System_R), "system_up must be an instance of System_R"
        self.system_up = system_up
        assert not system_up.is_phonon, "SystemSOC does not support phonons"
        if system_down is None:
            self.system_down = system_up
            self.up_down_same = True
        else:
            assert isinstance(system_down, System_R), "system_down must be an instance of System_R"
            self.up_down_same = False
            self.system_down = system_down
            assert system_up.num_wann == system_down.num_wann, \
                f"Number of Wannier functions must match for up and down systems ({system_up.num_wann} != {system_down.num_wann})"
            system_up.rvec.check_equals(system_down.rvec)
            assert np.allclose(system_up.real_lattice, system_down.real_lattice), \
                f"Real lattices of up and down systems should match {system_up.real_lattice} != {system_down.real_lattice}"
            assert np.all(system_up.periodic == system_down.periodic), \
                f"Periodicities of up and down systems should match {system_up.periodic} != {system_down.periodic}"
            assert not system_up.is_phonon, "SystemSOC does not support phonons in down system"
            
        self.is_phonon = False
        self.num_wann_scalar = system_up.num_wann
        self.num_wann = 2 * self.num_wann_scalar
        self.real_lattice = system_up.real_lattice
        self.periodic = system_up.periodic
        self.axis = np.array(axis, dtype=float)
        self.rvec = self.system_up.rvec.double()
        self.pointgroup = system_up.pointgroup
        self.force_internal_terms_only = any(
            [self.system_up.force_internal_terms_only, self.system_down.force_internal_terms_only])
        
        self.soc_R = None # to be set later
        
    
    def set_soc_R(self, soc_q_H, chk_up, chk_down=None, 
                  kptirr = None, weights_k=None):
        """
        Set the spin-orbit coupling matrix for a given k-point.
        
        Parameters:
        soc_q_H : np.array(num_k, 2* num_bands, 2 * num_bands)
            The spin-orbit coupling matrix in the basis of the bands of the non-spin-orbit coupled system.
            the ordering of bands id up-down-up-down... (interlaced)
        chk_up : bool
            Flag to check if the up-spin system is valid.
        chk_down : bool
            Flag to check if the down-spin system is valid.
        """
        if chk_down is None:
            chk_down = chk_up
        assert chk_up.num_kpts == chk_down.num_kpts, f"Number of k-points must match for up and down systems ({chk_up.num_k} != {chk_down.num_k})"
        assert np.allclose(chk_up.kpt_latt, chk_down.kpt_latt), f"k-point grids should match for up and down systems ({chk_up.kpt_latt} != {chk_down.kpt_latt})"
        assert (kptirr is None) == (weights_k is None), f"kptirr and weights_k must both be provided or both be None ({kptirr=}, {weights_k=})"
        
        self.rvec.set_fft_q_to_R(chk_up.kpt_latt, numthreads=1, fftlib='fftw')
        NK = chk_up.num_kpts
        if kptirr is None:
            kptirr = np.arange(NK)
            weights_k = np.ones(NK, dtype=float)
        else:
            raise NotImplementedError("kptirr and weights_k are not implemented yet")
        soc_q_W = np.zeros((NK, self.num_wann, self.num_wann), dtype=complex)
        for ik, w in zip(kptirr, weights_k):
            v =  [chk_up.v_matrix[ik], chk_down.v_matrix[ik]]
            vt = [v[i].T.conj() for i in range(2)]
            for i in range(2):
                for j in range(2):
                    soc_q_W[ik, i::2, j::2] = w* (vt[i] @ soc_q_H[ik, i::2, j::2] @ v[j])
        soc_q_W =  (soc_q_W + soc_q_W.transpose(0, 2, 1).conj()) / 2.0
        self.soc_R = self.rvec.q_to_R(soc_q_W)
        return self.soc_R

        


