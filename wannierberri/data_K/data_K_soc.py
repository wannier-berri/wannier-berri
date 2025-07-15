from functools import cached_property
import numpy as np
from .data_K_R import Data_K_R


class Data_K_soc(Data_K_R):
    """
    Class for handling data_K with non-selfconsistent spin-orbit coupling (SOC).
    """

    def __init__(self, system, dK, grid, **parameters):
        super().__init__(system, dK, grid, **parameters)
        self.data_K_up = Data_K_R(system.system_up, dK, grid, **parameters)
        if system.up_down_same:
            self.data_K_down = self.data_K_up
        else:
            self.data_K_down = Data_K_R(system.system_down, dK, grid, **parameters)
        soc_r_dk = self.rvec.apply_expdK(system.soc_R)
        self.set_R_mat('soc', soc_r_dk)

    @cached_property
    def HH_K(self):
        """
        Returns the Hamiltonian in k-space, including spin-orbit coupling.
        """
        H = np.zeros((self.nk, self.num_wann, self.num_wann), dtype=complex)
        H[:, ::2, ::2] = self.data_K_up.HH_K
        H[:, 1::2, 1::2] = self.data_K_down.HH_K
        H += self.rvec.R_to_k(self.get_R_mat('soc'), hermitian=True)
        return H

    # Additional methods specific to spin-orbit coupling can be added here.
