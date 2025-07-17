from functools import cached_property
import numpy as np
from ..utility import pauli_xyz
from .data_K_R import Data_K_R


class Data_K_soc(Data_K_R):
    """
    Class for handling data_K with non-selfconsistent spin-orbit coupling (SOC).
    """

    def __init__(self, system, dK, grid, **parameters):
        super().__init__(system, dK, grid, **parameters)
        self.num_wann_scalar = system.num_wann_scalar
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

    def Xbar(self, name, der=0):
        key = (name, der)
        if key not in self._bar_quantities:
            if key == ("SS",0):
                SS_K_W = np.array([self.SS_W]*self.nk)
                Xbar =  self._rotate( SS_K_W[self.select_K])
            elif name=="SS":
                Xbar = np.zeros((self.nk, self.num_wann, self.num_wann, 3))
            self._bar_quantities[key] = Xbar
                            
        return self._bar_quantities[key]

    # TODO : move to system
    @cached_property
    def SS_W(self): 
        SS_W = np.zeros( (self.num_wann, self.num_wann,3), dtype=complex)
        for i in range(self.num_wann_scalar):
            SS_W[2*i:2*i+2, 2*i:2*i+2,:] = self.system.S_ssa
        return SS_W

    # Additional methods specific to spin-orbit coupling can be added here.
