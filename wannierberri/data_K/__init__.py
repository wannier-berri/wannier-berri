from ..system.system_kp import SystemKP
from .data_K_k import Data_K_k
from .data_K_R import Data_K_R


def get_data_k(system, dK, grid, **parameters):
    if isinstance(system, SystemKP):
        return Data_K_k(system, dK, grid, **parameters)
    else:
        return Data_K_R(system, dK, grid, **parameters)
