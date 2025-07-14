from ..system.system_kp import SystemKP
from ..system.system_soc import SystemSOC
from .data_K_k import Data_K_k
from .data_K_R import Data_K_R
from .data_K_soc import Data_K_soc



def get_data_k(system, dK, grid, **parameters):
    if isinstance(system, SystemKP):
        return Data_K_k(system, dK, grid, **parameters)
    elif isinstance(system, SystemSOC):
        return Data_K_soc(system, dK, grid, **parameters)
    else:
        return Data_K_R(system, dK, grid, **parameters)
