from ..system.system_R import System_R
from ..system.system_kp import SystemKP
from ..system.system_soc import SystemSOC
from .data_K_k import Data_K_k
from .data_K_R import Data_K_R
from .data_K_soc import Data_K_soc



def get_data_k_class_from_system(system):
    if isinstance(system, SystemKP):
        return Data_K_k
    elif isinstance(system, SystemSOC):
        return Data_K_soc
    elif isinstance(system, System_R):
        return Data_K_R
    else:
        raise ValueError(f"unknown system type {type(system)}")
    