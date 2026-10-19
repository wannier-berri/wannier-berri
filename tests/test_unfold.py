import numpy as np
from pythtb import tb_model

from wannierberri.system.system_R import System_R



def test_unfold_1d():
    lat_pc = np.array([[1.0]])
    lat_sc = np.array([[2.0]])
    orb_pc = np.array([[0.0]])
    orb_sc = np.array([[0.0],[0.5]])
    transform_PC_to_SC = np.array([[2,0,0],[0,1,0],[0,0,1]])


    model_pc = tb_model(2, 1, lat=lat_pc)
    model_sc = tb_model(2, 1, lat=lat_sc)
    model_pc.set_hop(1.0, 0, 0, [1])
    model_sc.set_hop(1.0, 0, 1, [0])
    model_sc.set_hop(1.0, 1, 0, [1])
    system_pc = System_R.from_pythtb(model_pc, orb_pc)
    system_sc = System_R.from_pythtb(model_sc, orb_sc)

    path_pc = 
