

import numpy as np
from wannierberri.system.system_w90 import System_w90
from wannierberri.system.system_soc import SystemSOC
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.grid.path import Path

path = "../../tests/data/Si_Wannier90/Si"

system_scalar = System_w90(path)
# system_scalar.symmetrize(positions = [  [-0.125,-0.125, 0.375],
#                                         [ 0.375,-0.125,-0.125],
#                                         [-0.125, 0.375,-0.125],
#                                         [-0.125,-0.125,-0.125]],
#                         proj=['bond:s'],
#                         atom_name = ['bond']*4)
num_wann_scalar = system_scalar.num_wann
system_soc = SystemSOC(system_up=system_scalar)
chk  = CHK().from_w90_file(path)

soc_q_H = np.zeros((chk.num_kpts, 2 * chk.num_bands, 2 * chk.num_bands), dtype=complex)
ind_up = 2*np.arange(chk.num_bands)
ind_dw = ind_up + 1

soc_q_H[:, ind_up,ind_up] = 1
soc_q_H[:, ind_dw,ind_dw] = -1


system_soc.set_soc_R(soc_q_H, chk)

path = Path(system_scalar,
            nodes=[[1 / 2, 0, 0],
                    [0, 0, 0],
                    [1 / 2, 0, 1 / 2],
                    [3 / 8, -3 / 8, 0],
                    [0, 0, 0]], 
            labels=['L', 'G', 'X', 'K', 'G'], 
            length=1000)

bands_soc = evaluate_k_path(system_soc, path=path)
bands_scal = evaluate_k_path(system_scalar, path=path)

bands_soc.plot_path_fat(path=path,
                    save_file = 'bands.png',
                    close_fig=False,
                    show_fig=False,
                    linecolor='blue',
                    label='Si SOC',)
                    
bands_scal.plot_path_fat(path=path,
                    save_file = 'bands.png',
                    linecolor='red',
                    kwargs_line={'linestyle': '--'},
                    label='Si scal',)
