from wannierberri.system import System_R
from wannierberri.grid import Path

system_prim = System_R.from_npz('./Si_W90_JM_sym', legacy=True)

Nslab = 10
system_cubic = system_prim.make_supercell([[-1, 1, -1], [-1, 1, 1], [1, 1, -1]])
system_cubic.shift_wannier_centers_to_unit_cell()
system_cubic.rvec.iRvec
system_cubic.wannier_centers_red, system_cubic.wannier_centers_cart
print("building slab supercell")
system_slab_001 = system_cubic.make_supercell([[1, 0, 0], [0, 1, 0], [0, 0, Nslab]])
system_slab_001.wannier_centers_red
system_slab_001.rvec.iRvec
system_slab_001.get_R_mat("Ham").shape
print("setting periodic boundary conditions")
system_slab_001.set_periodic([True, True, False])
path_surf = Path.from_nodes(system_slab_001, nodes=[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]], labels=["G", "X", "K", "G"], dk=0.05)
print("calculating bandstructure")
bands_slab = system_slab_001.get_bandstructure(path=path_surf, return_path=False, k_batch=5)
print("plotting bandstructure")
bands_slab.plot_path_fat(path=path_surf, save_file=f"bands_slab_{Nslab}.png", linecolor='b', kwargs_line={'ls': '-'})
