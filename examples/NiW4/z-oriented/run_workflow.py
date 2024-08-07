
from collections import defaultdict
import os
import ase
# from ase.build import bulk
# from ase.calculators.espresso import Espresso, EspressoProfile
from matplotlib import pyplot as plt
import numpy as np

from wannierberri.wannierise.projections import Projection, ProjectionsSet
from wannierberri.workflow import WorkflowQE, Executables
from wannierberri.__factors import bohr
x=0.274
ca = 1.3368
alat =  8.42250934 * bohr
atomic_positions = [[x,x,0],[-x,0,1/3],[0,-x,2/3]]


# CELL_PARAMETERS angstrom
#    -2.8685210414    2.8685210414    1.7762410759
#    2.8685210414       -2.8685210414    1.7762410759
#    2.8685210414        2.8685210414    -1.7762410759
# ATOMIC_POSITIONS angstrom
# Ni    1.1446904831    2.2963985565    0.0000000000
# Ni    0.5721224849    4.0132115245    1.7762410759
# Ni    5.1649195979    1.7238305582    1.7762410759
# Ni    4.5923515996    3.4406435263    -0.0000000000
# W    0.0000000000    0.0000000000    0.0000000000

atoms_frac = np.array( [[ 0.20074946,  0.19952625,  0.4002757 ],
                        [ 0.59980195,  0.5997243 ,  0.19952625],
                        [-0.59980195,  0.4002757 , -0.19952625],
                        [-0.20074946,  0.80047375,  0.5997243 ],
                        [0,0,0]] )

a=2.8685210414
ca=1.7762410759/a
structure = ase.Atoms(symbols='Ni4W',
                scaled_positions = atoms_frac,
                cell=a*np.array([ [-1,1,ca], [1,1,ca], [1,1,-ca] ] ),
                pbc=True)

# Pseudopotentials from SSSP Efficiency v1.3.0
pseudopotentials = {"Ni":"Ni.upf", "W":"W.upf"}	

executables = Executables(parallel=True, npar=16, npar_k=16,
                          mpi=" /home/stepan/anaconda3/bin/mpirun",)

high_symmetry_points =  defaultdict(lambda: None)
high_symmetry_points.update(dict(
                            G=[0,     0,   0],
                            K=[1/3, 1/3,   0],
                            M=[0  , 1/2,   0],
                            A=[0  ,   0, 1/2],
                            Z=[0  , 1/2, 1/2],
                            H=[1/2, 1/2, 1/2],
                            ))
path = 'GZAHG'
k_nodes = [high_symmetry_points[k] for k in path]

workflow = WorkflowQE(atoms=structure, 
                      prefix="./tmp2/Ni4W",
                      pseudopotentials=pseudopotentials, 
                      pseudo_dir='../fromEleni/Ni4W',
                      executables=executables,
                      num_bands=140,
                      k_nodes=k_nodes,
                      pickle_file='./tmp2/workflow.pkl',
                      try_unpickle=True,
                      kwargs_gen=dict(ecutwfc=80,
                                       lforcet = False,
                                        occupations='smearing',
                                        smearing='marzari-vanderbilt',
                                        diagonalization='david',
                                        degauss=0.02),
                    use_flags=False
                      )


# workflow.ground_state(kpts=(8,8,8), enforce=False, verbosity="'high'", run=False )
workflow.nscf(mp_grid=(4,4,4), nosym=True, enforce=False, run=False)

spacegroup = workflow.get_spacegroup
proj_Ni_d = Projection(position_num=[ 0.20074946,  0.19952625,  0.4002757 ] , orbital='d', spacegroup=spacegroup)
proj_W_d = Projection(position_num=[ 0,0,0 ] , orbital='d', spacegroup=spacegroup)
# proj_1 = Projection(position_num=[0,1/2,-1/2], orbital='s', spacegroup=spacegroup)
# proj_2 = Projection(position_num=[1/2,1/4,1/4], orbital='s', spacegroup=spacegroup)
# proj_3 = Projection(position_num=[0.616521605167, 0.091962603023, 0.708484208190], orbital='s', spacegroup=spacegroup)

proj_1 = Projection(position_num=[0,1/2,-1/2], orbital='s', spacegroup=spacegroup)
proj_2 = Projection(position_num=[1/2,0,1/2], orbital='s', spacegroup=spacegroup)
proj_3 = Projection(position_num=[0,1/2,0], orbital='s', spacegroup=spacegroup)
proj_4 = Projection(position_num=[0.798005163329, 0.958557153188, 0.756562316517], orbital='s', spacegroup=spacegroup)


projections = ProjectionsSet([ proj_Ni_d, proj_W_d, proj_1, proj_2, proj_3])
# projections.maximize_distance()


workflow.set_projections(projections)


workflow.write_win(
                    # dis_win_min=8,dis_win_max=100,
                    # dis_froz_min=8,dis_froz_max=25,
                    # dis_num_iter=1000, 
                    # num_iter=1000,
                    # site_symmetry=False,symmetrize_eps=1e-9,
                    enforce=True,
                    )


# workflow.pw2wannier(targets=["eig", "mmn", "amn"], enforce=False, run=True)
try:
  os.remove(workflow.prefix+'.dmn.npz')
except:
  pass
try:
  os.remove(workflow.prefix+'.amn.npz')
except:
  pass
workflow.pw2wannier(targets=["amn"], enforce=False, run=True)
workflow.create_dmn(enforce=True)

# workflow.wannierise_w90(enforce=True)

# workflow.calc_bands_wannier_w90(kdensity=1000)
# workflow.calc_bands_qe(kdensity=300, enforce=False, disk_io='low', run=False)



workflow.wannierise_wberri(enforce=True,
                            froz_min=9, froz_max=25, num_iter=100, 
                            kwargs_window=dict(win_min=9, win_max=70),
                            conv_tol=1e-4, mix_ratio_z=1.0, mix_ratio_u=1.0, 
                            print_progress_every=20, sitesym=True, localise=True,
                            kwargs_sitesym={'epsilon':1e-6},)

workflow.flags.on('wannierise_wberri')
workflow.calc_bands_wannier_wberri(kdensity=1000)
workflow.plot(savefig='Ni4W.png', show=False, ylim=(18,22))
