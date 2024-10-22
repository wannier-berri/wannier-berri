
import os
import ase
# from ase.build import bulk
# from ase.calculators.espresso import Espresso, EspressoProfile
from ase.optimize import LBFGS
import numpy as np

from wannierberri.workflow import WorkflowQE, Executables

diamond = ase.Atoms(symbols='C2',
                scaled_positions = np.array([[-1,-1,-1],[1,1,1]])/8,
                cell=1.614*np.array([[-1,0,1],
                                     [0,1,1],
                                     [-1,1,0]]),
                pbc=True)

# Pseudopotentials from SSSP Efficiency v1.3.0
pseudopotentials = {"C": "C.pz-vbc.UPF"}

executables = Executables(parallel=True, npar=4,
                          mpi=" /home/stepan/anaconda3/bin/mpirun",)


# projections_bond = [ [pos,0,1] for x0 in (0.125,0.625) for pos in [[x0,0,0.5],[0,x0,-1/6],[-x0,-x0,1/6]] ]

workflow = WorkflowQE(atoms=diamond, 
                      prefix="./tmp1/diamond",
                      pseudopotentials=pseudopotentials, 
                      pseudo_dir='../../tests/data/diamond/pw_inputs',
                      executables=executables,
                      num_bands=10,
                      kwargs_gen={'ecutwfc': 40, },
                      k_nodes=[[0,0,0], [0, 0, 1/2]],
                      pickle_file='diamond.pkl',
                      try_unpickle=True
                      )


workflow.ground_state(kpts=(12,12,12), enforce=False)
workflow.nscf(kpts=(4,4,4), nosym=False, enforce=False)

projections = [([0,0,0],0),
               ([0,0,1/2],0),
               ([0,1/2,0],0),
               ([1/2,0,0],0),
               ]
workflow.set_projections(projections)

# workflow.write_win(enforce=True)
workflow.write_win(
                    dis_win_min=-20,dis_win_max=20,
                    dis_num_iter=0, 
                   num_iter=1000,
                   site_symmetry=True,symmetrize_eps=1e-9,
                   enforce=True, )

workflow.pw2wannier(targets=["amn", 'dmn', "eig", "mmn","dmn"], enforce=False)
workflow.calc_bands_qe(kdensity=300, enforce=True, disk_io='low')
workflow.wannierise_w90(enforce=True)
workflow.calc_bands_wannier_w90(kdensity=1000)

workflow.wannierise_wberri(enforce=True,
                            froz_min=-20, froz_max=20, num_iter=1000, 
                            conv_tol=1e-10, mix_ratio_z=1.0, mix_ratio_u=1.0, 
                            print_progress_every=20, sitesym=True, localise=True,)
workflow.calc_bands_wannier_wberri(kdensity=1000)
workflow.plot()
exit()

exit()


workflow_qe(te,
            prefix="_Te",
            # path_to_pw="m/home/stepan/QE/qe-7.3.1/bin/pw.x",
            mp_grid=(3,3,4),
            pseudopotentials=pseudopotentials,
            projections=projections,
            num_bands=20,
            take_every=10,
            recalc_gs=False, 
            recalc_nscf=False, 
            recalc_pw2wannier=False,
            pw2wannier_list=["amn","mmn","eig"],
            recalc_wannier=False, 
            recalc_bands_qe=True,
            kwargs_gen={'pseudo_dir': '../',},
            kwargs_wannier={'num_iter':0, 'dis_num_iter':1000, 'dis_froz_min':0, 'dis_froz_max':7},
            )

exit()


kwargs = dict( 
              ibrav=0,
    ecutwfc = 20,
    lspinorb=False,
    noncolin=False,
    occupations='smearing', smearing='cold', degauss=0.02,
    restart_mode='from_scratch',
    outdir='./',
    prefix='Te',
    verbosity='low',
    startingwfc='random',
    diagonalization='cg',
    conv_thr=1.0e-10,
    pseudo_dir='../',
    )

kwargs_scf = dict(calculation='scf',  **kwargs) 
kwargs_nscf = dict(calculation='nscf',  nbnd=20, nosym=True, 
                   wf_collect=True, 
                   **kwargs)



do_scf = False
do_nscf = True

if do_scf:
    write_espresso_in('Te.scf.in', te, kpts=(6,6,8), pseudopotentials=pseudopotentials,
                                    input_data=kwargs_scf)
    os.system('pw.x -i Te.scf.in > Te.scf.out')


kgrid_nscf = (3,3,4)
kpoints = np.array([[i,j,k] for i in range(kgrid_nscf[0]) for j in range(kgrid_nscf[1]) for k in range(kgrid_nscf[2])]
                    )/np.array(kgrid_nscf)[None,:]
kpoints_str= '\n'.join([" ".join([f"{x:16.12f}" for x in k])+"  1" for k in kpoints])

if do_nscf:
    write_espresso_in('Te.nscf.in', te, pseudopotentials=pseudopotentials,
                                    nbnd=20,
                                    input_data=kwargs_nscf, 
                                    kpoints_array=kpoints,
                                    #
                                    )
    os.system('pw.x -i Te.nscf.in > Te.nscf.out')

from irrep.bandstructure import BandStructure
bs = BandStructure(code='espresso', prefix='Te', Ecut=30,  )
kpoints = np.array([kp.K for kp in bs.kpoints])
print (kpoints)

data=dict(kpoints=kpoints,
          unit_cell_cart=te.get_cell(),
          atoms_frac=te.get_scaled_positions(),
          atoms_names = te.get_chemical_symbols(),
          num_wann=3,
          projections = ["Te:s"]
) 

win = WIN(seedname=None,data=data)
win.write('Te')
print (win.data['atoms_frac'])





exit()



te.calc = calc

te.get_potential_energy()  # This will run a single point calculation

opt = LBFGS(te)

opt.run(fmax=0.005)  # This will run a geometry optimization using ASE's LBFGS algorithm

# Print lattice constant...
print(te.get_volume)
print(te.get_cell_lengths_and_angles())