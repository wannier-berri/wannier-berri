#!/usr/bin/env python

from ase import Atoms
from gpaw import GPAW
from ase.parallel import paropen
import ase
from ase import dft
from ase.dft.wannier import Wannier
from matplotlib import pyplot as plt
import numpy as np

# you may set some of those flags to False to speed-up further runs
do_gpaw = False
do_localize = False
do_wberri = True
do_ase_plot = True
do_integrate = True


if do_gpaw:
    print ("doing gpaw calcolation")
    a = 4.4570000
    c = 5.9581176
    x = 0.274

    te = Atoms(
        symbols='Te3',
        scaled_positions=[(x, 0, 0), (0, x, 1. / 3), (-x, -x, 2. / 3)],
        cell=(a, a, c, 90, 90, 120),
        pbc=True)

    calc = GPAW(nbands=16, kpts={'size': (3, 3, 4), 'gamma': True}, symmetry='off', txt='Te.txt')

    te.calc = calc
    te.get_potential_energy()
    calc.write('Te.gpw', mode='all')
    print ("gpaw -done")
else:
    calc = GPAW("../tests/data/Te_ASE/Te.gpw")

if do_localize:
    print ("localizing")
    wan = Wannier(nwannier=12, calc=calc, fixedstates=10)
    wan.localize()  # Optimize rotation to give maximal localization
    print ("localizing-done")
    wan.save('wannier-12.json')  # Save localization and rotation matrix
else:
    wan = Wannier(nwannier=12, calc=calc, file='../tests/data/Te_ASE/wannier-12.json')

k1 = k2 = 1. / 3

if do_wberri:
    import wannierberri as wberri
    system = wberri.System_ASE(wan, ase_calc=calc, berry=True)

    parallel = wberri.parallel.Parallel(num_cpus=4)
#    parallel = wberri.parallel.Serial()

    path = wberri.Path(
        system, k_nodes=[[k1, k2, 0.35], [k1, k2, 0.5], [k1, k2, 0.65]], labels=["K<-", "H", "->K"], length=500)


    calculators = { "tabulate":wberri.calculators.TabulatorAll({
                            "Energy":wberri.calculators.tabulate.Energy(),
                            "berry":wberri.calculators.tabulate.BerryCurvature(),
                                  }, mode="path"
                                    )
                    }


    path_result = wberri.run(system, grid=path, calculators=calculators).results["tabulate"]
    

    path_result.plot_path_fat(
        path,
        save_file=None,
        quantity="berry",
        component="z",
        Eshift=0,
        Emin=5,
        Emax=6,
        iband=None,
        mode="fatband",
        kwargs_line={
            "color": "green",
            "linestyle": "-"
        },
        label="WanniewrBerri",
        fatfactor=5,
        fatmax=200,
        cut_k=True,
        close_fig=False,
        show_fig=False)

    if do_ase_plot:
        kz = np.linspace(0.35, 0.65, 201)
        E = np.array([np.linalg.eigvalsh(wan.get_hamiltonian_kpoint([k1, k2, _])) for _ in kz])
        for e in E.T:
            _line, = plt.plot((kz - kz[0]) * system.recip_lattice[2, 2], e, c='black')
        _line.set_label("directly from ASE")
        plt.legend()
    plt.xlim(0, 0.31)
    plt.ylim(5, 6)

    grid = wberri.Grid(system, length=200, NKFFT=[6, 6, 8])
    if do_integrate:
        Efermi=np.linspace(-10, 10, 1001)
        wberri.run(system,
            grid=grid,
            calculators = {
                "ahc":wberri.calculators.static.AHC(Efermi=Efermi,tetra=False,kwargs_formula={"external_terms": False}),
                "dos":wberri.calculators.static.DOS(Efermi=Efermi,tetra=False),
                          }, 
            parallel=parallel,
            adpt_num_iter=0,
            fout_name='Fe',
            suffix = "run",
            restart=False,
            )

    plt.savefig("Te-bands-wberri+ASE.pdf")
