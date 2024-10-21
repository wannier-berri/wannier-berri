from matplotlib import pyplot as plt
import scipy
import wannierberri as wberri
import numpy as np

from irrep.bandstructure import BandStructure
from wannierberri.w90files import DMN
from wannierberri.parallel import Parallel
path_data = "./"

results_path = {}
results_grid = {}
include_TR_list = [ True, False, None]

parallel = Parallel(ray_init={"num_gpus": 0})

for includeTR in include_TR_list:
    w90data=wberri.w90files.Wannier90data(seedname=path_data+"Fe")    
    if includeTR is not None:
        sitesym = True
        bandstructure = BandStructure(code='espresso', prefix=path_data+'Fe', Ecut=100,
                                    normalize=False, magmom=[[0,0,1]], 
                                    include_TR=includeTR)

        w90data.set_d_band(bandstructure, overwrite=True)
        pos = [[0,0,0]]
        w90data.set_D_wann_from_projections(projections=[(pos, 's'),  (pos, 'p'), (pos, 'd') ])
    else:
        sitesym = False

    #aidata.apply_outer_window(win_min=-8,win_max= 100 )
    froz_max=30
    w90data.wannierise( init = "amn",
                    froz_min=-8,
                    froz_max=froz_max,
                    print_progress_every=10,
                    num_iter=21,
                    conv_tol=1e-6,
                    mix_ratio_z=1.0,
                    localise=True,
                    sitesym=sitesym,
                    )

    # print (w90data.wannier_centers)
    system=wberri.system.System_w90(w90data= w90data, berry=True)
    # system.set_symmetry(spacegroup=bandstructure.spacegroup)
    tabulators = { "Energy": wberri.calculators.tabulate.Energy(),
                }


    tab_all_path = wberri.calculators.TabulatorAll(
                        tabulators,
                        ibands=np.arange(0,18),
                        mode="path"
                            )

    # all kpoints given in reduced coordinates
    path=wberri.Path(system,
                    k_nodes=[
                        [0.0000, 0.0000, 0.0000 ],   #  G
                        [0.500 ,-0.5000, -0.5000],   #  H
                        [0.7500, 0.2500, -0.2500],   #  P
                        [0.5000, 0.0000, -0.5000],   #  N
                        [0.0000, 0.0000, 0.000  ]
                            ] , #  G
                    labels=["G","H","P","N","G"],
                    length=200 )   # length [ Ang] ~= 2*pi/dk

    results_path[includeTR]=wberri.run(system,
                    parallel=parallel,
                    grid=path,
                    calculators={"tabulate" : tab_all_path},
                    print_Kpoints=False)
    
    grid =  wberri.Grid(system, NKFFT=6, NK=48)
    efermi = np.linspace(12.4,12.8,1001)
    param = dict(Efermi=efermi)
    calculators = {
        # "CDOS": wberri.calculators.static.CumDOS(**param),
                #    "ohmic": wberri.calculators.static.Ohmic_FermiSea(**param),
                   "ahc_internal": wberri.calculators.static.AHC(kwargs_formula={"external_terms":False}, **param),
                "ahc_external": wberri.calculators.static.AHC(kwargs_formula={"internal_terms":False}, **param ),	
    }

    results_grid[includeTR] = wberri.run(system, grid, calculators, 
                                          fout_name="Fe_grid",
                                          suffix = f"TR={includeTR}",
                                          adpt_num_iter=0,
                                          symmetrize=False,
                                          use_irred_kpt=False,
                                          print_progress_step=1,
                                          parallel=parallel,
                                          )



# plot the bands and compare with pw
EF = 12.6

A = np.loadtxt("Fe_bands_pw.dat")
bohr_ang = scipy.constants.physical_constants['Bohr radius'][0] / 1e-10
alatt = 5.4235* bohr_ang
A[:,0]*= 2*np.pi/alatt
A[:,1] = A[:,1] - EF
plt.scatter(A[:,0], A[:,1], c="black", s=5)



colors = ["blue", "red","black"]
for includeTR in include_TR_list:
    path_result = results_path[includeTR].results["tabulate"]

    path_result.plot_path_fat(path,
                quantity=None,
                # save_file="Fe_bands.pdf",
                Eshift=EF,
                Emin=-10, Emax=50,
                iband=None,
                mode="fatband",
                fatfactor=20,
                cut_k=False,
                linecolor=colors.pop(),
                close_fig=False,
                show_fig=False,
                label=f"TR={includeTR}"
                )

plt.ylim(-10, 20)
plt.hlines(froz_max-EF, 0, A[-1,0], linestyles="dashed")
plt.legend()
plt.savefig("Fe_bands.pdf")