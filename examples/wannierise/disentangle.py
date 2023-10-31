import wannierberri as wberri
import numpy as np
from matplotlib import pyplot as plt
aidata=wberri.system.AbInitioData(seedname='data/Fe_sym')
#aidata.apply_outer_window(win_min=-8,win_max= 100 )
aidata.disentangle( froz_min=-8,
                 froz_max=20,
                 num_iter=2000,
                 conv_tol=1e-9,
                 mix_ratio=1.0
                  )
print (aidata.wannier_centres)
system=aidata.getSystem()
#system=wb.System_Wanneirise(aidata)
tabulators = { "Energy": wberri.calculators.tabulate.Energy(),
             #  "berry" : wberri.calculators.tabulate.BerryCurvature(),
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

result=wberri.run(system,
                  grid=path,
                  calculators={"tabulate" : tab_all_path},
                  print_Kpoints=False)

# plot the bands and compare with W90
EF = 12.6
path_result = result.results["tabulate"]
# Import the pre-computed bands from quantum espresso
A = np.loadtxt(open("data/Fe_sym_band.dat","r"))
#bohr_ang = scipy.constants.physical_constants['Bohr radius'][0] / 1e-10
#alatt = 5.4235* bohr_ang
#A[:,0]*= 2*np.pi/alatt
A[:,1]-=EF
# plot it as dots
plt.scatter (A[:,0], A[:,1], s=5, label="W90")




path_result.plot_path_fat(path,
              quantity=None,
              save_file="Fe_bands.pdf",
              Eshift=EF,
              Emin=-10, Emax=50,
              iband=None,
              mode="fatband",
              fatfactor=20,
              cut_k=False,
              close_fig=True,
              show_fig=True,
              label="WB"
              )

