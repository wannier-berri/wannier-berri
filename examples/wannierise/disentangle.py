import wannierberri as wberri
import numpy as np
from matplotlib import pyplot as plt
path_data = "../../tests/data/Fe_sym_Wannier90/"
w90data=wberri.w90files.Wannier90data(seedname=path_data+"Fe_sym")
w90data.write(seedname="Fe_sym_orig", files=['eig','amn','mmn'])

#aidata.apply_outer_window(win_min=-8,win_max= 100 )
froz_max=20
w90data.wannierise( froz_min=-8,
                 froz_max=froz_max,
                print_progress_every=50,
                 num_iter=1001,
                 conv_tol=1e-6,
                 mix_ratio_z=0.7,
                 localise=False
                  )
print (w90data.wannier_centers)
system=wberri.system.System_w90(w90data= w90data)
w90data.write(seedname="Fe_sym_orig2", files=['eig','amn','mmn'])
w90data = w90data.get_disentangled(files = ['amn','mmn','eig'])
w90data.write(seedname="Fe_sym_disentangled", files=['eig','amn','mmn'])
#system=aidata.getSystem(berry=True)
tabulators = { "Energy": wberri.calculators.tabulate.Energy(),
             }

system2=wberri.System_w90(path_data+'Fe_sym')



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

result2=wberri.run(system2,
                  grid=path,
                  calculators={"tabulate" : tab_all_path},
                  print_Kpoints=False)

# plot the bands and compare with W90
EF = 12.6
path_result = result.results["tabulate"]
path_result2 = result2.results["tabulate"]
# Import the pre-computed bands from quantum espresso
# A = np.loadtxt(open(path_data+"Fe_sym_band.dat","r"))
#bohr_ang = scipy.constants.physical_constants['Bohr radius'][0] / 1e-10
#alatt = 5.4235* bohr_ang
#A[:,0]*= 2*np.pi/alatt
# A[:,1]-=EF
# plot it as dots
#plt.scatter (A[:,0], A[:,1], s=5, label="W90")


if False:
    path_result2.plot_path_fat(path,
              quantity=None,
              save_file="Fe_bands.pdf",
              Eshift=EF,
              Emin=-10, Emax=50,
              iband=None,
              mode="fatband",
              fatfactor=20,
              cut_k=False,
              close_fig=False,
              show_fig=False,
              label="W90-wb"
              )


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



energy = path_result.get_data(quantity="Energy",iband=np.arange(0,18))
energy2 = path_result2.get_data(quantity="Energy",iband=np.arange(0,18))
kline = path.getKline()

for i in range(18):
    plt.plot(kline, energy[:, i], color='green')
    plt.plot(kline, energy2[:, i], color='blue')
    plt.scatter(kline,(energy[:, i]+energy2[:, i])/2,color="red",s=abs(energy[:, i]-energy2[:, i])*100,)
plt.ylim(8,20)
plt.show()
select = energy2<froz_max
print (energy.min(),energy.max(),energy.mean())
print (energy2.min(),energy2.max(),energy2.mean())
diff = (energy[select]-energy2[select])
print(energy.shape,energy.size,diff.shape,np.max(abs(diff)), np.linalg.norm(diff)/diff.size)



