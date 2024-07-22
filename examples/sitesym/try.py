from matplotlib import pyplot as plt
import wannierberri as wberri

dmn = wberri.system.w90_files.DMN(seedname='diamond')
print (dmn.NB, dmn.check_unitary())
dmn = wberri.system.w90_files.DMN(seedname='diamond_disentangled')
print (dmn.NB, dmn.check_unitary())


w90data = wberri.system.Wannier90data(seedname='diamond')
dmn = w90data.dmn



w90data.write(seedname="diamond_disentangled", files=['eig','amn','mmn','dmn'])
w90data.dmn

w90data.check_symmetry()

# system2 = wberri.system.System_w90('diamond')
# system0 = wberri.system.System_w90('ref/diamond')

froz_min = -0
froz_max =  20
w90data.disentangle(
                 froz_min=froz_min,
                 froz_max=froz_max,
                 num_iter=1000,
                 conv_tol=1e-10,
                 mix_ratio=1.0,
                 print_progress_every=20,
                 sitesym=True
                  )
w90data = w90data.get_disentangled(files = ['eig','amn','mmn','dmn'])

w90data.write(seedname="diamond_disentangled", files=['eig','amn','mmn','dmn'])
exit()
system1 = wberri.system.System_w90(w90data=w90data)
path = wberri.Path(system2, k_nodes=[[0, 0, 0],
                                     [0.5, 0, 0],
                                     [0.5, 0.5, 0],
                                     [0, 0, 0] 
                                     ], labels=['G', 'L', 'X', 'G'
                                                ], length=100)
tabulator = wberri.calculators.TabulatorAll(tabulators={}, mode='path')
calculators = {'tabulate': tabulator}
kwargs = dict(grid=path, calculators=calculators, print_Kpoints=False, file_Klist=None)
result0 = wberri.run(system0, **kwargs)
result1 = wberri.run(system1, **kwargs)
result2 = wberri.run(system2, **kwargs)

# reference
result0.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='black')

# wberri disentranglement
result1.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='red', kwargs_line={'linestyle': '--'})

# w90 disentranglement -sitesym without frozen window
result2.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='blue')
plt.ylim(-10, 30)
xmin, xmax = plt.xlim()
for froz in froz_min, froz_max:
    plt.hlines(froz, xmin, xmax, colors='gray')
plt.show()
