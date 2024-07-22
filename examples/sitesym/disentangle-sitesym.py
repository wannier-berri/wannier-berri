import numpy as np
import wannierberri as wberri

w90data = wberri.system.Wannier90data(seedname='diamond')

w90data.check_symmetry()
# w90data.dmn.check_group("band")
# exit(

systems = []

w90data.disentangle(
                 froz_min=-8,
                 froz_max=20,
                 num_iter=1000,
                 conv_tol=1e-10,
                 mix_ratio=1.0,
                 print_progress_every=20,
                 sitesym=True
                  )

systems["disentangled"] = wberri.system.System_w90(w90data=w90data)
w90data_reduced = w90data.get_disentangled(files = ["eig","mmn","amn","dmn"])
w90data_reduced.write("wannier90_disentangled", files = ["eig","mmn","amn","dmn"])
exit()
path = wberri.Path(system2, k_nodes=[[0,0,0],[1/2,0,0]], labels=['G','L'], length=100)
tabulator = wberri.calculators.TabulatorAll(tabulators = {}, mode='path')
calculators = {'tabulate': tabulator}


result0 = wberri.run(system0, grid=path, calculators=calculators)
result1 = wberri.run(system1, grid=path, calculators=calculators)
result2 = wberri.run(system2, grid=path, calculators=calculators)

result0.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='black')
result1.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='blue')
result2.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=True, linecolor = 'red')

exit()


exit()
system = wberri.system.System_w90(w90data=w90data, berry=True, morb=True, use_wcc_phase=False)
    

print ( dmn.check_eig(eig) )
print ( dmn.check_unitary() )
print (dmn.check_amn(amn))
func = [ lambda x:x,
        lambda x:x.conj(),
        lambda x:x.T,
        lambda x:x.T.conj() ]
for i,f1 in enumerate(func):
    for j,f2 in enumerate(func):
        print (i,j, dmn.check_mmn(mmn, f1, f2)  )
exit()


for ikirr in range(dmn.NKirr):
    for isym in range(dmn.Nsym):
        for iorb in range(dmn.num_wann):
            for jorb in range(dmn.num_wann):
                k1 = dmn.kptirr[ikirr]
                k2 = dmn.kptirr2kpt[ikirr, isym]
                if k1==k2 and iorb==jorb:
                    D = dmn.D_wann_dag[ikirr, isym, iorb, jorb]
                    if abs(D) > 1e-10:
                        print (k1,k2, isym, iorb, jorb, D)
                


# print ('Symmetry:', isym)
# for isym in range(dmn.Nsym):
#     pass
