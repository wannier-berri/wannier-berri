import numpy as np
import wannierberri as wberri

dmn = wberri.system.w90_files.DMN('diamond')
eig = wberri.system.w90_files.EIG('diamond')
amn = wberri.system.w90_files.AMN('diamond')
mmn = wberri.system.w90_files.MMN('diamond')

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
