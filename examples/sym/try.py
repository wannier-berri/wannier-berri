import numpy as np
import wannierberri as wberri

dmn = wberri.system.w90_files.DMN('diamond')
eig = wberri.system.w90_files.EIG('diamond')
amn = wberri.system.w90_files.AMN('diamond')

print ( dmn.check_eig(eig) )
print ( dmn.check_unitary() )
print (dmn.check_amn(amn))

# print ('Symmetry:', isym)
# for isym in range(dmn.Nsym):
#     pass
