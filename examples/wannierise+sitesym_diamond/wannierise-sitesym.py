# This tutorial shows how to generate a DMN file inside Wanier Berri code (without pw2wannier90)
# and then use it to generate a Wannier90 output.
# It uses Irrep, and may be used with any DFT code that is supported by Irrep (QE, VASP, AINIT, ...)

import os, shutil
from time import time
import wannierberri as wberri

data_dir = "../../tests/data/diamond"

for ext in ["mmn","amn","dmn","eig","win"]:
    shutil.copy(os.path.join(data_dir, "diamond." + ext),
                 os.path.join("./", "diamond." + ext))
    


# if the dmn file needs to be geneated, (if False - will be read from the file)
generate_dmn = True
sitesym = False

# Read the data from the Wanier90 inputs 

t0 = time()
w90data = wberri.w90files.Wannier90data(seedname='diamond')
t1 = time()
if sitesym and generate_dmn:
    from irrep.bandstructure import BandStructure
    bandstructure = BandStructure(code='espresso', 
                                prefix=os.path.join(data_dir, "di"),
                                Ecut=100,
                                normalize=False, include_TR=False)
    w90data.set_d_band(bandstructure)
    pos = [[0,0,0],[0,0,1/2],[0,1/2,0],[1/2,0,0]]
    w90data.set_D_wann_from_projections(projections=[(pos, 's') ])
t2 = time()
amn = w90data.amn
t3 = time()
if sitesym:
    dmn = w90data.dmn
t4 = time()
# print (amn.data.shape)
if sitesym:
    print ("amn_symmetry", dmn.check_amn(w90data.amn.data, warning_precision=1e-4))
t5 = time()
#Now disentangle with sitesym and frozen window (the part that is not implemented in Wanier90)
w90data.wannierise(
                froz_min=0,
                froz_max=4,
                num_iter=1000,
                conv_tol=1e-10,
                mix_ratio_z=1.0,
                mix_ratio_u=1.0,
                print_progress_every=20,
                sitesym=sitesym,
                localise=True,
                )
t6 = time()
print ("Time to read w90data", t1-t0)
print ("Time to generate dmn", t2-t1)
print ("Time to read amn", t3-t2)
print ("Time to read dmn", t4-t3)
print ("Time to check amn", t5-t4)
print ("Time to wannierise", t6-t5)
print ("Total time", t6-t0)

exit()
system = wberri.system.System_w90(w90data=w90data)


# Now calculate bandstructure 
path = wberri.Path(system, k_nodes=[[0,0,0],[1/2,0,0],[1,0,0]], labels=['G','L','G'], length=100)
tabulator = wberri.calculators.TabulatorAll(tabulators = {}, mode='path')
calculators = {'tabulate': tabulator}

result = wberri.run(system, grid=path, calculators=calculators)
result.results['tabulate'].plot_path_fat(path,
                                         save_file='diamond-bands.pdf',
                                         close_fig=True,
                                         show_fig=False,
                                         )

