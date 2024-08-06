# This tutorial shows how to generate a DMN file inside Wanier Berri code (without pw2wannier90)
# and then use it to generate a Wannier90 output.
# It uses Irrep, and may be used with any DFT code that is supported by Irrep (QE, VASP, AINIT, ...)

import datetime
import subprocess
from matplotlib import pyplot as plt
import wannierberri as wberri
from wannierberri.w90files import DMN
from irrep.bandstructure import BandStructure


try:
    dmn_new = DMN("mydmn")
except:
        # see documentation of irrep for usage with other codes
    bandstructure = BandStructure(code='espresso', prefix='di', Ecut=100,
                                normalize=False)

    bandstructure.spacegroup.show()
    dmn_new = DMN(empty=True)
    dmn_new.from_irrep(bandstructure)
    pos = [[0,0,0],[0,0,1/2],[0,1/2,0],[1/2,0,0]]
    dmn_new.set_D_wann_from_projections(projections=[(pos, 's') ])
    dmn_new.to_w90_file("mydmn")


systems = {}

# Just fot Reference : run the Wannier90 with sitesym, but instead of frozen window use outer window
# to exclude valence bands
subprocess.run(["wannier90.x", "diamond"])
#record the system from the Wanier90 output ('diamond.chk' file)
systems["w90"] = wberri.system.System_w90(seedname='diamond')

# Read the data from the Wanier90 inputs 
w90data = wberri.w90files.Wannier90data(seedname='diamond')
w90data.set_file("dmn", dmn_new)
print (dmn_new.num_wann)
print (dmn_new.D_wann.shape)
amn = w90data.amn
print (amn.data.shape)
print ("amn_symmetry", dmn_new.check_amn(w90data.amn.data, warning_precision=1e-4))



#Now disentangle with sitesym and frozen window (the part that is not implemented in Wanier90)
w90data.wannierise(
                froz_min=0,
                froz_max=4,
                num_iter=1000,
                conv_tol=1e-10,
                mix_ratio_z=1.0,
                mix_ratio_u=1.0,
                print_progress_every=20,
                sitesym=True,
                localise=True,
                )

systems["wberri"] = wberri.system.System_w90(w90data=w90data)

# If needed - perform maximal localization using Wannier90

# first generate the reduced files - where num_bands is reduced to num_wann,
# by taking the optimized subspace
w90data_reduced = w90data.get_disentangled(files = ["eig","mmn","amn","dmn"])
w90data_reduced.write("diamond_disentangled", files = ["eig","mmn","amn","dmn"])

# Now write the diamond_disentangled.win file
# first take the existing file
win_file = wberri.w90files.WIN(seedname='diamond')
# # and modify some parameters
win_file["num_bands"] = win_file["num_wann"]
win_file["dis_num_iter"] = 0
win_file["num_iter"] = 1000
del win_file["dis_froz_win"]
del win_file["dis_froz_max"] 
win_file["site_symmetry"] =True
win_file.write("diamond_disentangled")
subprocess.run(["wannier90.x", "diamond_disentangled"])
systems["mlwf"] = wberri.system.System_w90(seedname="diamond_disentangled")

# Now calculate bandstructure for each of the systems
# for creating a path any of the systems will do the job
system0 = list(systems.values())[0]
path = wberri.Path(system0, k_nodes=[[0,0,0],[1/2,0,0],[1,0,0]], labels=['G','L','G'], length=100)
tabulator = wberri.calculators.TabulatorAll(tabulators = {}, mode='path')
calculators = {'tabulate': tabulator}

linecolors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
linestyles = ['-', '--', ':', '-.', ]*4
for key,sys in systems.items():
    result = wberri.run(sys, grid=path, calculators=calculators)
    result.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, 
                                             linecolor=linecolors.pop(0), label=key,
                                             kwargs_line={"ls":linestyles.pop(0)})
plt.title(f"time now {datetime.datetime.now()}")
plt.savefig("diamond-bands-2.png")

# One can see that results do not differ much. Also, the maximal localization does not have much effect.