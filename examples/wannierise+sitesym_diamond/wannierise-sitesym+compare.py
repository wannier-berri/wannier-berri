# This tutorial shows how to generate a DMN file inside Wanier Berri code (without pw2wannier90)
# and then use it to generate a Wannier90 output.
# It uses Irrep, and may be used with any DFT code that is supported by Irrep (QE, VASP, AINIT, ...)

import os, shutil
import subprocess
from matplotlib import pyplot as plt
import wannierberri as wberri
from wannierberri.symmetry.sawf import SymmetrizerSAWF
from wannierberri.symmetry.projections import Projection

data_dir = "../../tests/data/diamond"


for ext in ["mmn","amn","dmn","eig","win"]:
    shutil.copy(os.path.join(data_dir, "diamond." + ext),
                 os.path.join("./", "diamond." + ext))

# we use artificially small frozen window, for demonstration purposes
frozen_min = 0
frozen_max = 4

win_file = wberri.w90files.WIN(seedname='diamond')
win_file["dis_num_iter"] = 100
win_file["num_iter"] = 1000

# frozen window is not allowed with sitesym in Wannier90
del win_file["dis_froz_min"]
del win_file["dis_froz_max"]
win_file["site_symmetry"] = True
win_file.write("diamond")

# if the dmn file needs to be geneated, (if False - will be read from the file)
generate_dmn = True


# Read the data from the Wanier90 inputs 
w90data = wberri.w90files.Wannier90data(seedname='diamond')

if generate_dmn:
    from irrep.bandstructure import BandStructure
    bandstructure = BandStructure(code='espresso', 
                                prefix=os.path.join(data_dir, "di"),
                                Ecut=100,
                                normalize=False, include_TR=False)
    pos = [[0,0,0],[0,0,1/2],[0,1/2,0],[1/2,0,0]]
    spacegroup = bandstructure.spacegroup
    proj_s = Projection(position_num=pos, orbital='s', spacegroup=spacegroup)
    symmetrizer = SymmetrizerSAWF().from_irrep(bandstructure).set_D_wann_from_projections([proj_s])
else:
    symmetrizer = SymmetrizerSAWF().from_npz(os.path.join(data_dir, "diamond.sawf.npz"))

w90data.set_symmetrizer(symmetrizer)
systems = {}

# Just fot Reference : run the Wannier90 with sitesym, but instead of frozen window use outer window
# to exclude valence bands
subprocess.run(["wannier90.x", "diamond"])
#record the system from the Wanier90 output ('diamond.chk' file)
systems["w90"] = wberri.system.System_w90(seedname='diamond')

w90data.wannierise(
                froz_min=frozen_min,
                froz_max=frozen_max,
                num_iter=1000,
                conv_tol=1e-10,
                mix_ratio_z=1.0,
                print_progress_every=20,
                sitesym=True,
                localise=True,
                )

systems["wberri"] = wberri.system.System_w90(w90data=w90data)

w90data.wannierise(
                init="random",
                froz_min=frozen_min,
                froz_max=frozen_max,
                num_iter=1000,
                conv_tol=1e-10,
                mix_ratio_z=1.0,
                print_progress_every=20,
                sitesym=True,
                localise=True,
                )
systems["wberri-random"] = wberri.system.System_w90(w90data=w90data)


# Now calculate bandstructure for each of the systems
# for creating a path any of the systems will do the job
system0 = list(systems.values())[0]
G = r'$\Gamma$'
path = wberri.Path(system0, k_nodes=[[0,0,0],[1/2,0,0],[1,0,0]], labels=[r"$\Gamma$", "L", r"$\Gamma'$"], length=100)
tabulator = wberri.calculators.TabulatorAll(tabulators = {}, mode='path')
calculators = {'tabulate': tabulator}

linecolors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
linestyles = ['-', '--', ':', '-.', ]*4
for key,sys in systems.items():
    result = wberri.run(sys, grid=path, calculators=calculators)
    result.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, 
                                             linecolor=linecolors.pop(0), label=key,
                                             kwargs_line={"ls":linestyles.pop(0)})
plt.hlines([frozen_max, frozen_min], 0, 100, linestyles='dotted')


plt.savefig("diamond-bands+compare.pdf")

# One can see that results do not differ much. Also, the maximal localization does not have much effect.