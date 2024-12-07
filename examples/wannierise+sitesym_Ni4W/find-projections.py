from irrep.bandstructure import BandStructure
import numpy as np
from wannierberri.symmetry.symmetrizer_sawf import SymmetrizerSAWF as SAWF
from wannierberri.w90files import WIN, EIG


from wannierberri.wannierise.projections_searcher import EBRsearcher
from wannierberri.wannierise.projections import Projection, ProjectionsSet


print("calculating DMN")

path = "pwscf/"

bandstructure = BandStructure(prefix=path + "Ni4W", code="espresso",
                            Ecut=100, include_TR=False)
spacegroup = bandstructure.spacegroup
# spacegroup.show()

try:
    symmetrizer = SAWF().from_npz("Ni4W.sawf.npz")
except FileNotFoundError:
    symmetrizer = SAWF().from_irrep(bandstructure)
    symmetrizer.to_npz("Ni4W.sawf.npz")

win = WIN(path+"Ni4W")
eig = EIG(path+"Ni4W")

print (f"gap is {eig.data[:,:20].max()} : {eig.data[:,20:].min()}")

prefix = path + "diamond"

spacegroup = symmetrizer.spacegroup
trial_projections = ProjectionsSet()

atoms_frac = np.array( [[ 0.20074946,  0.19952625,  0.4002757 ],
                        [ 0.59980195,  0.5997243 ,  0.19952625],
                        [-0.59980195,  0.4002757 , -0.19952625],
                        [-0.20074946,  0.80047375,  0.5997243 ],
                        [0,0,0]] )


proj_Ni_d = Projection(position_num = [ 0.20074946,  0.19952625,  0.4002757 ], orbital = 'd', spacegroup=spacegroup)
proj_W_d = Projection(position_num=[0,0,0], orbital='d', spacegroup=spacegroup)

trial_projections = ProjectionsSet([proj_Ni_d, proj_W_d])


positions = ['0,0,0','0,1/2,1/2','0,x,x','x,y,0', '3*x,-2*x,x', 'x,y,z']
orbitals = ['s']

for p in positions:
    for o in orbitals:
        proj = Projection(position_sym=p, orbital=o,spacegroup=spacegroup)
        trial_projections.add(proj)

print ("trial_projections")
print(trial_projections.write_with_multiplicities(orbit=True))


for p in positions:
    for o in ['s']:
        proj = Projection(position_sym=p, orbital=o, spacegroup=spacegroup)
        trial_projections.add(proj)

print("trial_projections")
print(trial_projections.write_with_multiplicities(orbit=False))

ebrsearcher = EBRsearcher(
    win=win,
    symmetrizer=symmetrizer,
    eig=eig,
    spacegroup=spacegroup,
    trial_projections=trial_projections,
    froz_min=-10,
    froz_max=25,
    outer_min=-10,
    outer_max=60,
    debug=False
)

print ("searching for combinations")
combinations = ebrsearcher.find_combinations(max_num_wann=35, fixed=(0,1))

for c in combinations:
    print(("+" * 80 + "\n") * 2)
    print(trial_projections.write_with_multiplicities(c))
    newset = trial_projections.get_combination(c)
    newset.join_same_wyckoff()
    newset.maximize_distance()
    print(newset.write_wannier90(mod1=True))
