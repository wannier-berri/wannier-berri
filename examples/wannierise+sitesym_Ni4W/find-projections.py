from irrep.bandstructure import BandStructure
import numpy as np
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF


from wannierberri.wannierise.projections_searcher import EBRsearcher
from wannierberri.wannierise.projections import Projection, ProjectionsSet


print("calculating symmetrizer")

path = "pwscf/"

try:
    symmetrizer = SAWF().from_npz("Ni4W.sawf.npz")
except FileNotFoundError:
    bandstructure = BandStructure(prefix=path + "Ni4W", code="espresso",
                                Ecut=100, include_TR=False)
    symmetrizer = SAWF().from_irrep(bandstructure)
    symmetrizer.to_npz("Ni4W.sawf.npz")

eig = symmetrizer.eig_irr
print(f"gap is {eig[:, :20].max()} : {eig[:, 20:].min()}")

prefix = path + "diamond"

spacegroup = symmetrizer.spacegroup
trial_projections = ProjectionsSet()

atoms_frac = np.array([[0.20074946, 0.19952625, 0.4002757],
                       [0.59980195, 0.5997243, 0.19952625],
                       [-0.59980195, 0.4002757, -0.19952625],
                       [-0.20074946, 0.80047375, 0.5997243],
                       [0, 0, 0]])


proj_Ni_d = Projection(position_num=[0.20074946, 0.19952625, 0.4002757], orbital='d', spacegroup=spacegroup)
proj_W_d = Projection(position_num=[0, 0, 0], orbital='d', spacegroup=spacegroup)

trial_projections = ProjectionsSet([proj_Ni_d, proj_W_d])


positions = ['0,0,0', '0,1/2,1/2', '0,x,x', 'x,y,0', '3*x,-2*x,x', 'x,y,z']
orbitals = ['s']

for p in positions:
    for o in orbitals:
        proj = Projection(position_sym=p, orbital=o, spacegroup=spacegroup)
        trial_projections.add(proj)

print("trial_projections")
print(trial_projections.write_with_multiplicities(orbit=False))


ebrsearcher = EBRsearcher(
    symmetrizer=symmetrizer,
    trial_projections_set=trial_projections,
    froz_min=-10,
    froz_max=25,
    outer_min=-10,
    outer_max=60,
    debug=True
)

print("searching for combinations")
combinations = ebrsearcher.find_combinations(num_wann_max=40, num_wann_min=36, fixed=(0, 1))

print (f"found {len(combinations)} combinations")

for c in combinations:
    print(("+" * 80 + "\n") * 2)
    print(trial_projections.write_with_multiplicities(c))
    newset = trial_projections.get_combination(c)
    newset.join_same_wyckoff()
    print ("after merging: >>" )
    print(newset.write_with_multiplicities(orbit=False))
    print ("<<")
    newset.maximize_distance()
    print(newset.write_wannier90(mod1=True))
