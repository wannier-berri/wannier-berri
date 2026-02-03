from irrep.bandstructure import BandStructure
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF

MINIMAL_DISTANCE_THRESHOLD = 0.5

from wannierberri.symmetry.projections_searcher import EBRsearcher
from wannierberri.symmetry.projections import Projection, ProjectionsSet


print("calculating symmetrizer")

path = "../../tests/data/diamond/"

bandstructure = BandStructure(prefix=path + "di", code="espresso",
                            Ecut=100, include_TR=False)
spacegroup = bandstructure.spacegroup
# spacegroup.show()

try:
    symmetrizer = SAWF.from_npz("diamond.sawf.npz")
except FileNotFoundError:
    symmetrizer = SAWF.from_irrep(bandstructure)
    symmetrizer.to_npz("diamond.sawf.npz")


trial_projections = ProjectionsSet()

positions = spacegroup.get_wyckoff_positions(False,False)

print(positions)


for p in positions:
    for o in ['s','p']:
        proj = Projection(position_sym=p, orbital=o, spacegroup=spacegroup)
        trial_projections.add(proj)

print("trial_projections")
print(trial_projections.write_with_multiplicities(orbit=False))

ebrsearcher = EBRsearcher(
    symmetrizer=symmetrizer,
    trial_projections_set=trial_projections,
    froz_min=-10,
    froz_max=20,
    outer_min=-20,
    outer_max=50,
    debug=False
)


combinations = ebrsearcher.find_combinations(num_wann_max=8)

trial_sets = []
for c in combinations:
    print(("+" * 80 + "\n") * 2)
    print(trial_projections.write_with_multiplicities(c))
    newset = trial_projections.get_combination(c)
    newset.join_same_wyckoff()
    newset.maximize_distance()
    if newset.get_minimal_distance() > MINIMAL_DISTANCE_THRESHOLD:
        trial_sets.append(newset)
        # print(newset.write_wannier90(mod1=True))
    # print(newset.write_wannier90(mod1=True))


