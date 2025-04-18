from irrep.bandstructure import BandStructure
from fractions import Fraction
import sympy
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF


from wannierberri.wannierise.projections_searcher import EBRsearcher
from wannierberri.wannierise.projections import Projection, ProjectionsSet


print("calculating symmetrizer")

path = "../../tests/data/diamond/"

bandstructure = BandStructure(prefix=path + "di", code="espresso",
                            Ecut=100, include_TR=False)
spacegroup = bandstructure.spacegroup
# spacegroup.show()

try:
    symmetrizer = SAWF().from_npz("diamond.sawf.npz")
except FileNotFoundError:
    symmetrizer = SAWF().from_irrep(bandstructure)
    symmetrizer.to_npz("diamond.sawf.npz")



trial_projections = ProjectionsSet()

x, y, z = sympy.symbols('x y z')
F12 = Fraction(1, 2)
F14 = Fraction(1, 4)
F18 = Fraction(1, 8)
WP = [[0, 0, 0], [x, 0, 0], [F12, F12, F12], [F14, F14, F14], [F18, F18, F18], [0, x, z]]
# in principle, those should be all wyckoff position for the spacegroup
# but we will only consider a few random positions
positions = [",".join(str(y) for y in x) for x in WP]
print(positions)


for p in positions:
    for o in ['s']:
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


combinations = ebrsearcher.find_combinations(num_wann_max=10)

for c in combinations:
    print(("+" * 80 + "\n") * 2)
    print(trial_projections.write_with_multiplicities(c))
    newset = trial_projections.get_combination(c)
    newset.join_same_wyckoff()
    newset.maximize_distance()
    print(newset.write_wannier90(mod1=True))
