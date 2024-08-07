
from irrep.bandstructure import BandStructure
import numpy as np
from fractions import Fraction
import sympy
from wannierberri.w90files import DMN, EIG, WIN
prefix = 'tmp2/Ni4W'


from wannierberri.wannierise.projections_searcher import EBRsearcher
from wannierberri.wannierise.projections import Projection, ProjectionsSet

try:
    dmn = DMN(prefix)
    bandstructure = BandStructure(prefix="tmp2/Ni4W", code="espresso", onlysym=True)
    spacegroup = bandstructure.spacegroup
    
except:
    print ("calculating DMN")
    bandstructure = BandStructure(prefix="tmp2/Ni4W", code="espresso")
    spacegroup = bandstructure.spacegroup
    spacegroup.show()
    dmn = DMN(empty=True)
    dmn.from_irrep(bandstructure)
    dmn.to_w90_file(prefix)

eig = EIG(prefix)
win = WIN(prefix)

spacegroup.show()

atoms_frac = np.array( [[ 0.20074946,  0.19952625,  0.4002757 ],
                        [ 0.59980195,  0.5997243 ,  0.19952625],
                        [-0.59980195,  0.4002757 , -0.19952625],
                        [-0.20074946,  0.80047375,  0.5997243 ],
                        [0,0,0]] )


proj_Ni_d = Projection(position_num=[ 0.20074946,  0.19952625,  0.4002757 ], orbital= 'd',spacegroup=spacegroup)
proj_W_d = Projection(position_sym='0,0,0', orbital='d', spacegroup=spacegroup)

trial_projections = ProjectionsSet([proj_Ni_d, proj_W_d])

x,y,z = sympy.symbols('x y z')
# These are wyckoff positions from the Bilbao Crystallographic Server
# They are given in the reference unit cell, so we need to convert them to the primitive unit cell
F12 = Fraction(1,2)
F14 = Fraction(1,4)
WP_refuc =[ [0,0,F12],[0,F12,0],[0,F12,F14], [0,0,x], [F14,F14,F14], [0,F12,x],[x,y,0],[x,y,z] ]

# the vectors of the reference unit cell
# are expressed in the primitive unit cell
refuc = np.array([[-1,1,0],[1,0,1],[0,1,-1]])

WP_prim = np.dot(WP_refuc, refuc)

# now convert the wyckoff positions to strings # TODO: this should be done in the Projection class
positions = [",".join(str(y) for y in x) for x in WP_prim]
print (positions)
# exit()
orbitals = ['s']

for p in positions:
    for o in orbitals:
        proj = Projection(position_sym=p, orbital=o, spacegroup=spacegroup)
        trial_projections.add(proj)

print ("trial_projections")
print(trial_projections.write_with_multiplicities(orbit=True))

ebrsearcher = EBRsearcher(
    win=win,
    dmn=dmn,
    eig=eig,
    spacegroup=spacegroup,
    trial_projections=trial_projections,
    froz_min=10,
    froz_max=25,
    outer_min=10,
    outer_max=100,
)

combinations = ebrsearcher.find_combinations(max_num_wann=40,fixed=(0,1))
for c in combinations:
    print( ("+"*80+"\n")*2 )
    print (trial_projections.write_with_multiplicities(c))
    newset = trial_projections.get_combination(c)
    newset.join_same_wyckoff()
    newset.maximize_distance()
    print(newset.write_wannier90())

