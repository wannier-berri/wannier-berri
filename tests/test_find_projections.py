import os
from irrep.bandstructure import BandStructure
from fractions import Fraction
import numpy as np
import sympy
from wannierberri.w90files import DMN, EIG, WIN

from wannierberri.wannierise.projections_searcher import EBRsearcher
from wannierberri.wannierise.projections import Projection, ProjectionsSet

from common import OUTPUT_DIR, ROOT_DIR


def test_find_projections_diamond():
    data_dir = os.path.join(ROOT_DIR, "data", "diamond")
    bandstructure = BandStructure(prefix=data_dir + "/di", code="espresso",
                            Ecut=100, include_TR=False)
    spacegroup = bandstructure.spacegroup
    # spacegroup.show()

    try:
        dmn = DMN(OUTPUT_DIR + "diamond-only-bands")
    except FileNotFoundError:
        dmn = DMN(empty=True)
        dmn.from_irrep(bandstructure)
        dmn.to_npz(OUTPUT_DIR + "diamond-only-bands.dmn")

    prefix = data_dir + "/diamond"
    eig = EIG(prefix)
    win = WIN(prefix)


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
        win=win,
        dmn=dmn,
        eig=eig,
        spacegroup=spacegroup,
        trial_projections=trial_projections,
        froz_min=-10,
        froz_max=30,
        outer_min=-20,
        outer_max=50,
        debug=False
    )

    combinations = ebrsearcher.find_combinations(max_num_wann=10)
    assert len(combinations) == 1
    assert np.all(combinations[0] == [0, 0, 0, 1, 0, 0]), f"combinations[0] = {combinations[0]}, expected [0,0,0,1,0,0]"
    
    ebrsearcher = EBRsearcher(
        win=win,
        dmn=dmn,
        eig=eig,
        spacegroup=spacegroup,
        trial_projections=trial_projections,
        froz_min=-10,
        froz_max=20,
        outer_min=-20,
        outer_max=25,
        debug=False
    )

    combinations = ebrsearcher.find_combinations(max_num_wann=10)
    assert len(combinations) == 1
    assert np.all(combinations[0] == [1, 0, 0, 0, 0, 0]), f"combinations[0] = {combinations[0]}, expected [1,0,0,0,0,0]"