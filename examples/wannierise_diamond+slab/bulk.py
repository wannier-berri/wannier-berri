from irrep.bandstructure import BandStructure
import os
import numpy as np
import wannierberri as wberri
from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.system import System_R


projname = "s,p"

bandstructure = BandStructure.from_espresso(
    prefix=os.path.join("../../tests/data", "diamond", "di"),
    Ecut=200,
    normalize=False, include_TR=False)
spacegroup = bandstructure.spacegroup


pos_atom = np.array([[-1, -1, -1], [1, 1, 1]]) / 8
proj_sp3 = Projection(position_num=pos_atom, orbital='sp3', spacegroup=spacegroup, atom_name="C")
proj_p = Projection(position_num=pos_atom, orbital='p', spacegroup=spacegroup, atom_name="C")
proj_s = Projection(position_num=pos_atom, orbital='s', spacegroup=spacegroup, atom_name="C")


if projname == 's,p':
    projset = ProjectionsSet([proj_s, proj_p])
elif projname == 'sp3':
    projset = ProjectionsSet([proj_sp3])
else:
    raise ValueError(f"Unknown system name: {projname}")
froz_min = -10
froz_max = 30
win_min = -10
win_max = 1000

wandata = wberri.WannierData.from_bandstructure(bandstructure, projections=projset)

wberri.wannierise(
    wandata=wandata,
    froz_min=froz_min,
    froz_max=froz_max,
    outer_min=win_min,
    outer_max=win_max,
    num_iter=100,
    conv_tol=1e-10,
    print_progress_every=20,
    sitesym=True,
    localise=True,
)

system = System_R.from_wannierdata(wandata=wandata, symmetrize=False)
system.to_npz(f"diamond-bulk-{projname}")
print(f"wannier names in amn: {wandata.amn.wannier_names}")
print(f"wannier names in chk: {wandata.chk.wannier_names}")
print(f"System created with {system.num_wann} Wannier functions with names: {system.wannier_names}")

path_bulk, bands_bulk = system.get_bandstructure(dk=0.05)
bands_bulk.plot_path_fat(path_bulk,
                        close_fig=False,
                        show_fig=False,
                        linecolor='k',
                        label=projname,
                        Emin=-10,
                        Emax=50,
                        save_file=f'diamond-bulk-{projname}.pdf',
                        )
