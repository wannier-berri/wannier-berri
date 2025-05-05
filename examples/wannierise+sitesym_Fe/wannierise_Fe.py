from wannierberri.symmetry.projections import Projection, ProjectionsSet
from irrep.bandstructure import BandStructure
import wannierberri as wb

path_data = "../../tests/data/Fe-222-pw/"
w90data = wb.w90files.Wannier90data(seedname=path_data + "Fe", readfiles=["mmn", "eig", "win", ], read_npz=True)


bandstructure = BandStructure(code='espresso',
                            prefix=path_data + "/Fe",
                            Ecut=200,
                            normalize=True,
                            magmom=[[0, 0, 1.]],
                            include_TR=False)
spacegroup = bandstructure.spacegroup
spacegroup.show()
# exit()
# symmetrizer = wb.symmetry.sawf.SymmetrizerSAWF().from_npz(path_data + f"/Fe_TR={False}.sawf.npz")
symmetrizer = wb.symmetry.sawf.SymmetrizerSAWF().from_irrep(bandstructure)

projection_s = Projection(orbital='s', position_num=[0, 0, 0], spacegroup=spacegroup)
projection_p = Projection(orbital='p', position_num=[0, 0, 0], spacegroup=spacegroup)
projection_d = Projection(orbital='d', position_num=[0, 0, 0], spacegroup=spacegroup)
projection_sp3d2 = Projection(orbital='sp3d2', position_num=[0, 0, 0], spacegroup=spacegroup)
projection_t2g = Projection(orbital='t2g', position_num=[0, 0, 0], spacegroup=spacegroup)

# projections_set = ProjectionsSet(projections=[projection_s, projection_p, projection_d])
projections_set = ProjectionsSet(projections=[projection_sp3d2, projection_t2g])

symmetrizer.set_D_wann_from_projections(projections_set)
amn = wb.w90files.amn_from_bandstructure(bandstructure, projections=projections_set)
w90data.set_symmetrizer(symmetrizer)
w90data.set_file("amn", amn)

w90data.select_bands(win_min=-8, win_max=50)

w90data.wannierise(init="amn",
                   froz_min=-10,
                   froz_max=20,
                   print_progress_every=10,
                   num_iter=101,
                   conv_tol=1e-10,
                   mix_ratio_z=1.0,
                   sitesym=True,
                   parallel=False
                    )
