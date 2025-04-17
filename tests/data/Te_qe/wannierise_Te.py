from wannierberri.wannierise.projections import Projection, ProjectionsSet
from irrep.bandstructure import BandStructure
import wannierberri as wb
prefix = "tellurium"
w90data = wb.w90files.Wannier90data(seedname=prefix, readfiles=["mmn", "eig", "win", "uhu", "spn"])

bandstructure = BandStructure(code='espresso',
                            prefix=prefix,
                            Ecut=100,
                            normalize=False,
                            magmom=True,
                            include_TR=True)
spacegroup = bandstructure.spacegroup
symmetrizer = wb.symmetry.sawf.SymmetrizerSAWF().from_irrep(bandstructure)

x = 0.27
positions = [[x, x, 0], [-x, 0, 1 / 3], [0, -x, -1 / 3]]
projection_s = Projection(orbital='s', position_num=positions, spacegroup=spacegroup)
projection_p = Projection(orbital='p', position_num=positions, spacegroup=spacegroup)


projections_set = ProjectionsSet(projections=[projection_s, projection_p])

symmetrizer.set_D_wann_from_projections(projections_obj=projections_set)
amn = wb.w90files.amn_from_bandstructure(bandstructure, projections_set=projections_set)
w90data.set_symmetrizer(symmetrizer)
w90data.set_file("amn", amn)
print(f"amn {amn.data.shape}")


w90data.wannierise(init="amn",
                   froz_min=-100,
                   froz_max=8,
                   print_progress_every=1,
                   num_iter=11,
                   conv_tol=1e-6,
                   mix_ratio_z=1.0,
                   sitesym=True,
                   parallel=False
                    )



system = wb.system.System_w90(w90data=w90data, morb=True, spin=True)
system.symmetrize2(symmetrizer=symmetrizer)

system.save_npz(path="Te-sys", overwrite=True)


system2 = wb.system.System_R(berry=True, morb=True, spin=True,
                                    ).load_npz("Te-sys")
