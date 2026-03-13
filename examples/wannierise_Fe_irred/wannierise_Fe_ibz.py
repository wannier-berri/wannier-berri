from wannierberri.symmetry.projections import Projection, ProjectionsSet
from irrep.bandstructure import BandStructure
import wannierberri as wberri
from wannierberri.parallel import ray_init
from wannierberri.system import System_R


ray_init()
path_data = "../../tests/data/Fe-444-sitesym/pwscf-irred/"

bandstructure = BandStructure(code='espresso',
                            prefix=path_data + "/Fe",
                            magmom=[[0, 0, 1.]],
                            include_TR=True,
                            irreducible=True,
                            # select_grid=[2,2,2],  # optionaly -reduce the grid
                            )
spacegroup = bandstructure.spacegroup

spacegroup.show()


if True:

    projection_sp3d2 = Projection(orbital='sp3d2',
                                position_num=[0, 0, 0],
                                spacegroup=spacegroup)
    projection_t2g = Projection(orbital='t2g',
                                position_num=[0, 0, 0],
                                spacegroup=spacegroup)
    projections_spd = [Projection(orbital=orb, position_num=[0, 0, 0], spacegroup=spacegroup)
                       for orb in ['s', 'p', 'd']]


    projections_set = ProjectionsSet(projections=[projection_sp3d2, projection_t2g])

    # projections_set = ProjectionsSet(projections=projections_spd)


    wandata = wberri.WannierData.from_bandstructure(
        bandstructure,
        seedname="./Fe",
        files=['amn', 'mmn', 'spn', 'eig', 'symmetrizer'],
        projections=projections_set,
        normalize=False
    )

    wberri.wannierise(wandata=wandata,
                    init="amn",
                    froz_min=-10,
                    froz_max=20,
                    outer_min=-8,
                    outer_max=50,
                    print_progress_every=10,
                    num_iter=41,
                    conv_tol=1e-10,
                    mix_ratio_z=1.0,
                    sitesym=True,
                    parallel=True
                        )
    wandata.to_npz(seedname="./Fe_wan")

else:
    # for further runs just load the data
    wandata = wberri.WannierData.from_npz(seedname="./Fe_wan",
                                                   files=['chk', 'amn', 'mmn', 'spn', 'eig', 'symmetrizer'],)

system = System_R.from_wannierdata(wandata=wandata, spin=True, berry=True)


# all kpoints given in reduced coordinates
path = wberri.Path(system=system,
             nodes=[
                 [0.0000, 0.0000, 0.0000],  # G
                 [0.500, -0.5000, -0.5000],  # H
                 [0.7500, 0.2500, -0.2500],  # P
                 [0.5000, 0.0000, -0.5000],  # N
                 [0.0000, 0.0000, 0.000]],  # G
    labels=["G", "H", "P", "N", "G"],
    length=200)   # length [ Ang] ~= 2*pi/dk


bands = wberri.evaluate_k_path(system=system,
                           path=path
                           )

bands.plot_path_fat(path,
              quantity=None,
              save_file="Fe_bands.pdf",
              Emin=-10, Emax=50,
              iband=None,
              mode="fatband",
              fatfactor=20,
              cut_k=False,
              label="WB",
              )
