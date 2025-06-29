from wannierberri import Parallel
import wannierberri as wberri
from wannierberri.symmetry.projections import Projection, ProjectionsSet
import os
from matplotlib import pyplot as plt
import numpy as np
from pytest import approx
from irrep.bandstructure import BandStructure
ROOT_DIR = "../../tests/"


parallel = Parallel(num_cpus=16)


path_data = os.path.join(ROOT_DIR, "data", "Fe-222-pw")
nkfull = 8
nkirr = 4
# nkfull=64
# nkirr = 13
# path_data = os.path.join(ROOT_DIR, "data", "Fe-444-sitesym","pwscf")

kwargs_bs = dict(code='espresso',
                prefix=path_data + '/Fe',
                # Ecut=200,
                normalize=False,
                magmom=[[0, 0, 1]],
                include_TR=True,)

bandstructure_full = BandStructure(**kwargs_bs, irreducible=False)
print(f"kpoints in full bz: {[KP.k for KP in bandstructure_full.kpoints]}")
bandstructure_irr = BandStructure(**kwargs_bs, irreducible=True)
print(f"kpoints in irreducible bz: {[KP.k for KP in bandstructure_irr.kpoints]}")

nkp_full = len(bandstructure_full.kpoints)
nkp_irr = len(bandstructure_irr.kpoints)
assert nkp_full == nkfull, f"Expected {nkfull} k-points in full bandstructure, got {nkp_full}"
assert nkp_irr == nkirr, f"Expected {nkirr} k-points in irreducible bandstructure, got {nkp_irr}"
# return
sg = bandstructure_full.spacegroup
projection_sp3d2 = Projection(orbital='sp3d2', position_num=[0, 0, 0], spacegroup=sg)
projection_t2g = Projection(orbital='t2g', position_num=[0, 0, 0], spacegroup=sg)
# projections_set = ProjectionsSet([Projection(orbital=o, position_num=[0, 0, 0], spacegroup=sg) for o in ['s', 'p', 'd']])
projections_set = ProjectionsSet(projections=[projection_sp3d2, projection_t2g])

kwargs_w90file = dict(
    files=['amn', 'mmn', 'spn', 'eig', 'symmetrizer'],
    seedname="./Fe",
    projections=projections_set,
    read_npz_list=[],
    normalize=False)

w90data_full = wberri.w90files.Wannier90data().from_bandstructure(bandstructure_full, **kwargs_w90file)
w90data_irr = wberri.w90files.Wannier90data().from_bandstructure(bandstructure_irr, irreducible=True, **kwargs_w90file)
assert w90data_full.irreducible is False, "w90data_full should not be irreducible"
assert w90data_irr.irreducible is True, "w90data_irr should be irreducible"
nkp_full = len(w90data_full.mmn.data)
nkp_irr = len(w90data_irr.mmn.data)
assert nkp_full == nkfull, f"Expected {nkfull} k-points in full w90data, got {nkp_full}"
assert nkp_irr == nkirr, f"Expected {nkirr} k-points in irreducible w90data, got {nkp_irr}"

w90data_full.select_bands(win_min=-8, win_max=50)
w90data_irr.select_bands(win_min=-8, win_max=50)

kwargs_wannierise = dict(
    init="amn",
    froz_min=-10,
    froz_max=20,
    print_progress_every=10,
    num_iter=101,
    conv_tol=1e-10,
    mix_ratio_z=1.0,
    sitesym=True)

w90data_full.wannierise(**kwargs_wannierise)
w90data_irr.wannierise(**kwargs_wannierise)

print(
    f"Wannier spreads differ between full and irreducible bandstructure: "
    f"{w90data_full.chk.wannier_spreads} != {w90data_irr.chk.wannier_spreads}"
    f"differences : {w90data_full.chk.wannier_spreads - w90data_irr.chk.wannier_spreads}"
    f"maximal difference: {np.max(abs(w90data_full.chk.wannier_spreads - w90data_irr.chk.wannier_spreads))}"
)

print(f"Wannier centers differ between full and irreducible bandstructure: "
    f"{w90data_full.chk.wannier_centers_cart} != {w90data_irr.chk.wannier_centers_cart}"
    f"differences : {w90data_full.chk.wannier_centers_cart - w90data_irr.chk.wannier_centers_cart}"
    f"maximal difference: {np.max(abs(w90data_full.chk.wannier_centers_cart - w90data_irr.chk.wannier_centers_cart))}"
)


system_irr = wberri.system.System_w90(w90data=w90data_irr, spin=True, berry=True, SHCqiao=True)
system_full = wberri.system.System_w90(w90data=w90data_full, spin=True, berry=True, SHCqiao=True)

print(system_irr.pointgroup)
print(system_full.pointgroup)

assert system_irr.rvec.iRvec == approx(
    system_full.rvec.iRvec, abs=1e-5), (
    f"Rvecs differ between full and irreducible system: "
    f"{system_irr.rvec.iRvec} != {system_full.rvec.iRvec}"
    f"differences : {system_irr.rvec.iRvec - system_full.rvec.iRvec}"
    f"maximal difference: {max(abs(system_irr.rvec.iRvec - system_full.rvec.iRvec))}"
)



def report_diff(key):
    mat_irr = system_irr.get_R_mat(key)
    mat_full = system_full.get_R_mat(key)
    diff = abs(mat_irr - mat_full)
    mean_irr = np.mean(abs(mat_irr))
    mean_full = np.mean(abs(mat_full))
    select = (abs(mat_full) >= mean_full) + (abs(mat_irr) >= mean_irr)
    diff_rel = diff[select] / (abs(mat_full[select]) + abs(mat_irr[select]) + 1e-10)
    print(f"Difference in {key} R-matrix: \n"
          f"    max : {np.max(abs(diff))} mean: {np.mean(abs(diff))}\n "
          f"    relative max : {np.max(abs(diff_rel))} mean: {np.mean(abs(diff_rel))}\n"
          f"    mean {key}: {mean_irr}, {mean_full} \n")
    diff_rel_big = np.where(abs(diff_rel) > 0.1)
    for i in diff_rel_big[0]:
        print(f"    {key} {i} : {diff_rel[i]}  irr: {mat_irr[select][i]}, full: {mat_full[select][i]}")


for key in ["Ham", "SS", "AA"]:
    report_diff(key)

path = wberri.Path(system_irr,
             nodes=[
                 [0.0000, 0.0000, 0.0000],  # G
                 [0.500, -0.5000, -0.5000],  # H
                 [0.7500, 0.2500, -0.2500],  # P
                 [0.5000, 0.0000, -0.5000],  # N
                 [0.0000, 0.0000, 0.000]],  # G
    labels=["G", "H", "P", "N", "G"],
    length=200)   # length [ Ang] ~= 2*pi/dk



bands_irr = wberri.evaluate_k_path(system=system_irr, path=path)
bands_full = wberri.evaluate_k_path(system=system_full, path=path)

kwargs_plot = dict(path=path,
              quantity=None,
              save_file=None,
              Emin=-10, Emax=50,
              iband=None,
              mode="fatband",
              fatfactor=20,
              cut_k=False,
              close_fig=False,
              show_fig=False
              )

bands_full.plot_path_fat(**kwargs_plot, linecolor="red", label="full")
bands_irr.plot_path_fat(**kwargs_plot, linecolor="blue", label="irreducible")

plt.savefig("Fe_bands_compare.pdf")

print("K-points in full bandstructure:       ", [kp.k for kp in bandstructure_full.kpoints])
print("K-points in irreducible bandstructure:", [kp.k for kp in bandstructure_irr.kpoints])

print("K-points in full w90data:", w90data_full.mmn.data.keys())
print("K-points in irreducible w90data:", w90data_irr.mmn.data.keys())

Efermi = np.linspace(12, 13, 1001)
calculators = {
    "dos": wberri.calculators.static.DOS(Efermi=Efermi, tetra=True),
    "spin": wberri.calculators.static.Spin(Efermi=Efermi, tetra=True),
    "ahc_internal": wberri.calculators.static.AHC(Efermi=Efermi, tetra=True,
                                                  kwargs_formula={"external_terms": False}),
    "ahc_external": wberri.calculators.static.AHC(Efermi=Efermi, tetra=True,
                                                  kwargs_formula={"internal_terms": False}),
    # "ahc_full" : wberri.calculators.static.AHC(Efermi=Efermi, tetra=False)

}


print(f"point group irred \n{system_irr.pointgroup}\n")
print(f"point group full \n{system_full.pointgroup}\n")

grid = wberri.Grid(system=system_irr, NK=50, NKFFT=4)


kwargs_run = dict(
    grid=grid,
    parallel=parallel,
    adpt_num_iter=0,
    calculators=calculators,
)

results_irr = wberri.run(system=system_irr, suffix="irr",
                         **kwargs_run)

results_full = wberri.run(system=system_full, suffix="full",
                          **kwargs_run)
