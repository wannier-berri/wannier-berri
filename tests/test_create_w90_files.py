from .test_wannierise import spreads_Fe_spd_444_nowin as spreads_Fe_spd_444
import irrep
from irrep.bandstructure import BandStructure
import pytest
from pytest import approx, fixture
import wannierberri as wberri
import numpy as np
import os

from wannierberri.symmetry.projections import Projection, ProjectionsSet
from wannierberri.w90files.eig import EIG
from .common import OUTPUT_DIR, ROOT_DIR, REF_DIR
from wannierberri.symmetry.sawf import SymmetrizerSAWF


@fixture
def check_sawf():
    def _inner(sawf_new, sawf_ref):

        for key in ['NB', "num_wann", "NK", "NKirr", "kptirr", "kptirr2kpt", "kpt2kptirr", "time_reversals"]:
            assert np.all(getattr(sawf_ref, key) == getattr(sawf_new, key)), (
                f"key {key} differs between reference and new SymmetrizerSAWF\n"
                f"reference: {getattr(sawf_ref, key)}\n"
                f"new: {getattr(sawf_new, key)}\n"
            )

        assert np.all(sawf_ref.D_wann_block_indices == sawf_new.D_wann_block_indices), (
            f"D_wann_block_indices differs between reference and new SymmetrizerSAWF\n"
            f"reference: {sawf_ref.D_wann_block_indices}\n"
            f"new: {sawf_new.D_wann_block_indices}\n"
        )
        for ikirr in range(sawf_ref.NKirr):
            assert np.all(sawf_ref.d_band_block_indices[ikirr] == sawf_new.d_band_block_indices[ikirr]), (
                f"d_band_block_indices differs  at ikirr={ikirr} between reference and new SymmetrizerSAWF\n"
                f"reference: {sawf_ref.d_band_block_indices}\n"
                f"new: {sawf_new.d_band_block_indices}\n"
            )

        for i, blockpair in enumerate(zip(sawf_ref.rot_orb_list, sawf_new.rot_orb_list)):
            blockref, blocknew = blockpair
            assert blockref.shape == blocknew.shape, f"rot_orb in differs for block {i} between reference and new SymmetrizerSAWF\n"
            assert blockref == approx(blocknew, abs=1e-6), f"rot_orb in differs for block {i} between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"


        for isym in range(sawf_ref.Nsym):
            try:
                for ikirr in range(sawf_ref.NKirr):
                    for blockref, blocknew in zip(sawf_ref.D_wann_blocks[ikirr][isym], sawf_new.D_wann_blocks[ikirr][isym]):
                        assert blockref == approx(blocknew, abs=1e-6), f"D_wann at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"
                    for blockref, blocknew in zip(sawf_ref.d_band_blocks[ikirr][isym], sawf_new.d_band_blocks[ikirr][isym]):
                        assert blockref == approx(blocknew, abs=1e-6), f"d_band at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"
            except AssertionError as error:
                try:
                    for ikirr in range(sawf_ref.NKirr):
                        for blockref, blocknew in zip(sawf_ref.D_wann_blocks[ikirr][isym], sawf_new.D_wann_blocks[ikirr][isym]):
                            assert blockref == approx(-blocknew, abs=1e-6), f"D_wann at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref + blocknew))} > 1e-6"
                        for blockref, blocknew in zip(sawf_ref.d_band_blocks[ikirr][isym], sawf_new.d_band_blocks[ikirr][isym]):
                            assert blockref == approx(-blocknew, abs=1e-6), f"d_band at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref + blocknew))} > 1e-6"
                except AssertionError as error2:
                    raise AssertionError(
                        f"SymmetrizerSAWF objects differ for isym={isym}:\n{error}\n{error2}\n error1: {error}\n error2: {error2}\n"
                    ) from error


    return _inner


def test_create_sawf_diamond(check_sawf):
    data_dir = os.path.join(ROOT_DIR, "data", "diamond")

    bandstructure = irrep.bandstructure.BandStructure(prefix=data_dir + "/di", Ecut=100,
                                                      code="espresso",
                                                      include_TR=False,
                                                      )
    projection = Projection(position_num=[[0, 0, 0], [0, 0, 1 / 2], [0, 1 / 2, 0], [1 / 2, 0, 0]],
                            orbital='s',
                            spacegroup=bandstructure.spacegroup
                            )
    sawf_new = SymmetrizerSAWF().from_irrep(bandstructure)
    sawf_new.set_D_wann_from_projections([projection])

    tmp_sawf_path = os.path.join(OUTPUT_DIR, "diamond")
    sawf_new.to_npz(tmp_sawf_path + ".sawf.npz")
    sawf_ref = SymmetrizerSAWF().from_npz(data_dir + "/diamond.sawf.npz")
    check_sawf(sawf_new, sawf_ref)


@pytest.mark.parametrize("select_grid", [None, (4, 4, 4), (2, 2, 2)])
def test_create_w90files_diamond_irred(select_grid):

    data_dir = os.path.join(ROOT_DIR, "data", "diamond")

    bandstructure = irrep.bandstructure.BandStructure(prefix=data_dir + "/di-irred",
                                                      code="espresso",
                                                      include_TR=False,
                                                      select_grid=select_grid,
                                                      )
    path_tmp = os.path.join(OUTPUT_DIR, "diamond-create-w90-files-irred")
    prefix = "di-irred"
    projection = Projection(position_num=[[0, 0, 0], [0, 0, 1 / 2], [0, 1 / 2, 0], [1 / 2, 0, 0]],
                            orbital='s',
                            spacegroup=bandstructure.spacegroup
                            )
    proj_set = wberri.symmetry.projections.ProjectionsSet(projections=[projection])
    w90data = wberri.w90files.Wannier90data(
    ).from_bandstructure(bandstructure,
                         files=["mmn", "eig", "amn", "unk", "symmetrizer"],
                         write_npz_list=[],
                         read_npz_list=[],
                         seedname=os.path.join(path_tmp, prefix),
                         projections=proj_set,
                         normalize=False,
                         irreducible=True,
                         )
    w90data.wannierise(
        froz_min=-8,
        froz_max=20,
        num_iter=1000,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        print_progress_every=20,
        sitesym=True,
        localise=True
    )
    wannier_centers = w90data.chk.wannier_centers_cart
    wannier_spreads = w90data.chk.wannier_spreads
    wannier_spreads_mean = np.mean(wannier_spreads)
    assert wannier_spreads == approx(wannier_spreads_mean, abs=1e-9)
    spread_ref = 0.39865686  if select_grid == (2, 2, 2) else 0.580249066578
    assert wannier_spreads == approx(spread_ref, abs=1e-5)
    assert wannier_centers == approx(np.array([[0, 0, 0],
                                            [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ]).dot(w90data.chk.real_lattice) / 2,
        abs=1e-6)



@pytest.mark.parametrize("include_TR", [True, False])
def test_create_sawf_Fe(check_sawf, include_TR):
    path_data = os.path.join(ROOT_DIR, "data", "Fe-222-pw")

    bandstructure = BandStructure(code='espresso', prefix=path_data + '/Fe', Ecut=200,
                                normalize=False, magmom=[[0, 0, 1]], include_TR=include_TR)
    sawf_new = SymmetrizerSAWF().from_irrep(bandstructure, ecut=100)
    pos = [[0, 0, 0]]
    proj_s = Projection(position_num=pos, orbital='s', spacegroup=bandstructure.spacegroup)
    proj_p = Projection(position_num=pos, orbital='p', spacegroup=bandstructure.spacegroup)
    proj_d = Projection(position_num=pos, orbital='d', spacegroup=bandstructure.spacegroup)
    sawf_new.set_D_wann_from_projections(projections=[proj_s, proj_p, proj_d])
    fname = f"Fe_TR={include_TR}.sawf.npz"
    tmp_sawf_path = os.path.join(OUTPUT_DIR, fname)
    sawf_new.to_npz(tmp_sawf_path)
    sawf_ref = SymmetrizerSAWF().from_npz(os.path.join(REF_DIR, "sawf", fname))
    check_sawf(sawf_new, sawf_ref)


@pytest.mark.parametrize("include_TR", [True, False])
@pytest.mark.parametrize("irr_bs", [False, True])
def test_create_sawf_Fe_irreducible(check_sawf, include_TR, irr_bs):
    path_data = os.path.join(ROOT_DIR, "data", "Fe-222-pw")

    bandstructure = BandStructure(code='espresso', prefix=path_data + '/Fe', Ecut=100,
                                normalize=False, magmom=[[0, 0, 1]], include_TR=include_TR,
                                  irreducible=irr_bs)
    sawf_new = SymmetrizerSAWF().from_irrep(bandstructure, irreducible=True, ecut=None)
    pos = [[0, 0, 0]]
    proj_s = Projection(position_num=pos, orbital='s', spacegroup=bandstructure.spacegroup)
    proj_p = Projection(position_num=pos, orbital='p', spacegroup=bandstructure.spacegroup)
    proj_d = Projection(position_num=pos, orbital='d', spacegroup=bandstructure.spacegroup)
    sawf_new.set_D_wann_from_projections(projections=[proj_s, proj_p, proj_d])
    fname = f"Fe_TR={include_TR}-irr-bs={irr_bs}.sawf.npz"
    tmp_sawf_path = os.path.join(OUTPUT_DIR, fname)
    sawf_new.to_npz(tmp_sawf_path)
    fname = f"Fe_TR={include_TR}-irr.sawf.npz"
    sawf_ref = SymmetrizerSAWF().from_npz(os.path.join(REF_DIR, "sawf", fname))
    check_sawf(sawf_new, sawf_ref)


def test_irreducible_vs_full_Fe():
    path_data = os.path.join(ROOT_DIR, "data", "Fe-222-pw")
    kwargs_bs = dict(code='espresso',
                  prefix=path_data + '/Fe',
                normalize=False,
                  magmom=[[0, 0, 1]],
                  include_TR=True,)

    bandstructure_full = BandStructure(**kwargs_bs, irreducible=False)
    print(f"kpoints in full bz: {[KP.k for KP in bandstructure_full.kpoints]}")
    bandstructure_irr = BandStructure(**kwargs_bs, irreducible=True)
    print(f"kpoints in irreducible bz: {[KP.k for KP in bandstructure_irr.kpoints]}")

    nkp_full = len(bandstructure_full.kpoints)
    nkp_irr = len(bandstructure_irr.kpoints)
    assert nkp_full == 8, f"Expected 8 k-points in full bandstructure, got {nkp_full}"
    assert nkp_irr == 4, f"Expected 4 k-points in irreducible bandstructure, got {nkp_irr}"
    # return
    sg = bandstructure_full.spacegroup
    projection_sp3d2 = Projection(orbital='sp3d2', position_num=[0, 0, 0], spacegroup=sg)
    projection_t2g = Projection(orbital='t2g', position_num=[0, 0, 0], spacegroup=sg)
    projections_set = ProjectionsSet(projections=[projection_sp3d2, projection_t2g])

    kwargs_w90file = dict(
        files=['amn', 'mmn', 'spn', 'eig', 'symmetrizer'],
        seedname="./Fe",
        projections=projections_set,
        read_npz_list=[],
        normalize=False)

    w90data_full = wberri.w90files.Wannier90data().from_bandstructure(bandstructure_full, **kwargs_w90file)
    w90data_irr = wberri.w90files.Wannier90data().from_bandstructure(bandstructure_irr, **kwargs_w90file)
    assert w90data_full.irreducible is False, "w90data_full should not be irreducible"
    assert w90data_irr.irreducible is True, "w90data_irr should be irreducible"
    nkp_full = len(w90data_full.mmn.data)
    nkp_irr = len(w90data_irr.mmn.data)
    assert nkp_full == 8, f"Expected 8 k-points in full w90data, got {nkp_full}"
    assert nkp_irr == 4, f"Expected 4 k-points in irreducible w90data, got {nkp_irr}"


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

    assert w90data_full.chk.wannier_spreads == approx(
        w90data_irr.chk.wannier_spreads, abs=0.01), (
        f"Wannier spreads differ between full and irreducible bandstructure: "
        f"{w90data_full.chk.wannier_spreads} != {w90data_irr.chk.wannier_spreads}"
    )

    assert w90data_full.chk.wannier_centers_cart == approx(
        w90data_irr.chk.wannier_centers_cart, abs=0.01), (
        f"Wannier centers differ between full and irreducible bandstructure: "
        f"{w90data_full.chk.wannier_centers_cart} != {w90data_irr.chk.wannier_centers_cart}"
    )
    kwargs_system = dict(spin=True, berry=True, symmetrize=True)
    system_irr = wberri.system.System_w90(w90data=w90data_irr, **kwargs_system)
    system_full = wberri.system.System_w90(w90data=w90data_full, **kwargs_system)


    Efermi = np.linspace(12, 13, 1001)
    calculators = {
        "dos":
        wberri.calculators.static.DOS(Efermi=Efermi, tetra=True),
        "spin":
        wberri.calculators.static.Spin(Efermi=Efermi, tetra=True),
        # "ahc_internal" :
        #     wberri.calculators.static.AHC(Efermi=Efermi, tetra=True,
        #                                   kwargs_formula={"external_terms": False}),
        "ahc_external":
        wberri.calculators.static.AHC(Efermi=Efermi, tetra=True,
                                      kwargs_formula={"internal_terms": False}),
        # "ahc_full" : wberri.calculators.static.AHC(Efermi=Efermi, tetra=False)

    }


    grid = wberri.Grid(system=system_irr, NKdiv=4, NKFFT=4)


    kwargs_run = dict(
        grid=grid,
        adpt_num_iter=0,
        calculators=calculators,
    )

    results_irr = wberri.run(system=system_irr, suffix="irr",
                            **kwargs_run)

    results_full = wberri.run(system=system_full, suffix="full",
                            **kwargs_run)

    assert results_irr.results["dos"].data == approx(results_full.results["dos"].data, abs=1e-3)
    assert results_irr.results["spin"].data == approx(results_full.results["spin"].data, abs=1e-5)
    assert results_irr.results["ahc_external"].data == approx(results_full.results["ahc_external"].data, abs=1.0)




def check_create_w90files_Fe(path_data, path_ref=None,
                             irreducible=False, irr_bs=False,
                             select_grid=None,
                             wannierise=False, spreads_ref=None,
                             select_window=None,
                             prefix='Fe'):
    path_tmp = os.path.join(OUTPUT_DIR, f"Fe-create-w90-files-irreducible={irr_bs}-{irreducible}")
    os.makedirs(path_tmp, exist_ok=True)

    bandstructure = BandStructure(code='espresso', prefix=path_data + '/' + prefix,
                                normalize=False, magmom=[[0, 0, 1]],
                                irreducible=irr_bs,
                                include_TR=True,
                                select_grid=select_grid,
                                )

    norms = np.array([np.linalg.norm(kp.WF, axis=(1))**2 for kp in bandstructure.kpoints])
    if norms.ndim == 3:
        norms = norms.sum(axis=2)  # sum over spinor components in irrep>=2.2

    pos = [[0, 0, 0]]
    proj_s = Projection(position_num=pos, orbital='s', spacegroup=bandstructure.spacegroup)
    proj_p = Projection(position_num=pos, orbital='p', spacegroup=bandstructure.spacegroup)
    proj_d = Projection(position_num=pos, orbital='d', spacegroup=bandstructure.spacegroup)
    proj_set = wberri.symmetry.projections.ProjectionsSet(projections=[proj_s, proj_p, proj_d])

    w90data = wberri.w90files.Wannier90data(
    ).from_bandstructure(bandstructure,
                         files=["mmn", "eig", "amn", "unk", "spn", "symmetrizer"],
                         write_npz_list=[],
                         read_npz_list=[],
                         seedname=os.path.join(path_tmp, prefix),
                         projections=proj_set,
                         normalize=False,
                         unk_grid=(18,) * 3,
                         irreducible=irreducible,
                         )


    if wannierise:
        w90data.wannierise(init="amn",
                           froz_min=-8,
                           froz_max=30,
                           print_progress_every=20,
                           num_iter=100,
                           conv_tol=1e-6,
                           mix_ratio_z=1.0,
                           localise=True,
                           sitesym=True,
                            )
        spreads = w90data.chk.wannier_spreads
        print(f"Wannier spreads: {repr(spreads)}")
        if spreads_ref is not None:
            assert w90data.chk.wannier_spreads == approx(spreads_ref, abs=0.01), (
                f"Wannier spreads differ from reference: {w90data.chk.wannier_spreads} != {spreads_ref}"
                f" max diff: {np.max(np.abs(w90data.chk.wannier_spreads - spreads_ref))} > 0.01"
            )


    if path_ref is not None:
        eig = w90data.get_file("eig")
        eig_ref = EIG.from_npz(os.path.join(path_ref, f"{prefix}.eig.npz"))
        eql, msg = eig.equals(eig_ref, tolerance=1e-6)
        assert eql, f"EIG files differ: {msg}"

        mmn_new = w90data.get_file("mmn")
        mmn_ref = wberri.w90files.MMN.from_npz(os.path.join(path_ref, f"{prefix}.mmn.npz"))
        mmn_ref.reorder_bk(bk_latt_new=mmn_new.bk_latt)
        eql, msg = mmn_new.equals(mmn_ref, tolerance=3e-5, check_reorder=False)
        assert eql, f"MMN files differ: {msg}"

        amn = w90data.get_file("amn")
        amn_ref = wberri.w90files.AMN.from_npz(os.path.join(path_ref, f"{prefix}.amn.npz"))  # this file is genetated with WB (because in pw2wannier the definition of radial function is different, so it does not match precisely)
        eql, msg = amn.equals(amn_ref, tolerance=1e-6)
        assert eql, f"AMN files differ: {msg}"

        spn = w90data.get_file("spn")
        spn_ref = wberri.w90files.SPN.from_npz(os.path.join(path_ref, f"{prefix}.spn.npz"))
        eql, msg = spn.equals(spn_ref, tolerance=1e-6)
        assert eql, f"SPN files differ: {msg}"

        unk_new = w90data.get_file("unk")
        unk_new.select_kpoints((0, 3))  # select only k=0 and k=3
        unk_ref = wberri.w90files.unk.UNK.from_npz(os.path.join(path_ref, f"{prefix}-kp03-red18.unk.npz"))
        eql, msg = unk_new.equals(unk_ref, tolerance=1e-6)
        assert eql, f"UNK files differ: {msg}"


spreads_Fe_spd_222 = np.array([0.93060237, 0.92001971, 0.86780841, 0.85846711, 0.86796572,
            0.85814135, 0.86796572, 0.85814135, 0.39673449, 0.37348269,
            0.36050463, 0.34991325, 0.36050463, 0.34991325, 0.39650243,
            0.37318123, 0.36050311, 0.34992226])



@pytest.mark.parametrize("irreducible", [False, True])
@pytest.mark.parametrize("irr_bs", [False, True])
def test_create_w90files_Fe_222(irreducible, irr_bs):
    if irr_bs and not irreducible:
        pytest.skip("irreducible bandstructure is incompatible with non-irreducible Wannier90 files")
    path_data = os.path.join(ROOT_DIR, "data", "Fe-222-pw")
    if irreducible:
        path_ref = None
    else:
        path_ref = path_data

    check_create_w90files_Fe(path_data=path_data,
                             spreads_ref=spreads_Fe_spd_222,
                             wannierise=True,
                             path_ref=path_ref,
                             irreducible=irreducible, irr_bs=irr_bs,
                             select_grid=None)


def test_create_w90files_Fe_444():
    path_data = os.path.join(ROOT_DIR, "data", "Fe-444-sitesym", "pwscf-irred")
    check_create_w90files_Fe(path_data=path_data,
                             spreads_ref=spreads_Fe_spd_444,
                             wannierise=True,
                             prefix='Fe',
                             path_ref=None,
                             irreducible=True, irr_bs=True,
                             select_window={'froz_min': -8, 'froz_max': 30},
                             select_grid=None)


def test_create_w90files_Fe_reduce222():
    path_data = os.path.join(ROOT_DIR, "data", "Fe-444-sitesym", "pwscf-irred")
    check_create_w90files_Fe(path_data=path_data,
                             spreads_ref=spreads_Fe_spd_222,
                             wannierise=True,
                             path_ref=None,
                             prefix='Fe',
                             irreducible=True, irr_bs=True,
                             select_grid=(2, 2, 2))


@pytest.mark.parametrize("include_TR", [True, False])
def _test_create_sawf_Fe_444(check_sawf, include_TR):
    "this test is disabled, because the necessary data is not included into repo, but need to be generated with QE"
    path_data = os.path.join(ROOT_DIR, "data", "Fe-444-sitesym", "pwscf")

    bandstructure = BandStructure(code='espresso', prefix=path_data + '/Fe', Ecut=100,
                                normalize=False, magmom=[[0, 0, 1]], include_TR=include_TR)
    sawf_new = SymmetrizerSAWF().from_irrep(bandstructure)
    pos = [[0, 0, 0]]
    proj_s = Projection(position_num=pos, orbital='s', spacegroup=bandstructure.spacegroup)
    proj_p = Projection(position_num=pos, orbital='p', spacegroup=bandstructure.spacegroup)
    proj_d = Projection(position_num=pos, orbital='d', spacegroup=bandstructure.spacegroup)
    sawf_new.set_D_wann_from_projections(projections=[proj_s, proj_p, proj_d])
    tmp_sawf_path = os.path.join(OUTPUT_DIR, f"Fe_TR={include_TR}.sawf.npz")
    sawf_new.to_npz(tmp_sawf_path)
    sawf_ref = SymmetrizerSAWF().from_npz(os.path.join(REF_DIR, "sawf", f"Fe_TR={include_TR}.sawf.npz"))
    check_sawf(sawf_new, sawf_ref)
