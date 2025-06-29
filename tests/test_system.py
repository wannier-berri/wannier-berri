"""Test data of systems"""
import numpy as np
import pytest
import os
from .common import OUTPUT_DIR, REF_DIR

from .common_systems import model_1d_pythtb
import wannierberri as wberri

from wannierberri.system.system_R import System_R
from wannierberri.system.system_tb import System_tb

properties_wcc = ['wannier_centers_cart', 'wannier_centers_red']


@pytest.fixture
def check_system():
    def _inner(system, name,
               properties=['num_wann', 'recip_lattice', 'real_lattice', 'periodic',
                           'cell_volume', 'is_phonon',
                           ] + properties_wcc + ['nRvec', 'iRvec'],
               extra_properties=[],
               exclude_properties=[],
               precision_properties=1e-8,
               extra_precision={},  # for some properties we need different precision
               matrices=[],
               precision_matrix_elements=1e-7,
               suffix="",
               sort_iR=True,
               legacy=False
               ):
        if len(suffix) > 0:
            suffix = "_" + suffix
        out_dir = os.path.join(OUTPUT_DIR, 'systems', name + suffix)
        os.makedirs(out_dir, exist_ok=True)

        print(f"System {name} has the following attriburtes : {sorted(system.__dict__.keys())}")
        print(f"System {name} has the following matrices : {sorted(system._XX_R.keys())}")
        other_prop = sorted(list([p for p in set(dir(system)) - set(system.__dict__.keys()) if not p.startswith("__")]))
        print(f"System {name} additionaly has the following properties : {other_prop}")
        properties = [p for p in properties + extra_properties if p not in exclude_properties]
        # First save the system data, to produce reference data

        # we save each property as separate file, so that if in future we add more properties, we do not need to
        # rewrite the old files, so that the changes in a PR will be clearly visible
        for key in properties:
            print(f"saving {key}", end="")
            if key == 'iRvec':
                val = system.rvec.iRvec
            elif key == 'nRvec':
                val = system.rvec.nRvec
            else:
                val = getattr(system, key)
            np.savez(os.path.join(out_dir, key + ".npz"), val)
            print(" - Ok!")
        for key in matrices:
            print(f"saving {key}", end="")
            np.savez_compressed(os.path.join(out_dir, key + ".npz"), system.get_R_mat(key))
            print(" - Ok!")

        def check_property(key, prec, XX=False, sort=None, print_missed=False, legacy=False):
            if key in exclude_properties:
                return
            print(f"checking {key} prec={prec} XX={XX}", end="")
            data_ref = np.load(os.path.join(REF_DIR, "systems", name, key + ".npz"))['arr_0']
            if XX:
                data = system.get_R_mat(key)
                if legacy:
                    data_ref = data_ref.transpose((2, 0, 1) + tuple(i for i in range(3, data.ndim)))
            elif key == 'nRvec':
                data = system.rvec.nRvec
            elif key == 'iRvec':
                data = system.rvec.iRvec
            else:
                data = getattr(system, key)
            data = np.array(data)
            print(f"data.shape = {data.shape}, data_ref.shape = {data_ref.shape}", end="")
            print("sort = ", sort)
            if sort is not None:
                data_ref = data_ref[sort]
            if data.dtype == bool:
                data = np.array(data, dtype=int)
                data_ref = np.array(data_ref, dtype=int)
            if hasattr(data_ref, 'shape'):
                assert data.shape == data_ref.shape, f"{key} has the wrong shape {data.shape}, should be {data_ref.shape}"
            if prec < 0:
                req_precision = -prec * (abs(data_ref))
            else:
                req_precision = prec
            if not data == pytest.approx(data_ref, abs=req_precision):
                diff = abs(data - data_ref).max()
                missed = np.where(abs(data - data_ref) > req_precision)
                n_missed = len(missed[0])
                err_msg = (f"matrix elements {key} for system {name} give an "
                           f"absolute difference of {diff} greater than the required precision {req_precision}"
                           f"wrong elements {n_missed} out of {data.size}")
                if XX or print_missed:
                    if n_missed < data.size / 10:
                        err_msg += "\n" + ("\n".join(
                            f"{i} | {system.rvec.iRvec[i[2]]} | {data[i]} | {data_ref[i]} | {abs(data[i] - data_ref[i])}"
                            for i in zip(*missed)) + "\n\n")
                    else:
                        all_i = np.where(abs(data - data_ref) >= -np.inf)
                        print(f"all_i = {repr(all_i)}")
                        ratio = np.zeros(data_ref.shape)
                        select = abs(data_ref) > 1e-12
                        ratio[select] = data[select] / data_ref[select]
                        ratio[np.logical_not(select)] = None
                        if XX:
                            err_msg += "\n" + ("\n".join(
                                f"{i} | {system.rvec.iRvec[i[2]]} | {data[i]} | {data_ref[i]} | {abs(data[i] - data_ref[i])} | {ratio[i]} | {abs(data[i] - data_ref[i]) < req_precision} "
                                for i in zip(*all_i)) + "\n\n")
                            XX_R_sumR = data.sum(axis=2)
                            XX_R_sumR_ref = data_ref.sum(axis=2)
                            err_msg += f"\n the control sum differs by {XX_R_sumR.sum() - XX_R_sumR_ref.sum()} \n"
                            err_msg += f"maximal element-wise difference {abs(XX_R_sumR - XX_R_sumR_ref).max()} \n"
                        else:
                            err_msg += "\n" + "\n".join(f"{a} | {data[a]}  |  {data[b]} " for a, b in zip(*all_i))

                elif key in properties_wcc:
                    err_msg += f"new data : {data} \n ref data : {data_ref}"
                raise ValueError(err_msg)

            print(" - Ok!")

        if sort_iR:
            iRvec_ref = np.load(os.path.join(REF_DIR, "systems", name, "iRvec.npz"), allow_pickle=True)['arr_0'].tolist()
            iRvec_new = system.rvec.iRvec.tolist()
            try:
                assert len(iRvec_ref) == len(iRvec_new), f"iRvec_ref and iRvec_new have different lengths {len(iRvec_ref)} {len(iRvec_new)}"
                # assert len(set(iRvec_ref)) == len(iRvec_ref), f"iRvec_ref has duplicates {iRvec_ref}"
                # assert len(set(iRvec_new)) == len(iRvec_new), f"iRvec_new has duplicates {iRvec_new}"
                sort_R = [iRvec_ref.index(iR) for iR in iRvec_new]
            except (ValueError, AssertionError) as e:
                print(f"iRvec_ref : {iRvec_ref}")
                print(f"iRvec_new : {iRvec_new}")
                print(f"cRvec_new_lengths : {np.linalg.norm(system.rvec.cRvec, axis=1)}")
                cRvec_ref = np.load(os.path.join(REF_DIR, "systems", name, "cRvec.npz"), allow_pickle=True)["arr_0"]
                print(f"cRvec_ref_length : {np.linalg.norm(cRvec_ref, axis=1)}")
                raise ValueError(f"{e} : \n iRvec_ref and iRvec_new have different values {iRvec_ref} {iRvec_new}") from e
        else:
            sort_R = None

        for key in properties:
            if key in extra_precision:
                prec_loc = extra_precision[key]
            else:
                prec_loc = precision_properties
            if key in ['iRvec', 'cRvec']:
                check_property(key, prec_loc, XX=False, sort=sort_R, print_missed=True)
            elif key in ['cRvec_p_wcc']:
                check_property(key, prec_loc, XX=False, sort=sort_R, print_missed=True)
            else:
                check_property(key, prec_loc, XX=False)
        for key in matrices:
            print(f"checking matrix {key}", end="")
            if key in extra_precision:
                prec_loc = extra_precision[key]
            else:
                prec_loc = precision_matrix_elements
            check_property(key, prec_loc, XX=True, sort=sort_R, print_missed=True, legacy=legacy)
            print(f"check matrix {key} - Ok!")

    return _inner


def test_system_Fe_W90(check_system, system_Fe_W90):
    check_system(
        system_Fe_W90, "Fe_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SHA', 'SA'],
        legacy=True,
    )


def test_system_Fe_W90_npz(check_system, system_Fe_W90_npz):
    check_system(
        system_Fe_W90_npz, "Fe_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA'],
        suffix="_npz",
        legacy=True,
    )


def test_system_Fe_WB_irreducible(check_system, system_Fe_WB_irreducible):
    check_system(
        system_Fe_WB_irreducible, "Fe_WB_irreducible",
        matrices=['Ham', 'AA', 'SS', 'SR', 'SH', 'SHR'],
        legacy=False,
    )


def test_system_Fe_W90_sparse(check_system, system_Fe_W90_sparse):
    check_system(
        system_Fe_W90_sparse, "Fe_W90_sparse",
        exclude_properties=properties_wcc,
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA'],
        legacy=True,
    )


def test_system_Fe_sym_W90(check_system, system_Fe_sym_W90):
    check_system(
        system_Fe_sym_W90, "Fe_sym_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        sort_iR=True,
        legacy=True,
    )


def test_system_Fe_sym_W90_TR(check_system, system_Fe_sym_W90_TR):
    check_system(
        system_Fe_sym_W90_TR, "Fe_sym_W90_TR",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        sort_iR=True,
        legacy=False,
    )


def test_system_Fe_W90_proj_set_spin(check_system, system_Fe_W90_proj_set_spin):
    check_system(
        system_Fe_W90_proj_set_spin, "Fe_W90_proj_set_spin",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        legacy=True,
    )


def test_system_Fe_W90_proj(check_system, system_Fe_W90_proj):
    check_system(
        system_Fe_W90_proj, "Fe_W90_proj",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR'],
        legacy=True,
    )



def test_system_Fe_sym_W90_interpolate(check_system, system_Fe_sym_W90,
                                       system_Fe_sym_W90_TR):
    interpolator = wberri.system.interpolate.SystemInterpolator(system0=system_Fe_sym_W90,
                                                                system1=system_Fe_sym_W90_TR)
    system_Fe_sym_W90_interpolate = interpolator.interpolate(0.4)


    check_system(
        system_Fe_sym_W90_interpolate, "Fe_sym_W90_interpolate_04",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        sort_iR=True,
        legacy=False,
    )


def test_system_GaAs_W90(check_system, system_GaAs_W90):
    check_system(
        system_GaAs_W90, "GaAs_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        legacy=True,
    )


def test_system_GaAs_W90_JM(check_system, system_GaAs_W90_JM):
    check_system(
        system_GaAs_W90_JM, "GaAs_W90_JM",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SH', 'SA', 'SHA', 'OO', 'GG'],
        legacy=True,
    )


def test_system_GaAs_tb(check_system, system_GaAs_tb):
    check_system(
        system_GaAs_tb, "GaAs_tb",
        matrices=['Ham', 'AA'],
        sort_iR=True,
        legacy=True,
        # extra_precision={'Ham': 1e-5, 'AA': 5e-4}
    )


def test_system_GaAs_sym_tb(check_system, system_GaAs_sym_tb):
    check_system(
        system_GaAs_sym_tb, "GaAs_sym_tb",
        matrices=['Ham', 'AA'],
        sort_iR=True,
        legacy=True,
    )



def test_system_GaAs_tb_noAA(check_system, system_GaAs_tb_noAA):
    check_system(
        system_GaAs_tb_noAA, "GaAs_tb",
        matrices=['Ham',],
        legacy=True,
    )


def test_system_GaAs_tb_save_load(check_system, system_GaAs_tb):
    name = "GaAs_tb_save"
    path = os.path.join(OUTPUT_DIR, name)
    system_GaAs_tb.save_npz(path)
    system = System_R()
    system.load_npz(path, load_all_XX_R=True)
    check_system(
        system, "GaAs_tb",
        suffix="save-load",
        matrices=['Ham', 'AA'],
        legacy=True,
    )


def test_system_Si_W90_JM(check_system, system_Si_W90_JM):
    check_system(
        system_Si_W90_JM, "Si_W90_JM",
        matrices=['Ham', 'AA', 'BB', 'CC', 'GG', 'OO'],
        sort_iR=True,
        legacy=True,
    )


def test_system_Si_W90_JM_sym(check_system, system_Si_W90_JM_sym):
    check_system(
        system_Si_W90_JM_sym, "Si_W90_JM_sym",
        matrices=['Ham', 'AA', 'BB', 'CC', 'GG', 'OO'],
        sort_iR=True,
        legacy=True,
    )


def test_system_Si_W90(check_system, system_Si_W90):
    check_system(
        system_Si_W90, "Si_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'GG', 'OO'],
        sort_iR=True,
        legacy=True,
    )


def test_system_Si_W90_sym(check_system, system_Si_W90_sym):
    check_system(
        system_Si_W90_sym, "Si_W90_sym",
        matrices=['Ham', 'AA', 'BB', 'CC', 'GG', 'OO'],
        sort_iR=True,
        legacy=True,
    )



def test_system_Haldane_TBmodels(check_system, system_Haldane_TBmodels):
    check_system(
        system_Haldane_TBmodels, "Haldane", suffix="TBmodels",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_Haldane_PythTB(check_system, system_Haldane_PythTB):
    check_system(
        system_Haldane_PythTB, "Haldane", suffix="PythTB",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_KaneMele_odd_PythTB(check_system, system_KaneMele_odd_PythTB):
    check_system(
        system_KaneMele_odd_PythTB, "KaneMele", suffix="PythTB",
        matrices=['Ham', 'SS'],
        legacy=True,
    )


def test_system_Chiral_OSD(check_system, system_Chiral_OSD):
    check_system(
        system_Chiral_OSD, "Chiral_OSD",
        matrices=['Ham', 'SS'],
        legacy=True,
    )


def test_system_Chiral_left(check_system, system_Chiral_left):
    check_system(
        system_Chiral_left, "Chiral_left",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_Chiral_left_TR(check_system, system_Chiral_left_TR):
    check_system(
        system_Chiral_left_TR, "Chiral_left_TR",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_Chiral_right(check_system, system_Chiral_right):
    check_system(
        system_Chiral_right, "Chiral_right",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_Fe_FPLO(check_system, system_Fe_FPLO):
    check_system(
        system_Fe_FPLO, "Fe_FPLO",
        matrices=['Ham', 'SS'],
        legacy=True,
    )


def test_system_CuMnAs_2d_broken(check_system, system_CuMnAs_2d_broken):
    check_system(
        system_CuMnAs_2d_broken, "CuMnAs_2d_broken",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_Te_ASE(check_system, system_Te_ASE):
    check_system(
        system_Te_ASE, "Te_ASE2",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_Te_sparse(check_system, system_Te_sparse):
    check_system(
        system_Te_sparse, "Te_sparse",
        matrices=['Ham'],
        sort_iR=True,
        legacy=True,
    )


def test_system_Phonons_Si(check_system, system_Phonons_Si):
    check_system(
        system_Phonons_Si, "Phonons_Si",
        matrices=['Ham'],
        sort_iR=True,
        legacy=True,
    )


def test_system_Phonons_GaAs(check_system, system_Phonons_GaAs):
    check_system(
        system_Phonons_GaAs, "Phonons_GaAs",
        matrices=['Ham'],
        legacy=True,
    )


def test_system_Mn3Sn_sym_tb(check_system, system_Mn3Sn_sym_tb):
    check_system(
        system_Mn3Sn_sym_tb, "Mn3Sn_sym_tb",
        matrices=['Ham', 'AA'],
        sort_iR=True,
        legacy=True,
    )


def test_system_pythtb_spinor():
    model1, model2 = model_1d_pythtb()
    k = np.linspace(0, 1, 20, endpoint=False)
    e1 = model1.solve_all(k)
    e2 = model1.solve_all(k)
    assert e1 == pytest.approx(e2), ("models defined in pythtb using spinors and in scalar way gave energies"
                                     f"different by {abs(e1 - e2).max()}")

    for model, comment, e in (model2, "in a scalar way", e2), (model1, "using spinors", e1), :
        system = wberri.system.System_PythTB(model)
        grid = wberri.Grid(system=system, NKFFT=[20, 1, 1], NKdiv=1)
        datak = wberri.data_K.get_data_k(system=system, dK=[0, 0, 0], grid=grid)
        ew = datak.E_K.T
        assert e == pytest.approx(ew), ("System gave eigenvalues different from the model "
                                      f"defined in pythtb {comment}"
                                      f"maximal difference: {abs(e - ew).max()}")

# TODO : add tests for kp systems ?


def test_system_random(check_system, system_random):
    system = system_random
    assert system.wannier_centers_cart.shape == (system.num_wann, 3)
    assert system.get_R_mat('AA')[system.rvec.iR0].diagonal().imag == pytest.approx(0)
    system.save_npz(os.path.join(OUTPUT_DIR, "randomsys"))



def test_system_random_load_bare(check_system, system_random_load_bare):
    check_system(
        system_random_load_bare, "random_bare",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'GG', 'OO'],
        sort_iR=False,
        legacy=True,
    )


def test_system_random_to_tb_back(check_system, system_random_GaAs_load_bare):
    path = os.path.join(OUTPUT_DIR, "random_GaAS_tb")
    system_random_GaAs_load_bare.to_tb_file(path)
    system_tb = System_tb(path, berry=True)
    print(system_tb.wannier_centers_cart)
    print(system_random_GaAs_load_bare.wannier_centers_cart)

    check_system(
        system_tb, "random_GaAs_bare",
        suffix="_tb",
        matrices=['Ham', 'AA'],
        sort_iR=False,
        legacy=True,
    )


def test_system_random_GaAs(check_system, system_random_GaAs):
    system = system_random_GaAs
    assert system.wannier_centers_cart.shape == (system.num_wann, 3)
    assert system.num_wann == 16
    assert system.get_R_mat('AA')[system.rvec.iR0].diagonal() == pytest.approx(0)
    system.save_npz(os.path.join(OUTPUT_DIR, "randomsys_GaAs"))


def test_system_random_GaAs_load_bare(check_system, system_random_GaAs_load_bare):
    check_system(
        system_random_GaAs_load_bare, "random_GaAs_bare",
        matrices=['Ham', 'AA', 'SS'],
        sort_iR=False,
        legacy=True,
    )


def test_system_random_GaAs_load(check_system, system_random_GaAs_load_bare):
    check_system(
        system_random_GaAs_load_bare, "random_GaAs_bare",
        matrices=['Ham', 'AA', 'SS'],
        sort_iR=False,
        legacy=True,
    )


def test_system_random_GaAs_load_sym(check_system, system_random_GaAs_load_sym):
    check_system(
        system_random_GaAs_load_sym, "random_GaAs_sym",
        matrices=['Ham', 'AA', 'SS'],
        sort_iR=True,
        legacy=True,
    )
