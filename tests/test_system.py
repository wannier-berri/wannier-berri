"""Test data of systems"""
import numpy as np
import pytest
import os
from common import OUTPUT_DIR, REF_DIR

from common_systems import model_1d_pythtb
import wannierberri as wberri

from wannierberri.system.system_R import System_R
from wannierberri.system.system_tb import System_tb

properties_wcc = ['wannier_centers_cart', 'wannier_centers_reduced', 'wannier_centers_cart_wcc_phase',
                  'diff_wcc_cart', 'diff_wcc_red']  # , 'cRvec_p_wcc']


@pytest.fixture
def check_system():
    def _inner(system, name,
               properties=['num_wann', 'recip_lattice', 'real_lattice', 'use_ws', 'periodic',
                           'use_wcc_phase', 'cell_volume', 'is_phonon',
                           'nRvec', 'iRvec', 'cRvec'] + properties_wcc,
               extra_properties=[],
               exclude_properties=[],
               precision_properties=1e-8,
               extra_precision={},  # for some properties we need different precision
               matrices=[],
               precision_matrix_elements=1e-7,
               suffix="",
               sort_iR=False
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
            np.savez(os.path.join(out_dir, key + ".npz"), getattr(system, key))
            print(" - Ok!")
        for key in matrices:
            print(f"saving {key}", end="")
            np.savez_compressed(os.path.join(out_dir, key + ".npz"), system.get_R_mat(key))
            print(" - Ok!")

        def check_property(key, prec, XX=False, sort=None, sort_axis=2, print_missed=False):
            print(f"checking {key} prec={prec} XX={XX}", end="")
            data_ref = np.load(os.path.join(REF_DIR, "systems", name, key + ".npz"))['arr_0']
            if XX:
                data = system.get_R_mat(key)
            else:
                data = getattr(system, key)
            data = np.array(data)
            print("sort = ", sort)
            if sort is not None:
                if sort_axis == 0:
                    data_ref = data_ref[sort]
                elif sort_axis == 2:
                    data_ref = data_ref[:, :, sort]
                else:
                    raise ValueError(f"sorting only along axis 0 or 2, but {sort_axis} is requested")
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
                            f"{i} | {system.iRvec[i[2]]} | {data[i]} | {data_ref[i]} | {abs(data[i] - data_ref[i])}"
                            for i in zip(*missed)) + "\n\n")
                    else:
                        all_i = np.where(abs(data - data_ref) >= -np.Inf)
                        ratio = np.zeros(data_ref.shape)
                        select = abs(data_ref) > 1e-12
                        ratio[select] = data[select] / data_ref[select]
                        ratio[np.logical_not(select)] = None
                        err_msg += "\n" + ("\n".join(
                            f"{i} | {system.iRvec[i[2]]} | {data[i]} | {data_ref[i]} | {abs(data[i] - data_ref[i])} | {ratio[i]} | {abs(data[i] - data_ref[i]) < req_precision} "
                            for i in zip(*all_i)) + "\n\n")
                raise ValueError(err_msg)

            print(" - Ok!")

        if sort_iR:
            iRvec_ref = np.load(os.path.join(REF_DIR, "systems", name, "iRvec.npz"), allow_pickle=True)[
                'arr_0'].tolist()
            iRvec_new = system.iRvec.tolist()
            sort_R = [iRvec_ref.index(iR) for iR in iRvec_new]
        else:
            sort_R = None

        for key in properties:
            if key in extra_precision:
                prec_loc = extra_precision[key]
            else:
                prec_loc = precision_properties
            if key in ['iRvec', 'cRvec']:
                check_property(key, prec_loc, XX=False, sort=sort_R, sort_axis=0, print_missed=True)
            elif key in ['cRvec_p_wcc']:
                check_property(key, prec_loc, XX=False, sort=sort_R, sort_axis=2, print_missed=True)
            else:
                check_property(key, prec_loc, XX=False)
        for key in matrices:
            if key in extra_precision:
                prec_loc = extra_precision[key]
            else:
                prec_loc = precision_matrix_elements
            check_property(key, prec_loc, XX=True, sort=sort_R, print_missed=True)

    return _inner


def test_system_Fe_W90(check_system, system_Fe_W90):
    check_system(
        system_Fe_W90, "Fe_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA']
    )


def test_system_Fe_W90_npz(check_system, system_Fe_W90_npz):
    check_system(
        system_Fe_W90_npz, "Fe_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA'],
        suffix="_npz"
    )


def test_system_Fe_W90_wcc(check_system, system_Fe_W90_wcc):
    check_system(
        system_Fe_W90_wcc, "Fe_W90_wcc",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
    )


def test_system_Fe_W90_sparse(check_system, system_Fe_W90_sparse):
    check_system(
        system_Fe_W90_sparse, "Fe_W90_sparse",
        exclude_properties=properties_wcc,
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA']
    )


def test_system_Fe_sym_W90(check_system, system_Fe_sym_W90_wcc):
    check_system(
        system_Fe_sym_W90_wcc, "Fe_sym_W90_wcc",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        sort_iR=True
    )


def test_system_Fe_W90_proj_set_spin(check_system, system_Fe_W90_proj_set_spin):
    check_system(
        system_Fe_W90_proj_set_spin, "Fe_W90_proj_set_spin",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_Fe_W90_proj(check_system, system_Fe_W90_proj):
    check_system(
        system_Fe_W90_proj, "Fe_W90_proj",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR']
    )



def test_system_GaAs_W90(check_system, system_GaAs_W90):
    check_system(
        system_GaAs_W90, "GaAs_W90",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_GaAs_W90_wcc(check_system, system_GaAs_W90_wcc):
    check_system(
        system_GaAs_W90_wcc, "GaAs_W90_wcc",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_GaAs_W90_wccFD(check_system, system_GaAs_W90_wccFD):
    check_system(
        system_GaAs_W90_wccFD, "GaAs_W90_wccFD",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'OO', 'GG']
    )


def test_system_GaAs_W90_wccJM(check_system, system_GaAs_W90_wccJM):
    check_system(
        system_GaAs_W90_wccJM, "GaAs_W90_wccJM",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SH', 'SA', 'SHA', 'OO', 'GG'],
    )


def test_system_GaAs_tb(check_system, system_GaAs_tb):
    check_system(
        system_GaAs_tb, "GaAs_tb",
        matrices=['Ham', 'AA']
    )


def test_system_GaAs_sym_tb(check_system, system_GaAs_sym_tb_wcc):
    check_system(
        system_GaAs_sym_tb_wcc, "GaAs_sym_tb_wcc",
        matrices=['Ham', 'AA'],
        sort_iR=True,
        suffix="new"
    )


def test_system_GaAs_tb_wcc(check_system, system_GaAs_tb_wcc):
    check_system(
        system_GaAs_tb_wcc, "GaAs_tb_wcc",
        matrices=['Ham', 'AA']
    )


def test_system_GaAs_tb_wcc_ws(check_system, system_GaAs_tb_wcc_ws):
    check_system(
        system_GaAs_tb_wcc_ws, "GaAs_tb_wcc_ws",
        matrices=['Ham', 'AA']
    )


def test_system_GaAs_tb_wcc_ws_save_load(check_system, system_GaAs_tb_wcc_ws):
    name = "GaAs_tb_wcc_ws_save"
    path = os.path.join(OUTPUT_DIR, name)
    system_GaAs_tb_wcc_ws.save_npz(path)
    system = System_R()
    system.load_npz(path, load_all_XX_R=True)
    check_system(
        system, "GaAs_tb_wcc_ws",
        suffix="save-load",
        matrices=['Ham', 'AA']
    )


def test_system_Si_W90_JM(check_system, system_Si_W90_JM):
    check_system(
        system_Si_W90_JM, "Si_W90_JM",
        matrices=['Ham', 'AA', 'BB', 'CC', 'GG', 'OO']
    )


def test_system_Si_W90_wccFD(check_system, system_Si_W90_wccFD):
    check_system(
        system_Si_W90_wccFD, "Si_W90_wccFD",
        matrices=['Ham', 'AA', 'BB', 'CC', 'GG', 'OO']
    )



def test_system_Haldane_TBmodels(check_system, system_Haldane_TBmodels):
    check_system(
        system_Haldane_TBmodels, "Haldane", suffix="TBmodels",
        matrices=['Ham']
    )


def test_system_Haldane_PythTB(check_system, system_Haldane_PythTB):
    check_system(
        system_Haldane_PythTB, "Haldane", suffix="PythTB",
        matrices=['Ham']
    )


def test_system_KaneMele_odd_PythTB(check_system, system_KaneMele_odd_PythTB):
    check_system(
        system_KaneMele_odd_PythTB, "KaneMele", suffix="PythTB",
        matrices=['Ham', 'SS']
    )


def test_system_Chiral_OSD(check_system, system_Chiral_OSD):
    check_system(
        system_Chiral_OSD, "Chiral_OSD",
        matrices=['Ham', 'SS']
    )


def test_system_Chiral_left(check_system, system_Chiral_left):
    check_system(
        system_Chiral_left, "Chiral_left",
        matrices=['Ham']
    )


def test_system_Chiral_left_TR(check_system, system_Chiral_left_TR):
    check_system(
        system_Chiral_left_TR, "Chiral_left_TR",
        matrices=['Ham']
    )


def test_system_Chiral_right(check_system, system_Chiral_right):
    check_system(
        system_Chiral_right, "Chiral_right",
        matrices=['Ham']
    )


def test_system_Fe_FPLO_wcc(check_system, system_Fe_FPLO_wcc):
    check_system(
        system_Fe_FPLO_wcc, "Fe_FPLO_wcc",
        matrices=['Ham', 'SS']
    )


def test_system_Fe_FPLO_wcc_ws(check_system, system_Fe_FPLO_wcc_ws):
    check_system(
        system_Fe_FPLO_wcc_ws, "Fe_FPLO_wcc_ws",
        matrices=['Ham', 'SS']
    )


def test_system_CuMnAs_2d_broken(check_system, system_CuMnAs_2d_broken):
    check_system(
        system_CuMnAs_2d_broken, "CuMnAs_2d_broken",
        matrices=['Ham']
    )


def test_system_Te_ASE_wcc(check_system, system_Te_ASE_wcc):
    check_system(
        system_Te_ASE_wcc, "Te_ASE_wcc",
        matrices=['Ham']
    )


def test_system_Te_sparse(check_system, system_Te_sparse):
    check_system(
        system_Te_sparse, "Te_sparse",
        matrices=['Ham'],
        sort_iR=True
    )


def test_system_Phonons_Si(check_system, system_Phonons_Si):
    check_system(
        system_Phonons_Si, "Phonons_Si",
        matrices=['Ham']
    )


def test_system_Phonons_GaAs(check_system, system_Phonons_GaAs):
    check_system(
        system_Phonons_GaAs, "Phonons_GaAs",
        matrices=['Ham']
    )


def test_system_Mn3Sn_sym_tb(check_system, system_Mn3Sn_sym_tb_wcc):
    check_system(
        system_Mn3Sn_sym_tb_wcc, "Mn3Sn_sym_tb_wcc",
        matrices=['Ham', 'AA'],
        sort_iR=True
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
    assert system.get_R_mat('AA')[:, :, system.iR0].diagonal().imag == pytest.approx(0)
    system.save_npz(os.path.join(OUTPUT_DIR, "randomsys"))



def test_system_random_load_bare(check_system, system_random_load_bare):
    check_system(
        system_random_load_bare, "random_bare",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'GG', 'OO'],
        sort_iR=False
    )


def test_system_random_to_tb_back(check_system, system_random_GaAs_load_bare):
    path = os.path.join(OUTPUT_DIR, "random_GaAS_tb")
    system_random_GaAs_load_bare.to_tb_file(path)
    system_tb = System_tb(path, berry=True, use_wcc_phase=True)
    print(system_tb.wannier_centers_cart)
    print(system_random_GaAs_load_bare.wannier_centers_cart)

    check_system(
        system_tb, "random_GaAs_bare",
        suffix="_tb",
        matrices=['Ham', 'AA'],
        sort_iR=False
    )


def test_system_random_GaAs(check_system, system_random_GaAs):
    system = system_random_GaAs
    assert system.wannier_centers_cart.shape == (system.num_wann, 3)
    assert system.num_wann == 16
    assert system.get_R_mat('AA')[:, :, system.iR0].diagonal() == pytest.approx(0)
    system.save_npz(os.path.join(OUTPUT_DIR, "randomsys_GaAs"))


def test_system_random_GaAs_load_bare(check_system, system_random_GaAs_load_bare):
    check_system(
        system_random_GaAs_load_bare, "random_GaAs_bare",
        matrices=['Ham', 'AA', 'SS'],
        sort_iR=False
    )


def test_system_random_GaAs_load_ws(check_system, system_random_GaAs_load_ws):
    check_system(
        system_random_GaAs_load_ws, "random_GaAs_ws",
        matrices=['Ham', 'AA', 'SS'],
        sort_iR=False
    )


def test_system_random_GaAs_load_ws_sym(check_system, system_random_GaAs_load_ws_sym):
    check_system(
        system_random_GaAs_load_ws_sym, "random_GaAs_ws_sym",
        matrices=['Ham', 'AA', 'SS'],
        sort_iR=False
    )
