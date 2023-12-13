"""Test data of systems"""
import numpy as np
import pytest
import os
from common import OUTPUT_DIR, REF_DIR

properties_wcc = ['wannier_centers_cart', 'wannier_centers_reduced', 'wannier_centers_cart_wcc_phase',
                  'wannier_centers_cart_ws', 'diff_wcc_cart', 'diff_wcc_red', 'cRvec_p_wcc']


@pytest.fixture
def check_system():
    def _inner(system, name,
               properties=['num_wann', 'recip_lattice', 'real_lattice', 'nRvec', 'iRvec', 'cRvec', 'use_ws', 'periodic',
                           'use_wcc_phase', '_getFF',
                           'cRvec', 'cell_volume', 'is_phonon'] + properties_wcc,
               extra_properties=[],
               exclude_properties=[],
               precision_properties=1e-8,
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
            np.savez(os.path.join(out_dir, key + ".npz"), getattr(system, key), allow_pickle=True)
            print(" - Ok!")
        for key in matrices:
            print(f"saving {key}", end="")
            np.savez_compressed(os.path.join(out_dir, key + ".npz"), system.get_R_mat(key))
            print(" - Ok!")

        def check_property(key, prec, XX=False, sort=None, sort_axis=2):
            print(f"checking {key} prec={prec} XX={XX}", end="")
            data_ref = np.load(os.path.join(REF_DIR, "systems", name, key + ".npz"), allow_pickle=True)['arr_0']
            if XX:
                data = system.get_R_mat(key)
            else:
                data = getattr(system, key)
            data = np.array(data)
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
                           f"absolute difference of {diff} greater than the required precision {req_precision}\n"
                           f"wrong elements {n_missed} out of {data.size}")
                if XX:
                    if n_missed < data.size / 10:
                        err_msg += "\n" + ("\n".join(
                            f"{i} | {system.iRvec[i[2]]} | {data[i]} | {data_ref[i]} | {abs(data[i] - data_ref[i])}"
                            for i in zip(*missed)) + "\n\n")
                    else:
                        all_i = np.where(abs(data - data_ref) >= -np.Inf)
                        err_msg += "\n" + ("\n".join(
                            f"{i} | {system.iRvec[i[2]]} | {data[i]} | {data_ref[i]} | {abs(data[i] - data_ref[i])} | {data[i] / data_ref[i]} | {abs(data[i] - data_ref[i]) < req_precision} "
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
            if key in ['iRvec', 'cRvec']:
                check_property(key, precision_properties, XX=False, sort=sort_R, sort_axis=0)
            elif key in ['cRvec_p_wcc']:
                check_property(key, precision_properties, XX=False, sort=sort_R, sort_axis=2)
            else:
                check_property(key, precision_properties, XX=False)
        for key in matrices:
            check_property(key, precision_matrix_elements, XX=True, sort=sort_R)

    return _inner


def test_system_Fe_W90(check_system, system_Fe_W90):
    check_system(
        system_Fe_W90, "Fe_W90",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA']
    )


def test_system_Fe_W90_wcc(check_system, system_Fe_W90_wcc):
    check_system(
        system_Fe_W90_wcc, "Fe_W90_wcc",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_Fe_W90_sparse(check_system, system_Fe_W90_sparse):
    check_system(
        system_Fe_W90_sparse, "Fe_W90_sparse",
        exclude_properties=properties_wcc,
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA']
    )


def test_system_Fe_sym_W90_old(check_system, system_Fe_sym_W90_old):
    check_system(
        system_Fe_sym_W90_old, "Fe_sym_W90",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_Fe_sym_W90(check_system, system_Fe_sym_W90):
    check_system(
        system_Fe_sym_W90, "Fe_sym_W90",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        sort_iR=True
    )


def test_system_Fe_W90_proj_set_spin(check_system, system_Fe_W90_proj_set_spin):
    check_system(
        system_Fe_W90_proj_set_spin, "Fe_W90_proj_set_spin",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_Fe_W90_proj(check_system, system_Fe_W90_proj):
    check_system(
        system_Fe_W90_proj, "Fe_W90_proj",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR']
    )


def test_system_Fe_W90_disentngle(check_system, system_Fe_W90_disentangle):
    check_system(
        system_Fe_W90_disentangle, "Fe_W90_disentangle",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA']
    )


def test_system_GaAs_W90(check_system, system_GaAs_W90):
    check_system(
        system_GaAs_W90, "GaAs_W90",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_GaAs_W90_wcc(check_system, system_GaAs_W90_wcc):
    check_system(
        system_GaAs_W90_wcc, "GaAs_W90_wcc",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS']
    )


def test_system_GaAs_tb(check_system, system_GaAs_tb):
    check_system(
        system_GaAs_tb, "GaAs_tb",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA']
    )


def test_system_GaAs_sym_tb_old(check_system, system_GaAs_sym_tb_old):
    check_system(
        system_GaAs_sym_tb_old, "GaAs_sym_tb",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA']
    )


def test_system_GaAs_sym_tb(check_system, system_GaAs_sym_tb):
    check_system(
        system_GaAs_sym_tb, "GaAs_sym_tb",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA'],
        sort_iR=True,
        suffix="new"
    )


def test_system_GaAs_tb_wcc(check_system, system_GaAs_tb_wcc):
    check_system(
        system_GaAs_tb_wcc, "GaAs_tb_wcc",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA']
    )


def test_system_GaAs_tb_wcc_ws(check_system, system_GaAs_tb_wcc_ws):
    check_system(
        system_GaAs_tb_wcc_ws, "GaAs_tb_wcc_ws",
        extra_properties=['wannier_centers_cart_auto', 'mp_grid'],
        matrices=['Ham', 'AA']
    )


def test_system_Haldane_TBmodels(check_system, system_Haldane_TBmodels):
    check_system(
        system_Haldane_TBmodels, "Haldane", suffix="TBmodels",
        matrices=['Ham', 'AA']
    )


def test_system_Haldane_TBmodels_internal(check_system, system_Haldane_TBmodels_internal):
    check_system(
        system_Haldane_TBmodels_internal, "Haldane", suffix="TBmodels_internal",
        matrices=['Ham']
    )


def test_system_Haldane_PythTB(check_system, system_Haldane_PythTB):
    check_system(
        system_Haldane_PythTB, "Haldane", suffix="PythTB",
        matrices=['Ham', 'AA']
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


def test_system_Fe_FPLO(check_system, system_Fe_FPLO):
    check_system(
        system_Fe_FPLO, "Fe_FPLO",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA', 'SS']
    )


def test_system_Fe_FPLO_wcc(check_system, system_Fe_FPLO_wcc):
    check_system(
        system_Fe_FPLO_wcc, "Fe_FPLO_wcc",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA', 'SS']
    )


def test_system_CuMnAs_2d_broken(check_system, system_CuMnAs_2d_broken):
    check_system(
        system_CuMnAs_2d_broken, "CuMnAs_2d_broken",
        matrices=['Ham']
    )


def test_system_Te_ASE(check_system, system_Te_ASE):
    check_system(
        system_Te_ASE, "Te_ASE",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA']
    )


def test_system_Te_ASE_wcc(check_system, system_Te_ASE_wcc):
    check_system(
        system_Te_ASE_wcc, "Te_ASE_wcc",
        extra_properties=['wannier_centers_cart_auto'],
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


def test_system_Mn3Sn_sym_tb_old(check_system, system_Mn3Sn_sym_tb_old):
    check_system(
        system_Mn3Sn_sym_tb_old, "Mn3Sn_sym_tb",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA']
    )


def test_system_Mn3Sn_sym_tb(check_system, system_Mn3Sn_sym_tb):
    check_system(
        system_Mn3Sn_sym_tb, "Mn3Sn_sym_tb",
        extra_properties=['wannier_centers_cart_auto'],
        matrices=['Ham', 'AA'],
        sort_iR=True
    )

# TODO : add tests for kp systems ?
