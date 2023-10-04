"""Test data of systems"""
import numpy as np
import pytest, os
from common import OUTPUT_DIR, REF_DIR

properties_wcc = ['wannier_centers_cart', 'wannier_centers_reduced','wannier_centers_cart_wcc_phase','wannier_centers_cart_ws', 'diff_wcc_cart', 'diff_wcc_red','cRvec_p_wcc']

@pytest.fixture
def check_system():
    def _inner( system,name,
                properties=['num_wann','recip_lattice','real_lattice','nRvec','iRvec','cRvec','iR0','use_ws', 'periodic',
                'use_wcc_phase','_getFF',
                'cRvec',  'cell_volume','is_phonon'],
                extra_properties=[],
                exclude_properties=[],
                precision_properties=1e-8,
                matrices=[],
                precision_matrix_elements=1e-7,
                suffix=""
               ):
        out_dir = os.path.join(OUTPUT_DIR, 'systems',name+suffix)
        os.makedirs(out_dir,exist_ok=True)

        print (f"System {name} has the following attriburtes : {sorted(system.__dict__.keys())}")
        print (f"System {name} has the following matrices : {sorted(system._XX_R.keys())}")
        other_prop = sorted(list([ p for p in set(dir(system))-set(system.__dict__.keys()) if not p.startswith("__")] ))
        print (f"System {name} additionaly has the following properties : {other_prop}")
        properties = [p for p in properties + extra_properties if p not in exclude_properties]
        # First save the system data, to produce reference data

        # we save each property as separate file, so that if in future we add more properties, we do not need to
        # rewrite the old files, so that the changes in a PR will be clearly visible
        for key in properties:
            print (f"saving {key}",end="")
            np.savez( os.path.join(out_dir,key+".npz") , getattr(system,key) , allow_pickle=True)
            print (" - Ok!")
        for key in matrices:
            print (f"saving {key}",end="")
            np.savez_compressed( os.path.join(out_dir,key+".npz") , system.get_R_mat(key) )
            print (" - Ok!")


        def check_property(key,prec,XX=False):
            print (f"checking {key} prec={prec} XX={XX}", end="")
            data_ref = np.load( os.path.join(REF_DIR,"systems",  name, key+".npz"), allow_pickle=True )['arr_0']
            if XX:
                data = system.get_R_mat(key)
            else:
                data = getattr(system,key)
            data=np.array(data)
            if data.dtype==bool:
                data=np.array(data,dtype=int)
                data_ref=np.array(data_ref,dtype=int)
            if hasattr(data_ref,'shape'):
                assert data.shape == data_ref.shape, f"{key} has the wrong shape {data.shape}, should be {data_ref.shape}"
            if prec<0:
                req_precision = -prec*( abs(data_ref) )
            else:
                req_precision = prec
            if not data==pytest.approx(data_ref):
                diff = abs(data-data_ref).max()
                raise ValueError(
                                    f"matrix elements {key} for system {name} give an "
                                    f"absolute difference of {diff} greater than the required precision {req_precision}\n"
                                    f"the missed elements are : \n"+
                                    "\n".join ("{i} | system.iRvec[i[2]] | {mat[i]} | {mat_ref[i]} | {abs(mat[i]-mat_ref[i])}"
                                            for i in zip(np.where(abs(data-data_ref)>req_precision)) )+"\n\n"
                                )
            print (" - Ok!")

        for key in properties:
            check_property(key,precision_properties,XX=False)
        for key in matrices:
            check_property(key,precision_matrix_elements,XX=True)

    return _inner



def test_system_Fe_W90(check_system, system_Fe_W90):
    check_system(system_Fe_W90,"Fe_W90", extra_properties=['wannier_centers_cart_auto','mp_grid']+properties_wcc, matrices=['Ham','AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA'])

def test_system_Fe_W90_wcc(check_system, system_Fe_W90_wcc):
    check_system(system_Fe_W90_wcc,"Fe_W90_wcc", extra_properties=['wannier_centers_cart_auto','mp_grid']+properties_wcc, matrices=['Ham','AA', 'BB', 'CC', 'SS'])

def test_system_Fe_W90_sparse(check_system, system_Fe_W90_sparse):
    check_system(system_Fe_W90_sparse,"Fe_W90_sparse",matrices=['Ham','AA', 'BB', 'CC', 'SS', 'SR', 'SH', 'SHR', 'SA', 'SHA'])


"""
def system_Fe_W90_wcc(create_files_Fe_W90):
def system_Fe_sym_W90(create_files_Fe_W90):
def system_Fe_W90_proj_set_spin(create_files_Fe_W90):
def system_Fe_W90_proj(create_files_Fe_W90):
def system_GaAs_W90(create_files_GaAs_W90):
def system_GaAs_W90_wcc(create_files_GaAs_W90):
def system_GaAs_tb():
def system_GaAs_sym_tb():
def system_GaAs_tb_wcc():
def system_GaAs_tb_wcc_ws():
def system_Haldane_TBmodels():
def system_Haldane_TBmodels_internal():
def system_Haldane_PythTB():
def system_Chiral_left():
def system_Chiral_left_TR():
def system_Chiral_right():
def system_Fe_FPLO():
def system_Fe_FPLO_wcc():
def system_CuMnAs_2d_broken():
def system_Te_ASE(data_Te_ASE):
def system_Te_ASE_wcc(data_Te_ASE):
def system_Te_sparse():
def system_Phonons_Si():
def system_Phonons_GaAs():
def system_Mn3Sn_sym_tb():
def system_kp_mass_iso_0():
def system_kp_mass_iso_1():
def system_kp_mass_iso_2():
def system_kp_mass_aniso_0():
def system_kp_mass_aniso_1():
def system_kp_mass_aniso_2():
"""
