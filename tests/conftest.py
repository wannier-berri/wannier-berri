"""pytest configuration file for WannierBerri tests."""

import pytest
from common import ROOT_DIR, OUTPUT_DIR, REF_DIR

# WannierBerri Systems
from common_systems import (
    create_files_Fe_W90,
    create_files_GaAs_W90,
    create_files_Si_W90,
    create_W90_files,
    system_Fe_W90,
    system_Fe_W90_sparse,
    system_Fe_W90_wcc,
    system_Fe_sym_W90,
    system_Fe_sym_W90_old,
    system_Fe_W90_proj_set_spin,
    system_Fe_W90_proj,
    system_Fe_W90_proj_ws,
    system_Fe_W90_disentangle,
    system_Fe_FPLO,
    system_Fe_FPLO_wcc,
    data_Te_ASE,
    system_Te_ASE,
    system_Te_ASE_wcc,
    system_Te_sparse,
    system_GaAs_W90,
    system_GaAs_W90_wcc,
    system_GaAs_tb,
    system_GaAs_sym_tb,
    system_GaAs_sym_tb_old,
    system_GaAs_tb_wcc,
    system_GaAs_tb_wcc_ws,
    system_Si_W90_JM,
    system_Haldane_PythTB,
    system_KaneMele_odd_PythTB,
    system_Haldane_TBmodels,
    system_Haldane_TBmodels_internal,
    system_Chiral_left,
    system_Chiral_left_TR,
    system_Chiral_right,
    system_CuMnAs_2d_broken,
    system_Phonons_Si,
    system_Phonons_GaAs,
    system_Mn3Sn_sym_tb,
    system_Mn3Sn_sym_tb_old,
    system_kp_mass_iso_0,
    system_kp_mass_iso_1,
    system_kp_mass_iso_2,
    system_kp_mass_aniso_0,
    system_kp_mass_aniso_1,
    system_kp_mass_aniso_2,
)

# Comparers for tests
from common_comparers import (
    compare_any_result,
    compare_energyresult,
    compare_fermisurfer,
    compare_sym_asym,
)


from test_cluster import check_command_output
from test_run import check_run

# Parallel objects
from common_parallel import parallel_serial, parallel_ray


@pytest.fixture(scope="session", autouse=True)
def create_output_dir():
    # Create folder OUTPUT_DIR
    from pathlib import Path
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
