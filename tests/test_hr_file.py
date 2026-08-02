
import os
import shutil

from pytest import approx
import pytest

from wannierberri.system.system_R import System_R

from .common import OUTPUT_DIR, ROOT_DIR


@pytest.mark.parametrize("format", ["hr", "tb"])
def test_write_read_hr(system_GaAs_W90, format):
    seedname = os.path.join(OUTPUT_DIR, "GaAs")
    if format == "hr":
        system_GaAs_W90.to_hr_file(seedname=seedname)
        data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")
        shutil.copyfile(os.path.join(data_dir, "GaAs.win"), os.path.join(OUTPUT_DIR, "GaAs.win"))
        system_GaAs_tb2 = System_R.from_hr_file(seedname=seedname)
    elif format == "tb":
        system_GaAs_W90.to_tb_file(seedname=seedname)
        system_GaAs_tb2 = System_R.from_tb_file(seedname=seedname)
    assert system_GaAs_W90.get_R_mat('Ham') == approx(system_GaAs_tb2.get_R_mat('Ham'))
    assert system_GaAs_W90.rvec.iRvec == approx(system_GaAs_tb2.rvec.iRvec)
    assert system_GaAs_W90.real_lattice == approx(system_GaAs_tb2.real_lattice)
    assert system_GaAs_W90.wannier_centers_cart == approx(system_GaAs_tb2.wannier_centers_cart)
    assert system_GaAs_W90.spinor == approx(system_GaAs_tb2.spinor)
