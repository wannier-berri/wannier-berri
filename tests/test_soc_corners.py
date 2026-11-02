"""Energies at the corners of the cells used by the tetrahedron method, for systems with spin-orbit coupling"""

import copy

import numpy as np
import pytest

import wannierberri as wberri
from wannierberri.data_K import get_data_k_class_from_system
from wannierberri.fourier.rvectors import Rvectors


def reorder_down(system):
    """the same system, with the R vectors of the spin-down part in another order, so that they differ from those of
    the spin-up part"""
    system = copy.deepcopy(system)
    down = system.system_down
    order = np.random.default_rng(0).permutation(down.rvec.nRvec)
    rvec = down.rvec
    down.rvec = Rvectors(lattice=rvec.lattice, shifts_left_red=rvec.shifts_left_red,
                         shifts_right_red=rvec.shifts_right_red, iRvec=rvec.iRvec[order])
    for key in list(down._XX_R):
        down._XX_R[key] = down._XX_R[key][order]
    return system


@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("cell", ["tetrahedron", "parallelepiped"])
def test_soc_corners(system_Fe_gpaw_soc, cell, reorder):
    """the energies at the corners equal those of data_K evaluated at the corners, also when the R vectors of the
    spin-up and spin-down parts differ"""
    system = reorder_down(system_Fe_gpaw_soc) if reorder else system_Fe_gpaw_soc
    assert system.nspin == 2
    if cell == "tetrahedron":
        grid = wberri.grid.GridTetra(system, length=10, NKFFT=(3, 3, 3))
        Kpoint = grid.K_list[0]
    else:
        grid = wberri.Grid(system, NKFFT=3, NKdiv=1)
        Kpoint = grid.get_K_list()[0]
    data_K = get_data_k_class_from_system(system)(system, dK=Kpoint.Kp_fullBZ, grid=grid, Kpoint=Kpoint)
    if reorder:  # the reordered system is the same system
        assert not np.all(system.system_down.rvec.iRvec == system_Fe_gpaw_soc.system_down.rvec.iRvec)
        original = get_data_k_class_from_system(system_Fe_gpaw_soc)(system_Fe_gpaw_soc, dK=Kpoint.Kp_fullBZ, grid=grid,
                                                                   Kpoint=Kpoint)
        np.testing.assert_allclose(data_K.E_K, original.E_K, rtol=0, atol=1e-10)
    if cell == "tetrahedron":
        corners, reference = data_K.E_K_corners_tetra(), data_K.E_K_corners_tetra_test()
    else:
        corners, reference = data_K.E_K_corners_parallel(), data_K.E_K_corners_parallel_test()
    np.testing.assert_allclose(corners, reference, rtol=0, atol=1e-8)
