import copy

import numpy as np
import wannierberri as wberri
from .common import OUTPUT_DIR, REF_DIR
import pytest

from wannierberri.system import SystemInterpolator, SystemInterpolatorSOC


@pytest.mark.parametrize("check_numbers", [2, 3, 4])
def test_system_Fe_sym_W90_interpolate(check_system, system_Fe_sym_W90,
                                       system_Fe_sym_W90_TR, check_numbers):


    if check_numbers == 2:
        interpolator = SystemInterpolator(systems_list=[system_Fe_sym_W90, system_Fe_sym_W90_TR])
        system_Fe_sym_W90_interpolate = interpolator.interpolate([0.6, 0.4])
    elif check_numbers == 3:
        interpolator = SystemInterpolator(systems_list=[system_Fe_sym_W90, system_Fe_sym_W90, system_Fe_sym_W90_TR])
        system_Fe_sym_W90_interpolate = interpolator.interpolate([0.3, 0.3, 0.4])
    elif check_numbers == 4:
        interpolator = SystemInterpolator(systems_list=[system_Fe_sym_W90, system_Fe_sym_W90, system_Fe_sym_W90_TR, system_Fe_sym_W90_TR])
        system_Fe_sym_W90_interpolate = interpolator.interpolate([0.3, 0.3, 0.2, 0.2])


    check_system(
        system_Fe_sym_W90_interpolate, "Fe_sym_W90_interpolate_04",
        matrices=['Ham', 'AA', 'BB', 'CC', 'SS'],
        sort_iR=True,
        legacy=False,
    )


@pytest.mark.parametrize("check_numbers", [2, 3, 4])
def test_system_interpolate_soc(check_system, system_Fe_gpaw_soc_z, check_numbers):
    system_Fe_gpaw_soc_minusz = copy.deepcopy(system_Fe_gpaw_soc_z)
    system_Fe_gpaw_soc_minusz.swap_spin_channels()
    system_Fe_gpaw_soc_minusz.set_soc_axis(theta=0, phi=0.0, alpha_soc=1.0)
    check_system(
        system_Fe_gpaw_soc_minusz, "Fe_gpaw_soc_swapped",
        matrices=['overlap_up_down', 'dV_soc'],
        properties=['num_wann', 'real_lattice', 'periodic', 'is_phonon', 'wannier_centers_cart', 'iRvec'],
    )
    if check_numbers == 2:
        interpolator = SystemInterpolatorSOC(systems_list=[system_Fe_gpaw_soc_minusz, system_Fe_gpaw_soc_z])
    elif check_numbers == 3:
        interpolator = SystemInterpolatorSOC(systems_list=[system_Fe_gpaw_soc_minusz, system_Fe_gpaw_soc_z, system_Fe_gpaw_soc_z])
    elif check_numbers == 4:
        interpolator = SystemInterpolatorSOC(systems_list=[system_Fe_gpaw_soc_minusz, system_Fe_gpaw_soc_z, system_Fe_gpaw_soc_z, system_Fe_gpaw_soc_minusz])
    calculators = {}
    Efermi = np.linspace(0, 18, 10)
    from wannierberri.calculators.static import CumDOS, DOS, Spin
    calculators["cumdos"] = CumDOS(tetra=True, Efermi=Efermi)
    calculators["dos"] = DOS(tetra=True, Efermi=Efermi)
    calculators["ahc"] = Spin(tetra=True, Efermi=Efermi)
    calculators["spin"] = Spin(tetra=True, Efermi=Efermi)
    results = {}

    grid = wberri.grid.Grid(system=system_Fe_gpaw_soc_z, NK=4)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if check_numbers == 2:
            system_interp = interpolator.interpolate([1 - alpha, alpha])
        elif check_numbers == 3:
            system_interp = interpolator.interpolate([(1 - alpha), 0.3 * alpha, 0.7 * alpha])
        elif check_numbers == 4:
            system_interp = interpolator.interpolate([0.2 * (1 - alpha), 0.4 * alpha, 0.6 * alpha, 0.8 * (1 - alpha)])
        check_system(
            system_interp, f"Fe_gpaw_soc_interp_alpha_{alpha:.2f}",
            matrices=['overlap_up_down', 'dV_soc'],
            properties=['num_wann', 'real_lattice', 'periodic', 'is_phonon', 'wannier_centers_cart', 'iRvec'],
        )
        for ud in ["up", "down"]:
            check_system(
                getattr(system_interp, f'system_{ud}'), f"Fe_gpaw_soc_interp_alpha_{alpha:.2f}/system_{ud}",
                matrices=['Ham'],
                properties=['num_wann', 'real_lattice', 'periodic', 'is_phonon', 'wannier_centers_cart', 'iRvec'],
            )

        results[f"{alpha:.2f}"] = wberri.run(system=system_interp, grid=grid,
                    fout_name=f"{OUTPUT_DIR}/integrate_files/test_soc_interp_alpha_{alpha:.2f}",
            calculators=calculators)
        if alpha in [0.5, 0.75, 1.0]:
            for quant in calculators.keys():
                res = results[f"{alpha:.2f}"].results[quant].data
                ref = np.load(f"{REF_DIR}/integrate_files/test_soc_interp_alpha_{alpha:.2f}-{quant}_iter-0000.npz")["data"]
                err = np.max(np.abs(res - ref))
                assert err < 5e-7, f"SOC Interp alpha={alpha:.2f} quant={quant} failed with err={err}"

    results["-z"] = wberri.run(system=system_Fe_gpaw_soc_minusz, grid=grid,
                    fout_name=f"{OUTPUT_DIR}/integrate_files/test_soc_interp_alpha_-z",
                   calculators=calculators)
    results["+z"] = wberri.run(system=system_Fe_gpaw_soc_z, grid=grid,
                    fout_name=f"{OUTPUT_DIR}/integrate_files/test_soc_interp_alpha_+z",
                   calculators=calculators)

    for compare_pares in [("0.50", "0.50", 0), ("+z", "1.00", 1), ("0.25", "0.75", -1),]:
        for quant in calculators.keys():
            if quant in ["spin", "ahc"]:
                sign = compare_pares[2]
            else:
                sign = 1
            res = results[compare_pares[0]].results[quant].data * sign
            ref = np.load(f"{REF_DIR}/integrate_files/test_soc_interp_alpha_{compare_pares[1]}-{quant}_iter-0000.npz")["data"]
            err = np.max(np.abs(res - ref))
            assert err < 1e-7, f"SOC Interp alpha={compare_pares[0]} vs {compare_pares[1]} quant={quant} failed with err={err}"
