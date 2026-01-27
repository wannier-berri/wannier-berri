import copy

import numpy as np
import wannierberri as wberri
from .common import OUTPUT_DIR, REF_DIR


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


def test_system_interpolate_soc(check_system, system_Fe_gpaw_soc_z):
    system_Fe_gpaw_soc_minusz = copy.deepcopy(system_Fe_gpaw_soc_z)
    system_Fe_gpaw_soc_minusz.swap_spin_channels()
    system_Fe_gpaw_soc_minusz.set_soc_axis(theta=0, phi=0.0, alpha_soc=1.0)
    check_system(
        system_Fe_gpaw_soc_minusz, "Fe_gpaw_soc_-z",
        matrices=['Ham_SOC', 'SS', 'overlap_up_down', 'dV_soc_wann_0_0', 'dV_soc_wann_0_1', 'dV_soc_wann_1_1'],
        properties=['num_wann', 'real_lattice', 'periodic', 'is_phonon', 'wannier_centers_cart', 'iRvec'],
    )
    interpolator = wberri.system.interpolate.SystemInterpolatorSOC(system0=system_Fe_gpaw_soc_minusz,
                                                                   system1=system_Fe_gpaw_soc_z)
    calculators = {}
    Efermi = np.linspace(0, 18, 10)
    calculators["cumdos"] = wberri.calculators.static.CumDOS(tetra=True, Efermi=Efermi)
    calculators["dos"] = wberri.calculators.static.DOS(tetra=True, Efermi=Efermi)
    calculators["ahc"] = wberri.calculators.static.Spin(tetra=True, Efermi=Efermi)
    calculators["spin"] = wberri.calculators.static.Spin(tetra=True, Efermi=Efermi)
    results = {}

    grid = wberri.grid.Grid(system_Fe_gpaw_soc_z, NK=4)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        system_interp = interpolator.interpolate(alpha)
        check_system(
            system_interp, f"Fe_gpaw_soc_interp_alpha_{alpha:.2f}",
            matrices=['Ham_SOC', 'SS', 'overlap_up_down', 'dV_soc_wann_0_0', 'dV_soc_wann_0_1', 'dV_soc_wann_1_1'],
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
                assert err < 1e-8, f"SOC Interp alpha={alpha:.2f} quant={quant} failed with err={err}"

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
            assert err < 1e-8, f"SOC Interp alpha={compare_pares[0]} vs {compare_pares[1]} quant={quant} failed with err={err}"
