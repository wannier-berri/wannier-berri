import copy
import wannierberri as wberri

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
    interpolator = wberri.system.interpolate.SystemInterpolatorSOC(system0=system_Fe_gpaw_soc_minusz,
                                                                   system1=system_Fe_gpaw_soc_z)
    system_Fe_gpaw_soc_interpolate = interpolator.interpolate(0.3)
    