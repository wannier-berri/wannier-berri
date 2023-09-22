"""Test tab_plot.py"""
from wannierberri.utils import tab_plot
from wannierberri import calculators as calc


def test_Fe(check_run, system_Fe_W90):
    param_tab = {'degen_thresh': 5e-2}
    calculators = {}
    calculators["tabulate"] = calc.TabulatorAll (
            {
                "Energy": calc.tabulate.Energy(),  # yes, in old implementation degen_thresh was applied to qunatities,
                # but not to energies
                "V": calc.tabulate.Velocity(**param_tab),
                "berry": calc.tabulate.BerryCurvature(**param_tab),
                'spin': calc.tabulate.Spin(**param_tab),
            },
            ibands=[5, 6, 7, 8],
            save_mode="" )

    result = check_run(
        system_Fe_W90,
        calculators,
        fout_name="berry_Fe_W90",
        suffix="tab_plot",
        parameters_K={
            '_FF_antisym': True,
            '_CCab_antisym': True
        },
        extra_precision={"Morb": -1e-6},
        do_not_compare=True )

    tab_result = result.results.get("tabulate")

    tab_plot.main(["None","type=Line","quantity=berry"], tab_result=tab_result)
    tab_plot.main(["None","type=Plane","quantity=berry","vec1=1,1,0","vec2=1,2,0"], tab_result=tab_result)

