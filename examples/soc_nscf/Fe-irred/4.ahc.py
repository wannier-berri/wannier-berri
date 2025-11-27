import os
import numpy as np
import wannierberri as wb
from wannierberri.system.system_soc import SystemSOC

wb.ray_init()

system_soc = SystemSOC.from_npz("system_soc")
theta = 0
phi = 0
system_soc.set_soc_axis(theta=theta, phi=phi, units="degrees")


grid = wb.grid.Grid(system_soc, NK=200)

EF = 9.22085

Efermi = np.linspace(EF - 0.5, EF + 0.5, 1001)
calculators = {}

tetra = False
# calculators["dos"] = wb.calculators.static.DOS(Efermi=Efermi, tetra=tetra)
# # calculators["spin"] = wb.calculators.static.Spin(Efermi=Efermi, tetra=tetra)
# calculators["cumdos"] = wb.calculators.static.CumDOS(Efermi=Efermi, tetra=tetra)
# calculators["ohmic_surf"] = wb.calculators.static.Ohmic_FermiSurf(Efermi=Efermi, tetra=tetra)
# calculators["ohmic_sea"] = wb.calculators.static.Ohmic_FermiSea(Efermi=Efermi, tetra=tetra)
# calculators["ahc"] = wb.calculators.static.AHC(Efermi=Efermi, tetra=tetra)
calculators["ahc_int"] = wb.calculators.static.AHC(Efermi=Efermi, tetra=tetra, kwargs_formula={"external_terms": False})
# calculators["ahc_ext"] = wb.calculators.static.AHC(Efermi=Efermi, tetra=tetra, kwargs_formula={"internal_terms": False})


os.makedirs("results", exist_ok=True)
result_int = wb.run(system_soc,
        grid=grid,
        fout_name=f"results/Fe-irred-soc-ahc-th{theta}-phi{phi}",
        calculators={"ahc_int": wb.calculators.static.AHC(Efermi=Efermi, tetra=tetra, kwargs_formula={"external_terms": False})},
        adpt_num_iter=100,
        restart=False,
        print_progress_step_time=5,
        )

ahc_int = result_int.results["ahc_int"].data



grid = wb.grid.Grid(system_soc, NK=100)
result_ext = wb.run(system_soc,
        grid=grid,
        fout_name=f"results/Fe-irred-soc-ahc-th{theta}-phi{phi}",
        calculators={"ahc_ext": wb.calculators.static.AHC(Efermi=Efermi, tetra=tetra, kwargs_formula={"internal_terms": False})},
        adpt_num_iter=50,
        restart=False,
        print_progress_step_time=5,
        )
ahc_ext = result_ext.results["ahc_ext"].data
