import os
from matplotlib import pyplot as plt
import numpy as np
import wannierberri as wb
from wannierberri.system.system_R import System_R
from wannierberri.grid import Path
from wannierberri.evaluate_k import evaluate_k_path
from wannierberri.w90files.soc import SOC
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.chk import CheckPoint as CHK
from wannierberri.parallel import Parallel, Serial
from irrep.spacegroup import SpaceGroup
from irrep.bandstructure import BandStructure


bandstructure = BandStructure(code="gpaw",
                              calculator_gpaw="Fe-gs.gpw",
                              onlysym=True,)
sg = bandstructure.spacegroup
mg = SpaceGroup.from_cell(real_lattice=sg.real_lattice, positions=sg.positions, typat=sg.typat,
                          magmom=[[0, 0, 1]])
mg.show()



system_dw = System_R().load_npz("system_dw", load_all_XX_R=True)
system_up = System_R().load_npz("system_up", load_all_XX_R=True)
system_spinor = System_R().load_npz("system_spinor", load_all_XX_R=True)
system_spinor.set_spin_pairs([[2 * i, 2 * i + 1] for i in range(9)])
system_spinor.set_pointgroup(spacegroup=mg)
# system_spinor.set_pointgroup([])
# system_up.set_pointgroup([])
# system_dw.set_pointgroup([])

# print(system_spinor.pointgroup)

parallel = Parallel(num_cpus=24)
# _interlaced()


phi_deg = 0
theta_deg = 0


soc = SOC.from_gpaw("Fe-nscf.gpw")
chk_up = CHK.from_npz("system_up.chk.npz")
chk_dw = CHK.from_npz("system_dw.chk.npz")
system_soc = SystemSOC(system_up=system_up, system_down=system_dw,)
system_soc.set_soc_R(soc, chk_up=chk_up, chk_down=chk_dw,
                     theta=theta_deg / 180 * np.pi,
                     phi=phi_deg / 180 * np.pi,
                     alpha_soc=1.0)

system_soc.set_pointgroup(spacegroup=mg)
# system_soc.set_pointgroup(symmetry_gen=[])  # disable symmetry for SOC
# print(system_soc.pointgroup)
# exit()

grid = wb.grid.Grid(system_spinor, NK=200)

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


results = {}
os.makedirs("results", exist_ok=True)
# for name in ["up", "dw"]:
for name in ["soc", "spinor"]:
    system = {"up": system_up,
              "dw": system_dw,
              "soc": system_soc,
              "spinor": system_spinor}[name]


    print(f"Running {name}...")
    results[name] = wb.run(system,
           grid=grid,
           parallel=parallel,
           fout_name=f"results/{name}",
           calculators=calculators,
           adpt_num_iter=100,
           restart=False,
           print_progress_step=5,
           )


ahc_spinor = results["spinor"].results["ahc_int"].data
ahc_soc = results["soc"].results["ahc_int"].data

plt.plot(Efermi - EF, ahc_spinor[:,2]/100, label="spinor")
plt.plot(Efermi - EF, ahc_soc[:,2]/100, label="soc")
plt.legend()
plt.xlabel("E-EF (eV)")
plt.ylabel("AHC (S/cm)")
plt.savefig("ahc.png", dpi=300)