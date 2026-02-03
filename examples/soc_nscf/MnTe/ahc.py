import numpy as np
import wannierberri as wb
from wannierberri.system.system_R import System_R
from wannierberri.w90files.soc import SOC
from wannierberri.system.system_soc import SystemSOC
from wannierberri.w90files.chk import CheckPoint as CHK
from irrep.spacegroup import SpaceGroup
from wannierberri.symmetry.point_symmetry import PointGroup


a = 4.134
c = 6.652

lattice = a * np.array([[1, 0, 0], [-1 / 2, np.sqrt(3) / 2, 0], [0, 0, c / a]])
positions = np.array(
    [
        [0, 0, 0],
        [0, 0, 1 / 2],
        [1 / 3, 2 / 3, 1 / 4],
        [2 / 3, 1 / 3, 3 / 4],
    ]
)
typat = [1, 1, 2, 2]
magmom = [[0, 1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]]

mg = SpaceGroup.from_cell(real_lattice=lattice, positions=positions, typat=typat,
                          magmom=magmom)

pointgroup = PointGroup(spacegroup=mg)


system_dw = System_R.from_npz("system_dw")
system_up = System_R.from_npz("system_up")


wb.ray_init()

phi_deg = 90
theta_deg = 90


soc = SOC.from_gpaw("mnte-nscf.gpw")
chk_up = CHK.from_npz("system_up.chk.npz")
chk_dw = CHK.from_npz("system_dw.chk.npz")
system_soc = SystemSOC(system_up=system_up, system_down=system_dw,)
system_soc.set_soc_R(soc, chk_up=chk_up, chk_down=chk_dw,
                     theta=theta_deg / 180 * np.pi,
                     phi=phi_deg / 180 * np.pi,
                     alpha_soc=1.0)

system_soc.set_pointgroup(spacegroup=mg)
# system_soc.set_pointgroup(symmetry_gen=[])  # disable symmetry for SOC
print(system_soc.pointgroup)
# exit()

grid = wb.grid.Grid(system_soc, NK=400)

EF = 6.7

Efermi = np.linspace(EF - 2, EF + 0.1, 1001)
calculators = {}

tetra = False
# calculators["dos"] = wb.calculators.static.DOS(Efermi=Efermi, tetra=tetra)
# calculators["spin"] = wb.calculators.static.Spin(Efermi=Efermi, tetra=tetra)
# calculators["cumdos"] = wb.calculators.static.CumDOS(Efermi=Efermi, tetra=tetra)
# calculators["ohmic_surf"] = wb.calculators.static.Ohmic_FermiSurf(Efermi=Efermi, tetra=tetra)
# # calculators["ohmic_sea"] = wb.calculators.static.Ohmic_FermiSea(Efermi=Efermi, tetra=tetra)
# calculators["ahc"] = wb.calculators.static.AHC(Efermi=Efermi,tetra=tetra)
calculators["ahc_int"] = wb.calculators.static.AHC(Efermi=Efermi, tetra=tetra, kwargs_formula={"external_terms": False})
# calculators["ahc_ext"] = wb.calculators.static.AHC(Efermi=Efermi,tetra=tetra, kwargs_formula={"internal_terms":False})
wb.ray_init()

# for name in ["up", "dw"]:
for name in ["soc"]:
    system = {"up": system_up,
              "dw": system_dw,
              "soc": system_soc}[name]


    print(f"Running {name}...")
    wb.run(system,
           grid=grid,
           fout_name=f"results/{name}",
           calculators=calculators,
           adpt_num_iter=100,
           restart=False,
           print_progress_step_time=5,
           )
